import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import context
from thesis_tmg_pipeline.src import CheckpointManager, ExperimentConfig, compute_metrics, load_dataset, set_random_state
from thesis_tmg_pipeline.src.models.tmg_cd_model_tabular import TMGGANCDModelTabular
from thesis_tmg_pipeline.src.models.tmg_generator_tabular import TMGGANGeneratorTabular


def _sanitize_metric_value(value: float, sentinel: float = -1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(sentinel)
    return numeric if math.isfinite(numeric) else float(sentinel)


def _sanitize_metric_map(metrics: dict, sentinel: float = -1.0) -> dict:
    return {k: _sanitize_metric_value(v, sentinel=sentinel) for k, v in metrics.items()}


def _resolve_augmentation_target_count(class_counts: torch.Tensor, mode: str) -> int:
    counts = class_counts.to(dtype=torch.float32)
    if mode == "second_max":
        sorted_counts, _ = torch.sort(counts, descending=True)
        if len(sorted_counts) > 1:
            return int(sorted_counts[1].item())
        return int(sorted_counts[0].item())
    if mode == "median":
        return int(torch.median(counts).item())
    if mode == "p75":
        return int(torch.quantile(counts, 0.75).item())
    return int(counts.max().item())


def _build_class_weights(class_counts_after: list[int], mode: str, effective_num_beta: float) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = torch.tensor(class_counts_after, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)

    if mode == "inverse_freq":
        weights = counts.sum() / (len(counts) * counts)
    else:
        beta = min(max(float(effective_num_beta), 0.0), 0.999999)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)

    return weights / weights.mean()


def evaluate_cd_model(model: TMGGANCDModelTabular, dl: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    valid_samples = 0
    skipped_batches = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, class_logits, _ = model(x)
            if not torch.isfinite(class_logits).all():
                skipped_batches += 1
                continue
            loss = criterion(class_logits, y)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss_sum += loss.item() * len(y)
            valid_samples += len(y)
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.argmax(class_logits, dim=1).cpu().numpy())

    if valid_samples == 0:
        if skipped_batches > 0:
            print(f"WARNING: evaluate_cd_model skipped all {skipped_batches} batches due to non-finite values.")
        return float("nan"), {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "Accuracy": float("nan"),
        }

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    metrics = compute_metrics(y_true_np, y_pred_np)
    mean_loss = loss_sum / valid_samples
    if skipped_batches > 0:
        print(f"WARNING: evaluate_cd_model skipped {skipped_batches} non-finite batches.")
    return mean_loss, metrics


def split_by_class(x_train: torch.Tensor, y_train: torch.Tensor, num_classes: int) -> list[torch.Tensor]:
    samples_per_class = []
    for class_id in range(num_classes):
        class_mask = y_train == class_id
        class_samples = x_train[class_mask]
        if len(class_samples) == 0:
            raise ValueError(f"Class {class_id} has no samples in the training set.")
        samples_per_class.append(class_samples)
    return samples_per_class


def sample_real_batch(class_samples: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    indices = torch.randint(0, len(class_samples), (batch_size,))
    return class_samples[indices].to(device, non_blocking=True)


def generate_qualified_samples(
    generator: TMGGANGeneratorTabular,
    cd_model: TMGGANCDModelTabular,
    class_id: int,
    num: int,
    device: torch.device,
    max_rejects: int,
    strict_qualification_fallback: bool,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Batch-generate samples and filter by classifier prediction (Algorithm 2)."""
    accepted = []
    total_accepted = 0
    consecutive_empty = 0
    generated_candidates = 0
    matched_candidates = 0
    accepted_via_match = 0
    accepted_via_fallback = 0
    fallback_events = 0

    generator.eval()
    cd_model.eval()

    with torch.no_grad():
        while total_accepted < num:
            remaining = num - total_accepted
            gen_count = min(batch_size, remaining * (max_rejects + 1))
            candidates = generator.sample(gen_count, device=device)
            generated_candidates += gen_count
            _, class_logits, _ = cd_model(candidates)
            predicted = torch.argmax(class_logits, dim=1)
            mask = predicted == class_id
            matched_count = int(mask.sum().item())
            matched_candidates += matched_count

            if mask.any():
                matched = candidates[mask].cpu()
                take = min(len(matched), remaining)
                accepted.append(matched[:take])
                total_accepted += take
                accepted_via_match += take
                consecutive_empty = 0
            else:
                consecutive_empty += 1

            # Fallback: if classifier never predicts this class, accept raw samples
            if consecutive_empty >= max_rejects:
                if strict_qualification_fallback:
                    raise RuntimeError(
                        f"Qualification failed for class {class_id}: no accepted samples after {max_rejects} retries."
                    )
                take = min(gen_count, num - total_accepted)
                accepted.append(candidates[:take].cpu())
                total_accepted += take
                accepted_via_fallback += take
                fallback_events += 1
                consecutive_empty = 0

    stats = {
        "requested": int(num),
        "generated_candidates": int(generated_candidates),
        "matched_candidates": int(matched_candidates),
        "accepted_via_match": int(accepted_via_match),
        "accepted_via_fallback": int(accepted_via_fallback),
        "fallback_events": int(fallback_events),
    }

    return torch.cat(accepted, dim=0)[:num], stats


def build_augmented_dataset(
    generators: list[TMGGANGeneratorTabular],
    cd_model: TMGGANCDModelTabular,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    gen_batch_size: int,
    device: torch.device,
    max_rejects: int,
    augmentation_cap: int | None,
    augmentation_target_mode: str,
    max_synthetic_multiplier: float | None,
    strict_qualification_fallback: bool,
) -> tuple[TensorDataset, list[int], list[int], dict[str, dict[str, int | float]]]:
    class_counts = torch.bincount(y_train, minlength=num_classes)
    target_count = _resolve_augmentation_target_count(class_counts, augmentation_target_mode)
    if augmentation_cap is not None:
        target_count = min(target_count, int(augmentation_cap))
        print(f"Applying augmentation cap: target_count={target_count} (cap={augmentation_cap}).")
    print(f"Augmentation target mode='{augmentation_target_mode}' resolved target_count={target_count}.")

    x_blocks = [x_train]
    y_blocks = [y_train]
    qualification_stats: dict[str, dict[str, int | float]] = {}

    for class_id in range(num_classes):
        current = int(class_counts[class_id].item())
        need = target_count - current
        if need > 0 and (max_synthetic_multiplier is not None):
            bounded_need = int(round(current * max_synthetic_multiplier))
            bounded_need = max(0, bounded_need)
            if bounded_need < need:
                print(
                    f"Class {class_id}: limiting synthetic samples from {need} to {bounded_need} "
                    f"(max_synthetic_multiplier={max_synthetic_multiplier})."
                )
            need = min(need, bounded_need)
        if need > 0:
            generated, class_stats = generate_qualified_samples(
                generator=generators[class_id],
                cd_model=cd_model,
                class_id=class_id,
                num=need,
                device=device,
                max_rejects=max_rejects,
                strict_qualification_fallback=strict_qualification_fallback,
                batch_size=gen_batch_size,
            )
            x_blocks.append(generated)
            y_blocks.append(torch.full((len(generated),), class_id, dtype=torch.long))
        else:
            class_stats = {
                "requested": 0,
                "generated_candidates": 0,
                "matched_candidates": 0,
                "accepted_via_match": 0,
                "accepted_via_fallback": 0,
                "fallback_events": 0,
            }

        requested = int(class_stats["requested"])
        fallback_rate = (float(class_stats["accepted_via_fallback"]) / requested) if requested > 0 else 0.0
        class_stats["fallback_rate"] = fallback_rate
        qualification_stats[str(class_id)] = class_stats
        print(
            f"Qualification class {class_id}: requested={requested}, "
            f"accepted_match={class_stats['accepted_via_match']}, "
            f"accepted_fallback={class_stats['accepted_via_fallback']}, "
            f"fallback_rate={fallback_rate:.4f}, events={class_stats['fallback_events']}"
        )

    x_aug = torch.cat(x_blocks, dim=0)
    y_aug = torch.cat(y_blocks, dim=0)
    class_counts_after = torch.bincount(y_aug, minlength=num_classes).cpu().numpy().astype(int).tolist()

    return TensorDataset(x_aug, y_aug), class_counts.cpu().numpy().astype(int).tolist(), class_counts_after, qualification_stats


def train(
    config: ExperimentConfig,
    resume: bool,
    gan_epochs: int,
    gan_lr: float,
    z_dim: int,
    gan_hidden_dim: int,
    cd_steps: int,
    g_steps: int,
    gen_batch_size: int,
    hidden_warmup_epochs: int,
    hidden_loss_weight: float,
    max_rejects: int,
    gan_eval_interval: int,
    max_grad_norm: float,
    reset_clf_optimizer_on_resume: bool,
    augmentation_target_mode: str,
    max_synthetic_multiplier: float | None,
    clf_class_weighting: str,
    clf_effective_num_beta: float,
    clf_label_smoothing: float,
    clf_lr_patience: int,
    clf_lr_decay: float,
    clf_min_lr: float,
    clf_early_stop_patience: int,
    max_fallback_rate: float,
) -> None:
    set_random_state(config.seed, deterministic=not config.fast_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.fast_mode and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data = load_dataset(config.dataset_name, config.data_root, config.cache_dir)
    num_classes = data["num_classes"]
    feature_dim = data["input_dim"]

    dl_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    if config.num_workers > 0:
        dl_kwargs["persistent_workers"] = config.persistent_workers
        dl_kwargs["prefetch_factor"] = config.prefetch_factor

    test_dl = DataLoader(data["test_dataset"], shuffle=False, **dl_kwargs)

    samples_per_class = split_by_class(data["x_train"], data["y_train"], num_classes)

    cd_model = TMGGANCDModelTabular(feature_dim=feature_dim, num_classes=num_classes, hidden_dim=gan_hidden_dim).to(device)
    generators = [
        TMGGANGeneratorTabular(z_dim=z_dim, feature_dim=feature_dim, hidden_dim=gan_hidden_dim).to(device)
        for _ in range(num_classes)
    ]

    if config.compile_model and hasattr(torch, "compile"):
        cd_model = torch.compile(cd_model)
        generators = [torch.compile(g) for g in generators]

    cd_optimizer = Adam(cd_model.parameters(), lr=gan_lr, betas=(0.5, 0.999))
    g_optimizers = [Adam(g.parameters(), lr=gan_lr, betas=(0.5, 0.999)) for g in generators]
    clf_optimizer = Adam(cd_model.parameters(), lr=config.learning_rate)

    ce = nn.CrossEntropyLoss()
    cd_scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    g_scalers = [torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda") for _ in generators]
    clf_scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    manager = CheckpointManager(config.checkpoint_dir)
    phase = "gan"
    gan_start_epoch = 0
    clf_start_epoch = 0
    best_f1 = -1.0
    last_cd_loss = float("nan")
    last_g_loss = float("nan")

    if resume:
        checkpoint = manager.load_latest(device)
        if checkpoint is not None:
            resume_phase = checkpoint["phase"]
            cd_model.load_state_dict(checkpoint["cd_model_state_dict"])
            for i, state_dict in enumerate(checkpoint["generator_state_dicts"]):
                generators[i].load_state_dict(state_dict)
            cd_optimizer.load_state_dict(checkpoint["cd_optimizer_state_dict"])
            for i, state_dict in enumerate(checkpoint["g_optimizer_state_dicts"]):
                g_optimizers[i].load_state_dict(state_dict)
            cd_scaler.load_state_dict(checkpoint["cd_scaler_state_dict"])
            for i, state_dict in enumerate(checkpoint["g_scaler_state_dicts"]):
                g_scalers[i].load_state_dict(state_dict)
            should_reset_clf_state = reset_clf_optimizer_on_resume and resume_phase == "clf"
            if should_reset_clf_state:
                print("Resume mode: resetting classifier optimizer/scaler state for CLF phase.")
            if ("clf_optimizer_state_dict" in checkpoint) and (not should_reset_clf_state):
                clf_optimizer.load_state_dict(checkpoint["clf_optimizer_state_dict"])
            if ("clf_scaler_state_dict" in checkpoint) and (not should_reset_clf_state):
                clf_scaler.load_state_dict(checkpoint["clf_scaler_state_dict"])

            phase = checkpoint["phase"]
            gan_start_epoch = int(checkpoint["gan_epoch"] + 1)
            clf_start_epoch = int(checkpoint["clf_epoch"] + 1)
            best_f1 = float(checkpoint["best_f1"])
            last_cd_loss = float(checkpoint.get("last_cd_loss", float("nan")))
            last_g_loss = float(checkpoint.get("last_g_loss", float("nan")))
            rng_state = checkpoint.get("rng_state")
            if rng_state is not None:
                CheckpointManager.restore_rng_state(rng_state, robust=config.robust_rng_restore)
            elif config.robust_rng_restore:
                print("WARNING: rng_state missing from checkpoint. Continuing with fresh RNG state.")
            print(f"Resumed from phase={phase}, gan_epoch={gan_start_epoch}, clf_epoch={clf_start_epoch}")

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_dir / f"{config.run_name}.json"

    if phase == "gan" and gan_start_epoch < gan_epochs:
        nan_count = 0
        for gan_epoch in range(gan_start_epoch, gan_epochs):
            cd_model.train()
            for gen in generators:
                gen.train()

            cd_losses = []
            g_losses = []

            class_order = list(range(num_classes))
            random.shuffle(class_order)

            for class_id in class_order:
                class_target = torch.full((config.batch_size,), class_id, dtype=torch.long, device=device)

                # --- Train C+D (t_d steps per paper Algorithm 1 lines 2-7) ---
                for _ in range(cd_steps):
                    cd_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        real_batch = sample_real_batch(samples_per_class[class_id], config.batch_size, device)
                        score_real, pred_real, _ = cd_model(real_batch)

                        fake_batch = generators[class_id].sample(config.batch_size, device=device).detach()
                        score_fake, _, _ = cd_model(fake_batch)

                        # Paper Eq: (D(fake) - D(real)) / 2 + CE(pred_real, target)
                        d_loss = 0.5 * (score_fake.mean() - score_real.mean())
                        c_loss = ce(pred_real, class_target)
                        cd_loss = d_loss + c_loss

                    if torch.isfinite(cd_loss):
                        cd_scaler.scale(cd_loss).backward()
                        cd_scaler.unscale_(cd_optimizer)
                        torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_grad_norm)
                        cd_scaler.step(cd_optimizer)
                        cd_scaler.update()
                        cd_losses.append(float(cd_loss.item()))
                        nan_count = 0
                    else:
                        cd_optimizer.zero_grad(set_to_none=True)
                        nan_count += 1

                # --- Train G (t_g steps per paper Algorithm 1 lines 8-13) ---
                for _ in range(g_steps):
                    g_optimizers[class_id].zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        real_batch = sample_real_batch(samples_per_class[class_id], config.batch_size, device)
                        _, _, hidden_real = cd_model(real_batch)

                        fake_batch = generators[class_id].sample(config.batch_size, device=device)
                        score_fake, pred_fake, hidden_fake = cd_model(fake_batch)

                        # Paper: -D(fake) + CE(pred_fake, target) + O_k
                        label_loss = ce(pred_fake, class_target)
                        # Paper Eq. 4: cosine similarity between hidden features
                        if gan_epoch < hidden_warmup_epochs:
                            hidden_loss = torch.zeros(1, device=device, dtype=score_fake.dtype).squeeze(0)
                        else:
                            hidden_loss = -cosine_similarity(hidden_real.detach(), hidden_fake, dim=1).mean()

                        g_loss = -score_fake.mean() + label_loss + hidden_loss_weight * hidden_loss

                    if torch.isfinite(g_loss):
                        g_scalers[class_id].scale(g_loss).backward()
                        g_scalers[class_id].unscale_(g_optimizers[class_id])
                        torch.nn.utils.clip_grad_norm_(generators[class_id].parameters(), max_grad_norm)
                        g_scalers[class_id].step(g_optimizers[class_id])
                        g_scalers[class_id].update()
                        g_losses.append(float(g_loss.item()))
                        nan_count = 0
                    else:
                        g_optimizers[class_id].zero_grad(set_to_none=True)
                        nan_count += 1

            # --- Inter-generator diversity loss (original code lines 79-98) ---
            # Paper: penalize cosine similarity between different generators' hidden states
            # Normalized by feature_dim as in original code
            for opt in g_optimizers:
                opt.zero_grad(set_to_none=True)
            hidden_vectors = []
            for gen in generators:
                _ = gen.sample(3, device=device)
                hidden_vectors.append(gen.hidden_status)

            diversity_terms = []
            for i in range(len(hidden_vectors)):
                for j in range(len(hidden_vectors)):
                    if i != j:
                        diversity_terms.append(
                            cosine_similarity(hidden_vectors[i], hidden_vectors[j], dim=1)
                        )

            if diversity_terms:
                diversity_loss = torch.cat(diversity_terms).mean() / feature_dim
                if torch.isfinite(diversity_loss):
                    diversity_loss.backward()
                    for i_opt, optimizer in enumerate(g_optimizers):
                        torch.nn.utils.clip_grad_norm_(generators[i_opt].parameters(), max_grad_norm)
                        optimizer.step()

            if nan_count > 50:
                print(f"ABORT: {nan_count} consecutive NaN losses detected at GAN epoch {gan_epoch + 1}.")
                break

            last_cd_loss = float(np.mean(cd_losses)) if cd_losses else float("nan")
            last_g_loss = float(np.mean(g_losses)) if g_losses else float("nan")

            # Periodic GAN-phase evaluation
            gan_eval_metrics = {}
            if gan_eval_interval > 0 and ((gan_epoch + 1) % gan_eval_interval == 0):
                eval_loss, gan_eval_metrics = evaluate_cd_model(cd_model, test_dl, device)
                cd_model.train()
                for gen in generators:
                    gen.train()
                print(
                    f"  GAN Eval @ epoch {gan_epoch + 1}: "
                    f"Loss {eval_loss:.4f} | F1 {gan_eval_metrics.get('F1', 0):.4f}"
                )

            manager.save(
                payload={
                    "phase": "gan",
                    "gan_epoch": gan_epoch,
                    "clf_epoch": -1,
                    "cd_model_state_dict": cd_model.state_dict(),
                    "generator_state_dicts": [g.state_dict() for g in generators],
                    "cd_optimizer_state_dict": cd_optimizer.state_dict(),
                    "g_optimizer_state_dicts": [opt.state_dict() for opt in g_optimizers],
                    "clf_optimizer_state_dict": clf_optimizer.state_dict(),
                    "cd_scaler_state_dict": cd_scaler.state_dict(),
                    "g_scaler_state_dicts": [scaler.state_dict() for scaler in g_scalers],
                    "clf_scaler_state_dict": clf_scaler.state_dict(),
                    "best_f1": best_f1,
                    "last_cd_loss": last_cd_loss,
                    "last_g_loss": last_g_loss,
                    "rng_state": CheckpointManager.snapshot_rng_state(),
                    "config": vars(config),
                    "gan_config": {
                        "gan_epochs": gan_epochs,
                        "gan_lr": gan_lr,
                        "z_dim": z_dim,
                        "gan_hidden_dim": gan_hidden_dim,
                        "cd_steps": cd_steps,
                        "g_steps": g_steps,
                        "gen_batch_size": gen_batch_size,
                        "hidden_warmup_epochs": hidden_warmup_epochs,
                        "hidden_loss_weight": hidden_loss_weight,
                        "max_rejects": max_rejects,
                    },
                },
                epoch=gan_epoch + 1,
                is_best=False,
            )

            print(
                f"TMG GAN Epoch {gan_epoch + 1}/{gan_epochs} | "
                f"CD_loss {last_cd_loss:.4f} | G_loss {last_g_loss:.4f}"
            )

        phase = "clf"
        clf_start_epoch = 0

    augmented_train_dataset, class_counts_before, class_counts_after, qualification_stats = build_augmented_dataset(
        generators=generators,
        cd_model=cd_model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        num_classes=num_classes,
        gen_batch_size=gen_batch_size,
        device=device,
        max_rejects=max_rejects,
        augmentation_cap=config.augmentation_cap,
        augmentation_target_mode=augmentation_target_mode,
        max_synthetic_multiplier=max_synthetic_multiplier,
        strict_qualification_fallback=config.strict_qualification_fallback,
    )
    clf_train_dl = DataLoader(augmented_train_dataset, shuffle=True, **dl_kwargs)

    requested_total = sum(int(stats["requested"]) for stats in qualification_stats.values())
    fallback_total = sum(int(stats["accepted_via_fallback"]) for stats in qualification_stats.values())
    fallback_rate_total = (float(fallback_total) / requested_total) if requested_total > 0 else 0.0
    print(
        f"Qualification summary: requested={requested_total}, accepted_fallback={fallback_total}, "
        f"fallback_rate={fallback_rate_total:.4f}"
    )
    if fallback_rate_total > max_fallback_rate:
        raise RuntimeError(
            f"Fallback rate {fallback_rate_total:.4f} exceeded threshold {max_fallback_rate:.4f}. "
            "Generated sample quality is too low for this run configuration."
        )

    class_weights = _build_class_weights(class_counts_after, clf_class_weighting, clf_effective_num_beta)
    if class_weights is not None:
        class_weights = class_weights.to(device=device)
        print(f"Using class-weighted CLF loss mode='{clf_class_weighting}' weights={class_weights.detach().cpu().tolist()}")
    else:
        print("Using unweighted CLF loss.")

    clf_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=clf_label_smoothing)
    clf_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        clf_optimizer,
        mode="max",
        factor=clf_lr_decay,
        patience=clf_lr_patience,
        min_lr=clf_min_lr,
    )
    consecutive_fully_nonfinite_epochs = 0
    epochs_without_improve = 0

    for clf_epoch in range(clf_start_epoch, config.epochs):
        cd_model.train()
        nonfinite_train_steps = 0
        for x, y in clf_train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            clf_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                _, class_logits, _ = cd_model(x)
                clf_loss = clf_criterion(class_logits, y)

            if (not torch.isfinite(class_logits).all()) or (not torch.isfinite(clf_loss)):
                nonfinite_train_steps += 1
                clf_optimizer.zero_grad(set_to_none=True)
                continue

            clf_scaler.scale(clf_loss).backward()
            clf_scaler.unscale_(clf_optimizer)
            torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_grad_norm)
            clf_scaler.step(clf_optimizer)
            clf_scaler.update()

        if nonfinite_train_steps > 0:
            print(
                f"WARNING: CLF epoch {clf_epoch + 1} skipped {nonfinite_train_steps} non-finite training steps."
            )

        if nonfinite_train_steps == len(clf_train_dl):
            consecutive_fully_nonfinite_epochs += 1
        else:
            consecutive_fully_nonfinite_epochs = 0

        if consecutive_fully_nonfinite_epochs >= 3:
            print(
                f"ABORT: {consecutive_fully_nonfinite_epochs} consecutive fully non-finite CLF epochs "
                f"(stopped at epoch {clf_epoch + 1})."
            )
            break

        should_evaluate = ((clf_epoch + 1) % config.eval_interval == 0) or (clf_epoch + 1 == config.epochs)
        test_loss = float("nan")
        test_metrics = {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "Accuracy": float("nan"),
        }

        is_best = False
        if should_evaluate:
            test_loss, test_metrics = evaluate_cd_model(cd_model, test_dl, device)
            f1_value = float(test_metrics["F1"])
            is_best = math.isfinite(f1_value) and (f1_value > best_f1)
            scheduler_signal = f1_value if math.isfinite(f1_value) else -1.0
            clf_lr_scheduler.step(scheduler_signal)
            if is_best:
                best_f1 = f1_value
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

        should_save = ((clf_epoch + 1) % config.checkpoint_interval == 0) or is_best or (clf_epoch + 1 == config.epochs)
        if should_save:
            manager.save(
                payload={
                    "phase": "clf",
                    "gan_epoch": gan_epochs - 1,
                    "clf_epoch": clf_epoch,
                    "cd_model_state_dict": cd_model.state_dict(),
                    "generator_state_dicts": [g.state_dict() for g in generators],
                    "cd_optimizer_state_dict": cd_optimizer.state_dict(),
                    "g_optimizer_state_dicts": [opt.state_dict() for opt in g_optimizers],
                    "clf_optimizer_state_dict": clf_optimizer.state_dict(),
                    "cd_scaler_state_dict": cd_scaler.state_dict(),
                    "g_scaler_state_dicts": [scaler.state_dict() for scaler in g_scalers],
                    "clf_scaler_state_dict": clf_scaler.state_dict(),
                    "best_f1": best_f1,
                    "last_cd_loss": last_cd_loss,
                    "last_g_loss": last_g_loss,
                    "rng_state": CheckpointManager.snapshot_rng_state(),
                    "config": vars(config),
                    "class_counts_before": class_counts_before,
                    "class_counts_after": class_counts_after,
                    "gan_config": {
                        "gan_epochs": gan_epochs,
                        "gan_lr": gan_lr,
                        "z_dim": z_dim,
                        "gan_hidden_dim": gan_hidden_dim,
                        "cd_steps": cd_steps,
                        "g_steps": g_steps,
                        "gen_batch_size": gen_batch_size,
                        "hidden_warmup_epochs": hidden_warmup_epochs,
                        "hidden_loss_weight": hidden_loss_weight,
                        "max_rejects": max_rejects,
                    },
                    "qualification_stats": qualification_stats,
                },
                epoch=clf_epoch + 1,
                is_best=is_best,
            )

        report = {
            "phase": "clf",
            "epoch": clf_epoch + 1,
            "dataset": config.dataset_name,
            "run_name": config.run_name,
            "loss": _sanitize_metric_value(test_loss),
            **_sanitize_metric_map(test_metrics),
            "best_f1": best_f1,
            "checkpoint_dir": str(config.checkpoint_dir),
            "gan_epochs": gan_epochs,
            "gan_last_cd_loss": last_cd_loss,
            "gan_last_g_loss": last_g_loss,
            "class_counts_before": class_counts_before,
            "class_counts_after": class_counts_after,
            "hidden_warmup_epochs": hidden_warmup_epochs,
            "hidden_loss_weight": hidden_loss_weight,
            "augmentation_target_mode": augmentation_target_mode,
            "max_synthetic_multiplier": max_synthetic_multiplier,
            "clf_class_weighting": clf_class_weighting,
            "clf_effective_num_beta": clf_effective_num_beta,
            "clf_label_smoothing": clf_label_smoothing,
            "clf_learning_rate": float(clf_optimizer.param_groups[0]["lr"]),
            "epochs_without_improve": epochs_without_improve,
            "qualification_stats": qualification_stats,
            "fallback_rate_total": fallback_rate_total,
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if should_evaluate:
            print(
                f"TMG CLF Epoch {clf_epoch + 1}/{config.epochs} | "
                f"Loss {test_loss:.4f} | "
                f"P {test_metrics['Precision']:.4f} | "
                f"R {test_metrics['Recall']:.4f} | "
                f"F1 {test_metrics['F1']:.4f} | "
                f"LR {clf_optimizer.param_groups[0]['lr']:.6f}"
            )
        else:
            print(f"TMG CLF Epoch {clf_epoch + 1}/{config.epochs} | train step complete (evaluation skipped)")

        if should_evaluate and (clf_early_stop_patience > 0) and (epochs_without_improve >= clf_early_stop_patience):
            print(
                f"EARLY STOP: no F1 improvement for {epochs_without_improve} evaluated epochs "
                f"(patience={clf_early_stop_patience})."
            )
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resumable tabular TMG-GAN baseline for CICIDS2017/UNSW-NB15")

    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="tmg_gan_tabular")

    parser.add_argument("--epochs", type=int, default=100, help="Classifier fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gan-epochs", type=int, default=2000)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--gan-hidden-dim", type=int, default=512)
    parser.add_argument("--cd-steps", type=int, default=5)
    parser.add_argument("--g-steps", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-warmup-epochs", type=int, default=1000)
    parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-rejects", type=int, default=10)
    parser.add_argument("--gan-eval-interval", type=int, default=100, help="Evaluate CD model every N GAN epochs (0=disable)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--augmentation-cap", type=int, default=None, help="Cap per-class samples after augmentation")
    parser.add_argument("--augmentation-target-mode", type=str, default="second_max", choices=["max", "second_max", "median", "p75"], help="Rule for selecting target class size before synthetic generation")
    parser.add_argument("--max-synthetic-multiplier", type=float, default=1.5, help="Max synthetic samples as a multiple of each class's original count")
    parser.add_argument("--strict-qualification-fallback", action="store_true", help="Fail instead of force-accepting unqualified generated samples")
    parser.add_argument("--robust-rng-restore", action="store_true", help="Skip RNG restore errors when resuming from older/incompatible checkpoints")
    parser.add_argument("--clf-class-weighting", type=str, default="none", choices=["none", "inverse_freq", "effective_num"], help="Class weighting strategy for CLF fine-tuning")
    parser.add_argument("--clf-effective-num-beta", type=float, default=0.9999, help="Beta parameter for effective-number class weighting")
    parser.add_argument("--clf-label-smoothing", type=float, default=0.0, help="Label smoothing for CLF cross-entropy")
    parser.add_argument("--clf-lr-patience", type=int, default=3, help="ReduceLROnPlateau patience in evaluated epochs")
    parser.add_argument("--clf-lr-decay", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--clf-min-lr", type=float, default=1e-5, help="Minimum classifier LR for scheduler")
    parser.add_argument("--clf-early-stop-patience", type=int, default=0, help="Stop CLF phase after N evaluated epochs without F1 improvement (0=disable)")
    parser.add_argument("--max-fallback-rate", type=float, default=0.05, help="Abort run when accepted_via_fallback/requested exceeds this threshold")

    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-clf-optimizer-on-resume", action="store_true", help="When resuming from CLF phase, ignore saved CLF optimizer/scaler state")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.gan_hidden_dim,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        compile_model=args.compile,
        fast_mode=args.fast_mode,
        use_amp=not args.no_amp,
        augmentation_cap=args.augmentation_cap,
        augmentation_target_mode=args.augmentation_target_mode,
        max_synthetic_multiplier=args.max_synthetic_multiplier,
        strict_qualification_fallback=args.strict_qualification_fallback,
        robust_rng_restore=args.robust_rng_restore,
        clf_class_weighting=args.clf_class_weighting,
        clf_effective_num_beta=args.clf_effective_num_beta,
        clf_label_smoothing=args.clf_label_smoothing,
        clf_lr_patience=args.clf_lr_patience,
        clf_lr_decay=args.clf_lr_decay,
        clf_min_lr=args.clf_min_lr,
        clf_early_stop_patience=args.clf_early_stop_patience,
        max_fallback_rate=args.max_fallback_rate,
    )

    train(
        config=cfg,
        resume=args.resume,
        gan_epochs=args.gan_epochs,
        gan_lr=args.gan_lr,
        z_dim=args.z_dim,
        gan_hidden_dim=args.gan_hidden_dim,
        cd_steps=args.cd_steps,
        g_steps=args.g_steps,
        gen_batch_size=args.gen_batch_size,
        hidden_warmup_epochs=args.hidden_warmup_epochs,
        hidden_loss_weight=args.hidden_loss_weight,
        max_rejects=args.max_rejects,
        gan_eval_interval=args.gan_eval_interval,
        max_grad_norm=args.max_grad_norm,
        reset_clf_optimizer_on_resume=args.reset_clf_optimizer_on_resume,
        augmentation_target_mode=args.augmentation_target_mode,
        max_synthetic_multiplier=args.max_synthetic_multiplier,
        clf_class_weighting=args.clf_class_weighting,
        clf_effective_num_beta=args.clf_effective_num_beta,
        clf_label_smoothing=args.clf_label_smoothing,
        clf_lr_patience=args.clf_lr_patience,
        clf_lr_decay=args.clf_lr_decay,
        clf_min_lr=args.clf_min_lr,
        clf_early_stop_patience=args.clf_early_stop_patience,
        max_fallback_rate=args.max_fallback_rate,
    )
