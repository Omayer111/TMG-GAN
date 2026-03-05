import argparse
import json
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


def evaluate_cd_model(model: TMGGANCDModelTabular, dl: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, class_logits, _ = model(x)
            loss = criterion(class_logits, y)
            loss_sum += loss.item() * len(y)
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.argmax(class_logits, dim=1).cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    metrics = compute_metrics(y_true_np, y_pred_np)
    mean_loss = loss_sum / len(dl.dataset)
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
) -> torch.Tensor:
    accepted = []
    rejects_left = max_rejects

    generator.eval()
    cd_model.eval()

    with torch.no_grad():
        while len(accepted) < num:
            candidate = generator.sample(1, device=device)
            _, class_logits, _ = cd_model(candidate)
            predicted = int(torch.argmax(class_logits, dim=1).item())
            if predicted == class_id or rejects_left <= 0:
                accepted.append(candidate.cpu())
                rejects_left = max_rejects
            else:
                rejects_left -= 1

    return torch.cat(accepted, dim=0)


def build_augmented_dataset(
    generators: list[TMGGANGeneratorTabular],
    cd_model: TMGGANCDModelTabular,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    gen_batch_size: int,
    device: torch.device,
    max_rejects: int,
) -> tuple[TensorDataset, list[int], list[int]]:
    class_counts = torch.bincount(y_train, minlength=num_classes)
    target_count = int(class_counts.max().item())

    x_blocks = [x_train]
    y_blocks = [y_train]

    for class_id in range(num_classes):
        current = int(class_counts[class_id].item())
        need = target_count - current
        while need > 0:
            batch = min(need, gen_batch_size)
            generated = generate_qualified_samples(
                generator=generators[class_id],
                cd_model=cd_model,
                class_id=class_id,
                num=batch,
                device=device,
                max_rejects=max_rejects,
            )
            x_blocks.append(generated)
            y_blocks.append(torch.full((batch,), class_id, dtype=torch.long))
            need -= batch

    x_aug = torch.cat(x_blocks, dim=0)
    y_aug = torch.cat(y_blocks, dim=0)
    class_counts_after = torch.bincount(y_aug, minlength=num_classes).cpu().numpy().astype(int).tolist()

    return TensorDataset(x_aug, y_aug), class_counts.cpu().numpy().astype(int).tolist(), class_counts_after


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
    diversity_loss_weight: float,
    max_rejects: int,
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
            cd_model.load_state_dict(checkpoint["cd_model_state_dict"])
            for i, state_dict in enumerate(checkpoint["generator_state_dicts"]):
                generators[i].load_state_dict(state_dict)
            cd_optimizer.load_state_dict(checkpoint["cd_optimizer_state_dict"])
            for i, state_dict in enumerate(checkpoint["g_optimizer_state_dicts"]):
                g_optimizers[i].load_state_dict(state_dict)
            cd_scaler.load_state_dict(checkpoint["cd_scaler_state_dict"])
            for i, state_dict in enumerate(checkpoint["g_scaler_state_dicts"]):
                g_scalers[i].load_state_dict(state_dict)
            if "clf_optimizer_state_dict" in checkpoint:
                clf_optimizer.load_state_dict(checkpoint["clf_optimizer_state_dict"])
            if "clf_scaler_state_dict" in checkpoint:
                clf_scaler.load_state_dict(checkpoint["clf_scaler_state_dict"])

            phase = checkpoint["phase"]
            gan_start_epoch = int(checkpoint["gan_epoch"] + 1)
            clf_start_epoch = int(checkpoint["clf_epoch"] + 1)
            best_f1 = float(checkpoint["best_f1"])
            last_cd_loss = float(checkpoint.get("last_cd_loss", float("nan")))
            last_g_loss = float(checkpoint.get("last_g_loss", float("nan")))
            CheckpointManager.restore_rng_state(checkpoint["rng_state"])
            print(f"Resumed from phase={phase}, gan_epoch={gan_start_epoch}, clf_epoch={clf_start_epoch}")

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_dir / f"{config.run_name}.json"

    if phase == "gan" and gan_start_epoch < gan_epochs:
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

                for _ in range(cd_steps):
                    cd_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        real_batch = sample_real_batch(samples_per_class[class_id], config.batch_size, device)
                        score_real, pred_real, _ = cd_model(real_batch)

                        fake_batch = generators[class_id].sample(config.batch_size, device=device).detach()
                        score_fake, _, _ = cd_model(fake_batch)

                        d_loss = 0.5 * (score_fake.mean() - score_real.mean())
                        c_loss = ce(pred_real, class_target)
                        cd_loss = d_loss + c_loss

                    cd_scaler.scale(cd_loss).backward()
                    cd_scaler.step(cd_optimizer)
                    cd_scaler.update()
                    cd_losses.append(float(cd_loss.item()))

                for _ in range(g_steps):
                    g_optimizers[class_id].zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        real_batch = sample_real_batch(samples_per_class[class_id], config.batch_size, device)
                        _, _, hidden_real = cd_model(real_batch)

                        fake_batch = generators[class_id].sample(config.batch_size, device=device)
                        score_fake, pred_fake, hidden_fake = cd_model(fake_batch)

                        label_loss = ce(pred_fake, class_target)
                        hidden_loss = -cosine_similarity(hidden_real, hidden_fake, dim=1).mean()
                        if gan_epoch < hidden_warmup_epochs:
                            hidden_loss = torch.zeros(1, device=device, dtype=score_fake.dtype).squeeze(0)

                        g_loss = -score_fake.mean() + label_loss + hidden_loss_weight * hidden_loss

                    g_scalers[class_id].scale(g_loss).backward()
                    g_scalers[class_id].step(g_optimizers[class_id])
                    g_scalers[class_id].update()
                    g_losses.append(float(g_loss.item()))

            if diversity_loss_weight > 0:
                for opt in g_optimizers:
                    opt.zero_grad(set_to_none=True)
                hidden_vectors = []
                for gen in generators:
                    _ = gen.sample(8, device=device)
                    hidden_vectors.append(gen.hidden_status.mean(dim=0))

                diversity_terms = []
                for i in range(len(hidden_vectors)):
                    for j in range(i + 1, len(hidden_vectors)):
                        diversity_terms.append(
                            cosine_similarity(
                                hidden_vectors[i].unsqueeze(0),
                                hidden_vectors[j].unsqueeze(0),
                                dim=1,
                            ).mean()
                        )

                if diversity_terms:
                    diversity_loss = diversity_loss_weight * torch.stack(diversity_terms).mean()
                    diversity_loss.backward()
                    for optimizer in g_optimizers:
                        optimizer.step()

            last_cd_loss = float(np.mean(cd_losses)) if cd_losses else float("nan")
            last_g_loss = float(np.mean(g_losses)) if g_losses else float("nan")

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
                        "diversity_loss_weight": diversity_loss_weight,
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

    augmented_train_dataset, class_counts_before, class_counts_after = build_augmented_dataset(
        generators=generators,
        cd_model=cd_model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        num_classes=num_classes,
        gen_batch_size=gen_batch_size,
        device=device,
        max_rejects=max_rejects,
    )
    clf_train_dl = DataLoader(augmented_train_dataset, shuffle=True, **dl_kwargs)

    clf_criterion = nn.CrossEntropyLoss()

    for clf_epoch in range(clf_start_epoch, config.epochs):
        cd_model.train()
        for x, y in clf_train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            clf_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                _, class_logits, _ = cd_model(x)
                clf_loss = clf_criterion(class_logits, y)
            clf_scaler.scale(clf_loss).backward()
            clf_scaler.step(clf_optimizer)
            clf_scaler.update()

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
            is_best = test_metrics["F1"] > best_f1
            if is_best:
                best_f1 = test_metrics["F1"]

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
                        "diversity_loss_weight": diversity_loss_weight,
                        "max_rejects": max_rejects,
                    },
                },
                epoch=clf_epoch + 1,
                is_best=is_best,
            )

        report = {
            "phase": "clf",
            "epoch": clf_epoch + 1,
            "dataset": config.dataset_name,
            "run_name": config.run_name,
            "loss": test_loss,
            **test_metrics,
            "best_f1": best_f1,
            "checkpoint_dir": str(config.checkpoint_dir),
            "gan_epochs": gan_epochs,
            "gan_last_cd_loss": last_cd_loss,
            "gan_last_g_loss": last_g_loss,
            "class_counts_before": class_counts_before,
            "class_counts_after": class_counts_after,
            "hidden_warmup_epochs": hidden_warmup_epochs,
            "hidden_loss_weight": hidden_loss_weight,
            "diversity_loss_weight": diversity_loss_weight,
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if should_evaluate:
            print(
                f"TMG CLF Epoch {clf_epoch + 1}/{config.epochs} | "
                f"Loss {test_loss:.4f} | "
                f"P {test_metrics['Precision']:.4f} | "
                f"R {test_metrics['Recall']:.4f} | "
                f"F1 {test_metrics['F1']:.4f}"
            )
        else:
            print(f"TMG CLF Epoch {clf_epoch + 1}/{config.epochs} | train step complete (evaluation skipped)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resumable tabular TMG-GAN baseline for CICIDS2017/UNSW-NB15")

    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="tmg_gan_tabular")

    parser.add_argument("--epochs", type=int, default=100, help="Classifier fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--gan-hidden-dim", type=int, default=256)
    parser.add_argument("--cd-steps", type=int, default=1)
    parser.add_argument("--g-steps", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-warmup-epochs", type=int, default=100)
    parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-rejects", type=int, default=10)

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
        diversity_loss_weight=args.diversity_loss_weight,
        max_rejects=args.max_rejects,
    )
