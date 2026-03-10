import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import context
from thesis_tmg_pipeline.src import (
    CheckpointManager,
    DNNClassifier,
    ExperimentConfig,
    compute_metrics,
    load_dataset,
    set_random_state,
)
from thesis_tmg_pipeline.src.models.sngan_discriminator_tabular import SNGANTabularDiscriminator
from thesis_tmg_pipeline.src.models.sngan_generator_tabular import SNGANTabularGenerator


def evaluate_classifier(model: nn.Module, dl: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * len(y)
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    metrics = compute_metrics(y_true_np, y_pred_np)
    mean_loss = loss_sum / len(dl.dataset)
    return mean_loss, metrics


def build_augmented_dataset(
    generator: SNGANTabularGenerator,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    gen_batch_size: int,
    device: torch.device,
) -> tuple[TensorDataset, list[int], list[int]]:
    class_counts = torch.bincount(y_train, minlength=num_classes)
    target_count = int(class_counts.max().item())

    x_blocks = [x_train]
    y_blocks = [y_train]

    generator.eval()
    with torch.no_grad():
        for class_id in range(num_classes):
            current = int(class_counts[class_id].item())
            need = target_count - current
            while need > 0:
                batch = min(need, gen_batch_size)
                labels = torch.full((batch,), class_id, dtype=torch.long, device=device)
                generated = generator.sample(labels, device=device).cpu()
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
    d_steps: int,
    g_steps: int,
    gen_batch_size: int,
) -> None:
    set_random_state(config.seed, deterministic=not config.fast_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.fast_mode and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data = load_dataset(config.dataset_name, config.data_root, config.cache_dir)

    dl_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    if config.num_workers > 0:
        dl_kwargs["persistent_workers"] = config.persistent_workers
        dl_kwargs["prefetch_factor"] = config.prefetch_factor

    gan_train_dl = DataLoader(data["train_dataset"], shuffle=True, **dl_kwargs)
    test_dl = DataLoader(data["test_dataset"], shuffle=False, **dl_kwargs)

    generator = SNGANTabularGenerator(
        z_dim=z_dim,
        num_classes=data["num_classes"],
        feature_dim=data["input_dim"],
        hidden_dim=gan_hidden_dim,
    ).to(device)
    discriminator = SNGANTabularDiscriminator(
        feature_dim=data["input_dim"],
        num_classes=data["num_classes"],
        hidden_dim=gan_hidden_dim,
    ).to(device)

    classifier = DNNClassifier(
        input_dim=data["input_dim"],
        num_classes=data["num_classes"],
        hidden_dim=config.hidden_dim,
    ).to(device)

    if config.compile_model and hasattr(torch, "compile"):
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)
        classifier = torch.compile(classifier)

    g_optimizer = Adam(generator.parameters(), lr=gan_lr, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=gan_lr, betas=(0.5, 0.999))
    clf_optimizer = Adam(classifier.parameters(), lr=config.learning_rate)

    clf_criterion = nn.CrossEntropyLoss()
    g_scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    d_scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    clf_scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    manager = CheckpointManager(config.checkpoint_dir)
    phase = "gan"
    gan_start_epoch = 0
    clf_start_epoch = 0
    best_f1 = -1.0
    last_d_loss = float("nan")
    last_g_loss = float("nan")

    if resume:
        checkpoint = manager.load_latest(device)
        if checkpoint is not None:
            generator.load_state_dict(checkpoint["generator_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            classifier.load_state_dict(checkpoint["classifier_state_dict"])
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
            clf_optimizer.load_state_dict(checkpoint["clf_optimizer_state_dict"])
            g_scaler.load_state_dict(checkpoint["g_scaler_state_dict"])
            d_scaler.load_state_dict(checkpoint["d_scaler_state_dict"])
            clf_scaler.load_state_dict(checkpoint["clf_scaler_state_dict"])
            phase = checkpoint["phase"]
            gan_start_epoch = int(checkpoint["gan_epoch"] + 1)
            clf_start_epoch = int(checkpoint["clf_epoch"] + 1)
            best_f1 = float(checkpoint["best_f1"])
            last_d_loss = float(checkpoint.get("last_d_loss", float("nan")))
            last_g_loss = float(checkpoint.get("last_g_loss", float("nan")))
            CheckpointManager.restore_rng_state(checkpoint["rng_state"])
            print(
                f"Resumed from phase={phase}, gan_epoch={gan_start_epoch}, clf_epoch={clf_start_epoch}"
            )

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_dir / f"{config.run_name}.json"

    if phase == "gan" and gan_start_epoch < gan_epochs:
        for gan_epoch in range(gan_start_epoch, gan_epochs):
            generator.train()
            discriminator.train()
            d_losses = []
            g_losses = []

            for real_x, real_y in gan_train_dl:
                real_x = real_x.to(device, non_blocking=True)
                real_y = real_y.to(device, non_blocking=True)

                for _ in range(d_steps):
                    d_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        fake_x = generator.sample(real_y, device=device).detach()
                        score_real = discriminator(real_x, real_y)
                        score_fake = discriminator(fake_x, real_y)
                        d_loss = (torch.relu(1.0 - score_real).mean() + torch.relu(1.0 + score_fake).mean())
                    d_scaler.scale(d_loss).backward()
                    d_scaler.step(d_optimizer)
                    d_scaler.update()
                    d_losses.append(float(d_loss.item()))

                for _ in range(g_steps):
                    g_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                        fake_x = generator.sample(real_y, device=device)
                        score_fake = discriminator(fake_x, real_y)
                        g_loss = -score_fake.mean()
                    g_scaler.scale(g_loss).backward()
                    g_scaler.step(g_optimizer)
                    g_scaler.update()
                    g_losses.append(float(g_loss.item()))

            last_d_loss = float(np.mean(d_losses)) if d_losses else float("nan")
            last_g_loss = float(np.mean(g_losses)) if g_losses else float("nan")

            manager.save(
                payload={
                    "phase": "gan",
                    "gan_epoch": gan_epoch,
                    "clf_epoch": -1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "clf_optimizer_state_dict": clf_optimizer.state_dict(),
                    "g_scaler_state_dict": g_scaler.state_dict(),
                    "d_scaler_state_dict": d_scaler.state_dict(),
                    "clf_scaler_state_dict": clf_scaler.state_dict(),
                    "best_f1": best_f1,
                    "last_d_loss": last_d_loss,
                    "last_g_loss": last_g_loss,
                    "rng_state": CheckpointManager.snapshot_rng_state(),
                    "config": vars(config),
                    "gan_config": {
                        "gan_epochs": gan_epochs,
                        "gan_lr": gan_lr,
                        "z_dim": z_dim,
                        "gan_hidden_dim": gan_hidden_dim,
                        "d_steps": d_steps,
                        "g_steps": g_steps,
                        "gen_batch_size": gen_batch_size,
                    },
                },
                epoch=gan_epoch + 1,
                is_best=False,
            )

            print(
                f"GAN Epoch {gan_epoch + 1}/{gan_epochs} | "
                f"D_loss {last_d_loss:.4f} | G_loss {last_g_loss:.4f}"
            )

        phase = "clf"
        clf_start_epoch = 0

    augmented_train_dataset, class_counts_before, class_counts_after = build_augmented_dataset(
        generator=generator,
        x_train=data["x_train"],
        y_train=data["y_train"],
        num_classes=data["num_classes"],
        gen_batch_size=gen_batch_size,
        device=device,
    )
    clf_train_dl = DataLoader(augmented_train_dataset, shuffle=True, **dl_kwargs)

    for clf_epoch in range(clf_start_epoch, config.epochs):
        classifier.train()
        for x, y in clf_train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            clf_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                logits = classifier(x)
                clf_loss = clf_criterion(logits, y)
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
            test_loss, test_metrics = evaluate_classifier(classifier, test_dl, device)
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
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "clf_optimizer_state_dict": clf_optimizer.state_dict(),
                    "g_scaler_state_dict": g_scaler.state_dict(),
                    "d_scaler_state_dict": d_scaler.state_dict(),
                    "clf_scaler_state_dict": clf_scaler.state_dict(),
                    "best_f1": best_f1,
                    "last_d_loss": last_d_loss,
                    "last_g_loss": last_g_loss,
                    "rng_state": CheckpointManager.snapshot_rng_state(),
                    "config": vars(config),
                    "gan_config": {
                        "gan_epochs": gan_epochs,
                        "gan_lr": gan_lr,
                        "z_dim": z_dim,
                        "gan_hidden_dim": gan_hidden_dim,
                        "d_steps": d_steps,
                        "g_steps": g_steps,
                        "gen_batch_size": gen_batch_size,
                    },
                    "class_counts_before": class_counts_before,
                    "class_counts_after": class_counts_after,
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
            "gan_last_d_loss": last_d_loss,
            "gan_last_g_loss": last_g_loss,
            "class_counts_before": class_counts_before,
            "class_counts_after": class_counts_after,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if should_evaluate:
            print(
                f"CLF Epoch {clf_epoch + 1}/{config.epochs} | "
                f"Loss {test_loss:.4f} | "
                f"P {test_metrics['Precision']:.4f} | "
                f"R {test_metrics['Recall']:.4f} | "
                f"F1 {test_metrics['F1']:.4f}"
            )
        else:
            print(f"CLF Epoch {clf_epoch + 1}/{config.epochs} | train step complete (evaluation skipped)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resumable tabular SNGAN baseline for CICIDS2017/UNSW-NB15")

    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="sngan_tabular")

    parser.add_argument("--epochs", type=int, default=100, help="Classifier epochs after GAN augmentation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gan-epochs", type=int, default=200)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--gan-hidden-dim", type=int, default=256)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--g-steps", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=1024)

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
        hidden_dim=args.hidden_dim,
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
        d_steps=args.d_steps,
        g_steps=args.g_steps,
        gen_batch_size=args.gen_batch_size,
    )
