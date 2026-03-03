import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import context
from thesis_tmg_pipeline.src import (
    CheckpointManager,
    DNNClassifier,
    ExperimentConfig,
    compute_metrics,
    load_dataset,
    set_random_state,
)


def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
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


def train(config: ExperimentConfig, resume: bool) -> None:
    set_random_state(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(config.dataset_name, config.data_root, config.cache_dir)

    train_dl = DataLoader(data["train_dataset"], batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_dl = DataLoader(data["test_dataset"], batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = DNNClassifier(data["input_dim"], data["num_classes"], config.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    manager = CheckpointManager(config.checkpoint_dir)
    start_epoch = 0
    best_f1 = -1.0

    if resume:
        checkpoint = manager.load_latest(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["epoch"] + 1)
            best_f1 = float(checkpoint["best_f1"])
            CheckpointManager.restore_rng_state(checkpoint["rng_state"])
            print(f"Resumed from epoch {start_epoch}")

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_dir / f"{config.run_name}.json"

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        test_loss, test_metrics = evaluate(model, test_dl, device)
        is_best = test_metrics["F1"] > best_f1
        if is_best:
            best_f1 = test_metrics["F1"]

        should_save = ((epoch + 1) % config.checkpoint_interval == 0) or is_best or (epoch + 1 == config.epochs)
        if should_save:
            manager.save(
                payload={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_f1": best_f1,
                    "rng_state": CheckpointManager.snapshot_rng_state(),
                    "config": vars(config),
                },
                epoch=epoch + 1,
                is_best=is_best,
            )

        report = {
            "epoch": epoch + 1,
            "dataset": config.dataset_name,
            "run_name": config.run_name,
            "loss": test_loss,
            **test_metrics,
            "best_f1": best_f1,
            "checkpoint_dir": str(config.checkpoint_dir),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Loss {test_loss:.4f} | "
            f"P {test_metrics['Precision']:.4f} | "
            f"R {test_metrics['Recall']:.4f} | "
            f"F1 {test_metrics['F1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resumable DNN baseline for CICIDS2017/UNSW-NB15")
    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="dnn_baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
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
        checkpoint_interval=args.checkpoint_interval,
        use_amp=not args.no_amp,
    )
    train(cfg, resume=args.resume)
