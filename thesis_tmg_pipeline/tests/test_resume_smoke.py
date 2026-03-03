from pathlib import Path

import torch
from torch.optim import Adam

from thesis_tmg_pipeline.src.checkpointing import CheckpointManager
from thesis_tmg_pipeline.src.models.dnn import DNNClassifier


def test_checkpoint_roundtrip(tmp_path: Path):
    model = DNNClassifier(input_dim=16, num_classes=3, hidden_dim=8)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    manager = CheckpointManager(tmp_path / "checkpoints")
    payload = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": 0.6,
        "rng_state": CheckpointManager.snapshot_rng_state(),
    }
    manager.save(payload, epoch=4, is_best=True)

    restored = manager.load_latest(torch.device("cpu"))
    assert restored is not None
    assert restored["epoch"] == 3
    assert abs(restored["best_f1"] - 0.6) < 1e-9
    assert (tmp_path / "checkpoints" / "best.pt").exists()
