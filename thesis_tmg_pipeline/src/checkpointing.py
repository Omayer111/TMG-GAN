import os
import random
from pathlib import Path

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.checkpoint_dir / "latest.pt"

    def _atomic_save(self, payload: dict, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def save(self, payload: dict, epoch: int, is_best: bool = False) -> None:
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        self._atomic_save(payload, epoch_path)
        self._atomic_save(payload, self.latest_path)
        if is_best:
            self._atomic_save(payload, self.checkpoint_dir / "best.pt")

    def load_latest(self, device: torch.device) -> dict | None:
        if not self.latest_path.exists():
            return None
        return torch.load(self.latest_path, map_location=device, weights_only=False)

    @staticmethod
    def snapshot_rng_state() -> dict:
        payload = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        return payload

    @staticmethod
    def restore_rng_state(payload: dict) -> None:
        random.setstate(payload["python"])
        np.random.set_state(payload["numpy"])
        torch.set_rng_state(payload["torch"])
        if torch.cuda.is_available() and payload["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda"])
