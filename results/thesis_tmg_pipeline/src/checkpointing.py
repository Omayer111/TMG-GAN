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
    def _to_uint8_cpu_tensor(state) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.to(device="cpu", dtype=torch.uint8).contiguous()
        return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()

    @staticmethod
    def restore_rng_state(payload: dict, robust: bool = False) -> None:
        def _handle_restore_error(name: str, exc: Exception) -> None:
            if robust:
                print(f"WARNING: Failed to restore {name} RNG state ({exc}). Continuing with fresh RNG state.")
                return
            raise exc

        try:
            random.setstate(payload["python"])
        except Exception as exc:
            _handle_restore_error("python", exc)

        try:
            np.random.set_state(payload["numpy"])
        except Exception as exc:
            _handle_restore_error("numpy", exc)

        try:
            torch_state = CheckpointManager._to_uint8_cpu_tensor(payload["torch"])
            torch.set_rng_state(torch_state)
        except Exception as exc:
            _handle_restore_error("torch", exc)

        if torch.cuda.is_available() and payload.get("torch_cuda") is not None:
            try:
                cuda_states = [CheckpointManager._to_uint8_cpu_tensor(s) for s in payload["torch_cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception as exc:
                _handle_restore_error("torch_cuda", exc)
