from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    dataset_name: str
    data_root: Path
    cache_dir: Path
    output_dir: Path
    run_name: str = "dnn_baseline"
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    num_workers: int = 0
    checkpoint_interval: int = 5
    use_amp: bool = True

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.dataset_name / self.run_name

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics" / self.dataset_name
