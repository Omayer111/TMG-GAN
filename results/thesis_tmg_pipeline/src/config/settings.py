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

    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    checkpoint_interval: int = 5
    use_amp: bool = True
    eval_interval: int = 1
    compile_model: bool = False
    fast_mode: bool = False

    augmentation_cap: int | None = None
    augmentation_target_mode: str = "second_max"
    max_synthetic_multiplier: float | None = 1.5
    strict_qualification_fallback: bool = False
    robust_rng_restore: bool = False

    clf_class_weighting: str = "none"
    clf_effective_num_beta: float = 0.9999
    clf_label_smoothing: float = 0.0
    clf_lr_patience: int = 3
    clf_lr_decay: float = 0.5
    clf_min_lr: float = 1e-5
    clf_early_stop_patience: int = 0
    max_fallback_rate: float = 0.05
    pretrain_cd_epochs: int = 30
    pretrain_cd_lr: float = 1e-3
    pretrain_cd_only: bool = False
    pretrain_eval_interval: int = 5
    load_pretrained_cd: str = ""
    save_pretrained_cd_only: bool = False

    # ---- NEW: minority-class-focused CD pretraining knobs ----
    pretrain_loss: str = "cb_focal"      # one of: ce, focal, cb_focal
    focal_gamma: float = 2.0
    use_balanced_sampler: bool = True

    # ---- NEW: slightly stronger CD backbone ----
    cd_dropout: float = 0.15
    embedding_dim: int = 128
    cd_freeze_backbone_epochs: int = 10
    g_steps_c2: int = 2
    g_steps_c3: int = 3
    g_steps_c4: int = 4
    g_steps_c5: int = 4

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.dataset_name / self.run_name

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics" / self.dataset_name
