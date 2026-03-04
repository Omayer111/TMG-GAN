from pathlib import Path

import torch

from thesis_tmg_pipeline.src.data.resampling import build_adasyn_dataset


def test_adasyn_resampling_roundtrip(tmp_path: Path):
    x_majority = torch.randn(60, 8)
    y_majority = torch.zeros(60, dtype=torch.long)

    x_minority = torch.randn(8, 8) + 1.5
    y_minority = torch.ones(8, dtype=torch.long)

    x_train = torch.cat([x_majority, x_minority], dim=0)
    y_train = torch.cat([y_majority, y_minority], dim=0)

    payload = build_adasyn_dataset(
        x_train=x_train,
        y_train=y_train,
        cache_dir=tmp_path,
        dataset_name="SMOKE",
        seed=123,
        n_neighbors=5,
    )

    assert payload["train_dataset"] is not None
    assert len(payload["class_counts_after"]) == 2
    assert payload["class_counts_after"][1] >= payload["class_counts_before"][1]

    cached = build_adasyn_dataset(
        x_train=x_train,
        y_train=y_train,
        cache_dir=tmp_path,
        dataset_name="SMOKE",
        seed=123,
        n_neighbors=5,
    )
    assert cached["class_counts_after"] == payload["class_counts_after"]
