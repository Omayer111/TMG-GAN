import argparse
import os
import platform
import sys
from pathlib import Path

import torch

import context


REQUIRED_DATA_FILES = ("x_train.csv", "y_train.csv", "x_test.csv", "y_test.csv")


def _print_line(label: str, value: str) -> None:
    print(f"[preflight] {label}: {value}")


def _check_dataset_layout(data_root: Path, dataset: str) -> tuple[bool, Path, list[str]]:
    dataset_dir = data_root / dataset
    missing = [name for name in REQUIRED_DATA_FILES if not (dataset_dir / name).exists()]
    return len(missing) == 0, dataset_dir, missing


def _check_writeable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for script-first Colab runs.")
    parser.add_argument("--dataset", required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="tmg_gan_tabular")
    parser.add_argument("--resume", action="store_true", help="Also validate latest checkpoint presence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _print_line("python", sys.version.split()[0])
    _print_line("platform", platform.platform())
    _print_line("cwd", os.getcwd())
    _print_line("torch", torch.__version__)
    _print_line("cuda_available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        _print_line("gpu_count", str(torch.cuda.device_count()))
        _print_line("gpu_name", torch.cuda.get_device_name(0))
    else:
        _print_line("warning", "CUDA not available; run will use CPU.")

    data_ok, dataset_dir, missing = _check_dataset_layout(args.data_root, args.dataset)
    if data_ok:
        _print_line("dataset", f"OK ({dataset_dir})")
    else:
        _print_line("dataset", f"MISSING files in {dataset_dir}: {', '.join(missing)}")

    output_ok = _check_writeable_dir(args.output_dir)
    cache_ok = _check_writeable_dir(args.cache_dir)
    _print_line("output_dir", f"{'OK' if output_ok else 'NOT WRITEABLE'} ({args.output_dir})")
    _print_line("cache_dir", f"{'OK' if cache_ok else 'NOT WRITEABLE'} ({args.cache_dir})")

    if args.resume:
        latest_ckpt = args.output_dir / "checkpoints" / args.dataset / args.run_name / "latest.pt"
        if latest_ckpt.exists():
            _print_line("resume_checkpoint", f"OK ({latest_ckpt})")
        else:
            _print_line("resume_checkpoint", f"MISSING ({latest_ckpt})")
            return 1

    if not data_ok or not output_ok or not cache_ok:
        return 1

    _print_line("status", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
