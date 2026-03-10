import argparse
import json
from pathlib import Path

import context


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bar(value: float, width: int = 30) -> str:
    if not (0.0 <= value <= 1.0):
        return "-" * width
    fill = int(round(value * width))
    return "#" * fill + "." * (width - fill)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one run metrics JSON as an ASCII dashboard.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--run-name", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = args.output_dir / "metrics" / args.dataset / f"{args.run_name}.json"
    if not metrics_path.exists():
        print(f"[metrics] file not found: {metrics_path}")
        return 1

    with open(metrics_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    metric_names = ["Precision", "Recall", "F1", "Accuracy"]
    print(f"[metrics] dataset={report.get('dataset', args.dataset)} run={report.get('run_name', args.run_name)}")
    print(f"[metrics] phase={report.get('phase')} epoch={report.get('epoch')} best_f1={report.get('best_f1')}")

    for name in metric_names:
        value = _safe_float(report.get(name))
        print(f"{name:>10}: {value:7.4f} |{_bar(value)}|")

    before = report.get("class_counts_before")
    after = report.get("class_counts_after")
    if before is not None and after is not None:
        print(f"[metrics] class_counts_before={before}")
        print(f"[metrics] class_counts_after={after}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
