import os
import json
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


PAPER_CLASS_ORDER = [
    "Benign",
    "DoS",
    "Port Scan",
    "Brute Force",
    "Web Attack",
    "Bot",
]

PAPER_CLASS_TO_IDX = {name: i for i, name in enumerate(PAPER_CLASS_ORDER)}


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_raw_csvs(input_paths: list[str], low_memory: bool = True) -> pd.DataFrame:
    dfs = []
    for p in input_paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(".csv"):
                    dfs.append(pd.read_csv(os.path.join(p, fn), low_memory=low_memory))
        else:
            dfs.append(pd.read_csv(p, low_memory=low_memory))
    if not dfs:
        raise FileNotFoundError("No CSVs found in provided --input paths.")
    return pd.concat(dfs, axis=0, ignore_index=True)


def normalize_label(label: str) -> str:
    s = str(label).strip()

    # common canonicalization
    s_low = s.lower()

    # benign / normal
    if s_low in {"normal traffic", "benign", "normal"}:
        return "Benign"

    # dos family (merge DoS + DDoS to match paper-style 6 classes)
    if s_low in {"dos", "ddos"}:
        return "DoS"

    # port scan
    if s_low in {"port scanning", "port scan", "portscan"}:
        return "Port Scan"

    # brute force
    if s_low in {"brute force", "bruteforce", "ftp-patator", "ssh-patator"}:
        return "Brute Force"

    # web attack
    if s_low in {
        "web attacks",
        "web attack",
        "webattacks",
        "web attack - brute force",
        "web attack - sql injection",
        "web attack - xss",
    }:
        return "Web Attack"

    # bot
    if s_low in {"bots", "bot"}:
        return "Bot"

    raise ValueError(f"Unknown raw label encountered: {label!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--dataset-name", default="CICIDS2017_paper6")
    ap.add_argument("--label-col", default="Attack Type")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nan-policy", choices=["drop", "zero", "median"], default="drop")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_root, args.dataset_name)
    safe_makedirs(out_dir)

    x_train_path = os.path.join(out_dir, "x_train.csv")
    y_train_path = os.path.join(out_dir, "y_train.csv")
    x_test_path = os.path.join(out_dir, "x_test.csv")
    y_test_path = os.path.join(out_dir, "y_test.csv")
    meta_path = os.path.join(out_dir, "preprocess_meta.json")

    if args.resume and all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path, meta_path]):
        print("All outputs already exist; --resume enabled, skipping.")
        return

    print("Loading raw CSV(s)...")
    df = load_raw_csvs(args.input)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found. First columns: {list(df.columns)[:20]}")

    print("Cleaning: drop duplicates, replace inf with NaN...")
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)

    # map raw labels to paper-style 6 classes
    print("Mapping raw labels to paper-style 6 classes...")
    raw_labels = df[args.label_col].astype(str)
    mapped_labels = raw_labels.map(normalize_label)

    print("Mapped class distribution:")
    print(mapped_labels.value_counts())

    X = df.drop(columns=[args.label_col])
    y = mapped_labels

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"Found non-numeric columns, applying one-hot encoding: {non_numeric}")
        X = pd.get_dummies(X, columns=non_numeric, drop_first=False)

    if X.isna().any().any():
        print(f"NaNs detected. Applying nan policy: {args.nan_policy}")
        if args.nan_policy == "drop":
            keep = ~X.isna().any(axis=1)
            X = X.loc[keep].reset_index(drop=True)
            y = y.loc[keep].reset_index(drop=True)
        elif args.nan_policy == "zero":
            X = X.fillna(0.0)
        elif args.nan_policy == "median":
            X = X.fillna(X.median(numeric_only=True))

    print("Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("MinMax scaling to [0,1] (fit on train only)...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    # fixed paper-style mapping, not alphabetical auto-mapping
    y_train_idx = y_train.map(PAPER_CLASS_TO_IDX).astype(int)
    y_test_idx = y_test.map(PAPER_CLASS_TO_IDX).astype(int)

    print("Writing outputs...")
    atomic_write_csv(X_train_scaled, x_train_path)
    atomic_write_csv(pd.DataFrame({"label": y_train_idx}), y_train_path)
    atomic_write_csv(X_test_scaled, x_test_path)
    atomic_write_csv(pd.DataFrame({"label": y_test_idx}), y_test_path)

    meta = {
        "dataset_name": args.dataset_name,
        "label_col": args.label_col,
        "test_size": args.test_size,
        "seed": args.seed,
        "nan_policy": args.nan_policy,
        "n_train": int(len(X_train_scaled)),
        "n_test": int(len(X_test_scaled)),
        "n_features": int(X_train_scaled.shape[1]),
        "classes": PAPER_CLASS_TO_IDX,
        "paper_class_order": PAPER_CLASS_ORDER,
        "note": "Paper-style 6-class mapping used: Benign, DoS(merged DoS+DDoS), Port Scan, Brute Force, Web Attack, Bot."
    }
    atomic_write_json(meta, meta_path)

    print("DONE. Saved to:", out_dir)
    print("Shapes:")
    print("  x_train:", X_train_scaled.shape)
    print("  x_test :", X_test_scaled.shape)
    print("Class mapping:", PAPER_CLASS_TO_IDX)


if __name__ == "__main__":
    main()
