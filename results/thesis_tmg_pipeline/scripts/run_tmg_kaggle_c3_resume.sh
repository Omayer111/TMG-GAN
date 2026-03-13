#!/usr/bin/env bash
set -euo pipefail

# C3: Resume validation run from best C1/C2 lineage
REPO_DIR=${REPO_DIR:-/kaggle/working/thesis_tmg_pipeline}
DATA_ROOT=${DATA_ROOT:-/kaggle/working/datasets}
OUT_DIR=${OUT_DIR:-/kaggle/working/outputs_tmg}
CACHE_DIR=${CACHE_DIR:-/kaggle/working/data_cache}
RUN_NAME=${RUN_NAME:-tmg_c2_weighted}

LOG_DIR="$OUT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}_resume.log"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

python -u scripts/train_tmg_gan.py \
  --dataset CICIDS2017 \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --run-name "$RUN_NAME" \
  --gan-epochs 300 \
  --gan-lr 0.0002 \
  --z-dim 64 \
  --gan-hidden-dim 256 \
  --cd-steps 1 \
  --g-steps 1 \
  --gen-batch-size 2048 \
  --hidden-warmup-epochs 100 \
  --hidden-loss-weight 1.0 \
  --max-rejects 10 \
  --epochs 240 \
  --batch-size 512 \
  --lr 0.0003 \
  --eval-interval 5 \
  --checkpoint-interval 1 \
  --gan-eval-interval 100 \
  --augmentation-cap 300000 \
  --augmentation-target-mode second_max \
  --max-synthetic-multiplier 1.5 \
  --max-fallback-rate 0.15 \
  --clf-class-weighting effective_num \
  --clf-effective-num-beta 0.9999 \
  --clf-label-smoothing 0.02 \
  --clf-lr-patience 5 \
  --clf-lr-decay 0.5 \
  --clf-min-lr 3e-5 \
  --clf-early-stop-patience 20 \
  --robust-rng-restore \
  --reset-clf-optimizer-on-resume \
  --resume \
  --no-amp 2>&1 | tee -a "$LOG_FILE"
