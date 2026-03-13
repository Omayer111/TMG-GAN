# Thesis TMG Pipeline (Isolated)

This folder is fully separate from the original repository and is designed for thesis-grade reproducible experiments.

## Current implemented slice

- Resumable DNN baseline training for `CICIDS2017` and `UNSW-NB15`
- Resumable ADASYN + DNN baseline training for `CICIDS2017` and `UNSW-NB15`
- Resumable tabular SNGAN baseline training for `CICIDS2017` and `UNSW-NB15`
- Resumable tabular TACGAN-IDS baseline training for `CICIDS2017` and `UNSW-NB15`
- Resumable tabular TMG-GAN baseline training for `CICIDS2017` and `UNSW-NB15`
- Automatic checkpointing (`latest.pt`, `best.pt`, `epoch_XXXX.pt`)
- Resume support after interruption (`--resume`)
- Deterministic seed setup and metric export (Precision, Recall, F1, Accuracy)
- Cached tensor dataset loading

## Directory layout

```
thesis_tmg_pipeline/
  scripts/
    context.py
    train_dnn.py
    train_adasyn_dnn.py
    train_sngan_tabular.py
    train_tacgan_ids.py
    train_tmg_gan.py
  src/
    checkpointing.py
    utils.py
    config/settings.py
    data/tabular_loader.py
    data/resampling.py
    models/dnn.py
    models/sngan_generator_tabular.py
    models/sngan_discriminator_tabular.py
    models/tacgan_generator_tabular.py
    models/tacgan_discriminator_tabular.py
    models/tmg_generator_tabular.py
    models/tmg_cd_model_tabular.py
  tests/
    test_resume_smoke.py
  outputs/
    checkpoints/
    metrics/
  data_cache/
```

## Data format (required)

Place each dataset under your chosen `--data-root` like this:

```
<data-root>/
  CICIDS2017/
    x_train.csv
    y_train.csv
    x_test.csv
    y_test.csv
  UNSW-NB15/
    x_train.csv
    y_train.csv
    x_test.csv
    y_test.csv
```

- `x_*`: numeric feature matrix
- `y_*`: either class index column or one-hot encoded columns

## Local run

From `thesis_tmg_pipeline`:

```powershell
pip install -r requirements.txt
python scripts/train_dnn.py --dataset UNSW-NB15 --data-root C:/path/to/datasets --epochs 100 --resume

python scripts/train_adasyn_dnn.py --dataset UNSW-NB15 --data-root C:/path/to/datasets --epochs 100 --resume

python scripts/train_sngan_tabular.py --dataset UNSW-NB15 --data-root C:/path/to/datasets --gan-epochs 20 --epochs 20 --resume

python scripts/train_tacgan_ids.py --dataset UNSW-NB15 --data-root C:/path/to/datasets --gan-epochs 20 --epochs 20 --resume

python scripts/train_tmg_gan.py --dataset UNSW-NB15 --data-root C:/path/to/datasets --gan-epochs 20 --epochs 20 --resume
```

## Resume behavior

- Interrupt training any time.
- Re-run the same command with `--resume`.
- Training continues from the latest checkpoint epoch, not from epoch 0.

## Robust run protocol (recommended)

Keep your existing model-specific commands unchanged, but execute them through a logging wrapper so every run has a persistent log file.

### Naming convention

- Use a stable `--run-name` for each experiment if you want resume continuity.
- Keep `--output-dir` stable for that run.
- Save logs under `<output-dir>/logs/<run-name>.log`.

### Kaggle/Colab bash wrapper (recommended)

```bash
set -euo pipefail

RUN_NAME=tmg_gan_tabular
OUT_DIR=/kaggle/working/outputs_tmg
LOG_DIR=$OUT_DIR/logs
LOG_FILE=$LOG_DIR/${RUN_NAME}.log

mkdir -p "$LOG_DIR"

python -u scripts/train_tmg_gan.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir "$OUT_DIR" \
  --cache-dir /kaggle/working/data_cache \
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
  --augmentation-cap 300000 \
  --max-rejects 10 \
  --epochs 200 \
  --batch-size 1024 \
  --lr 0.001 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
  --robust-rng-restore \
  --reset-clf-optimizer-on-resume \
  --resume 2>&1 | tee -a "$LOG_FILE"
```

### TMG recovery protocol scripts (Kaggle)

The repository now includes three Kaggle-oriented scripts to run the conservative recovery matrix directly:

- `scripts/run_tmg_kaggle_c1.sh`: fresh conservative run (quality-first synthetic augmentation)
- `scripts/run_tmg_kaggle_c2.sh`: fresh conservative run + weighted classifier fine-tuning
- `scripts/run_tmg_kaggle_c3_resume.sh`: resume validation run from best C1/C2 lineage

Run order:

1. Run C1 first.
2. Run C2 with a unique `RUN_NAME`.
3. Resume only the best lineage with C3.

Example usage in Kaggle:

```bash
cd /kaggle/working/thesis_tmg_pipeline

bash scripts/run_tmg_kaggle_c1.sh
RUN_NAME=tmg_c2_weighted_v2 bash scripts/run_tmg_kaggle_c2.sh
RUN_NAME=tmg_c2_weighted_v2 bash scripts/run_tmg_kaggle_c3_resume.sh
```

All three scripts enforce:

- conservative target sizing: `--augmentation-target-mode second_max`
- bounded synthetic ratio: `--max-synthetic-multiplier 1.5`
- hard quality gating: `--strict-qualification-fallback --max-fallback-rate 0.05`

### Windows PowerShell wrapper (local)

```powershell
$RunName = "tmg_gan_tabular"
$OutDir = "C:/path/to/outputs_tmg"
$LogDir = "$OutDir/logs"
$LogFile = "$LogDir/$RunName.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

python -u scripts/train_tmg_gan.py `
  --dataset CICIDS2017 `
  --data-root C:/path/to/datasets `
  --output-dir $OutDir `
  --cache-dir C:/path/to/data_cache `
  --run-name $RunName `
  --gan-epochs 300 `
  --gan-lr 0.0002 `
  --z-dim 64 `
  --gan-hidden-dim 256 `
  --cd-steps 1 `
  --g-steps 1 `
  --gen-batch-size 2048 `
  --hidden-warmup-epochs 100 `
  --hidden-loss-weight 1.0 `
  --augmentation-cap 300000 `
  --max-rejects 10 `
  --epochs 200 `
  --batch-size 1024 `
  --lr 0.001 `
  --eval-interval 5 `
  --checkpoint-interval 5 `
  --robust-rng-restore `
  --reset-clf-optimizer-on-resume `
  --resume 2>&1 | Tee-Object -FilePath $LogFile -Append
```

### Monitor run status anytime

```bash
tail -n 80 /kaggle/working/outputs_tmg/logs/tmg_gan_tabular.log
cat /kaggle/working/outputs_tmg/metrics/CICIDS2017/tmg_gan_tabular.json
ls -lh /kaggle/working/outputs_tmg/checkpoints/CICIDS2017/tmg_gan_tabular
```

### Safe stop and resume checklist

1. Near session end, optionally lower risk of lost progress with `--checkpoint-interval 1`.
2. Stop run after a checkpoint boundary when possible.
3. Save notebook version with output files in Kaggle.
4. Re-run with same `--run-name`, `--output-dir`, and `--resume`.
5. If divergence occurs, copy `best.pt` to `latest.pt` and resume.

## Kaggle (recommended for long free-tier runs)

1. Upload prepared CSV dataset as a Kaggle Dataset.
2. In notebook/script, set `--data-root /kaggle/input/<your-dataset-name>`.
3. Set outputs to `/kaggle/working/outputs` and copy checkpoints to a Kaggle Dataset artifact at the end of session.
4. Use moderate batch size (e.g. 128 or 256) and keep checkpoint interval small (5-10 epochs).

Example command:

```bash
python scripts/train_dnn.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs \
  --cache-dir /kaggle/working/data_cache \
  --epochs 2000 \
  --batch-size 256 \
  --checkpoint-interval 5 \
  --resume
```

ADASYN baseline command:

```bash
python scripts/train_adasyn_dnn.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs \
  --cache-dir /kaggle/working/data_cache \
  --run-name adasyn_dnn \
  --epochs 2000 \
  --batch-size 256 \
  --adasyn-neighbors 5 \
  --checkpoint-interval 5 \
  --resume
```

Tabular SNGAN baseline command:

```bash
python scripts/train_sngan_tabular.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs_sngan \
  --cache-dir /kaggle/working/data_cache \
  --run-name sngan_tabular \
  --gan-epochs 300 \
  --gan-lr 0.0002 \
  --z-dim 64 \
  --gan-hidden-dim 256 \
  --batch-size 1024 \
  --gen-batch-size 2048 \
  --epochs 200 \
  --hidden-dim 1024 \
  --lr 0.001 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
  --resume
```

Tabular TACGAN-IDS baseline command:

```bash
python scripts/train_tacgan_ids.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs_tacgan \
  --cache-dir /kaggle/working/data_cache \
  --run-name tacgan_ids \
  --gan-epochs 300 \
  --gan-lr 0.0002 \
  --z-dim 64 \
  --gan-hidden-dim 256 \
  --batch-size 1024 \
  --gen-batch-size 2048 \
  --lambda-cls 1.0 \
  --epochs 200 \
  --hidden-dim 1024 \
  --lr 0.001 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
  --resume
```

Tabular TMG-GAN baseline command:

```bash
python scripts/train_tmg_gan.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs_tmg \
  --cache-dir /kaggle/working/data_cache \
  --run-name tmg_gan_tabular \
  --gan-epochs 300 \
  --gan-lr 0.0002 \
  --z-dim 64 \
  --gan-hidden-dim 256 \
  --cd-steps 1 \
  --g-steps 1 \
  --gen-batch-size 2048 \
  --hidden-warmup-epochs 100 \
  --hidden-loss-weight 1.0 \
  --augmentation-cap 300000 \
  --max-rejects 10 \
  --epochs 200 \
  --batch-size 1024 \
  --lr 0.001 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
  --robust-rng-restore \
  --reset-clf-optimizer-on-resume \
  --resume
```

### Increase GPU utilization on Kaggle

For tabular DNN workloads, GPU utilization is often lower than CV/NLP models; use this command to push utilization higher:

```bash
python scripts/train_dnn.py \
  --dataset CICIDS2017 \
  --data-root /kaggle/input/my-nids-csv \
  --output-dir /kaggle/working/outputs \
  --cache-dir /kaggle/working/data_cache \
  --epochs 2000 \
  --batch-size 4096 \
  --hidden-dim 1024 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
  --compile \
  --fast-mode \
  --resume
```

If out-of-memory occurs, reduce `--batch-size` (4096 -> 2048 -> 1024).

## Colab (best for quick iteration)

1. Mount Google Drive.
2. Put datasets and outputs on Drive so checkpoints persist after disconnect.
3. Use `--output-dir` and `--cache-dir` inside Drive-mounted folder.

Example:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
python scripts/train_dnn.py \
  --dataset UNSW-NB15 \
  --data-root /content/drive/MyDrive/nids_data \
  --output-dir /content/drive/MyDrive/tmg_outputs \
  --cache-dir /content/drive/MyDrive/tmg_cache \
  --epochs 2000 \
  --checkpoint-interval 5 \
  --resume
```

## Next implementation steps

- Add unified benchmark runner (DNN vs ADASYN-DNN vs SNGAN vs TACGAN-IDS vs GAN-family)
- Add ablation toggles and multi-seed experiment orchestration
- Add MAGENTO wrapper and finalize paper-specific hyperparameter alignment
