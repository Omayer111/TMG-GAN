# Thesis TMG Pipeline (Isolated)

This folder is fully separate from the original repository and is designed for thesis-grade reproducible experiments.

## Current implemented slice

- Resumable DNN baseline training for `CICIDS2017` and `UNSW-NB15`
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
  src/
    checkpointing.py
    utils.py
    config/settings.py
    data/tabular_loader.py
    models/dnn.py
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
```

## Resume behavior

- Interrupt training any time.
- Re-run the same command with `--resume`.
- Training continues from the latest checkpoint epoch, not from epoch 0.

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

- Add tabular TMG-GAN modules with identical checkpoint/resume logic
- Add ADASYN baseline and unified benchmark runner
- Add ablation toggles and multi-seed experiment orchestration
