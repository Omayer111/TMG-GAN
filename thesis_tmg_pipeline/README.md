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
  --diversity-loss-weight 0.1 \
  --max-rejects 10 \
  --epochs 200 \
  --batch-size 1024 \
  --lr 0.001 \
  --eval-interval 5 \
  --checkpoint-interval 5 \
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
