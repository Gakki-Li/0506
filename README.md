# Brain MRI Segmentation Server Guide

## Files

- `unet.py`: completed PyTorch U-Net baseline
- `train.py`: CLI training script
- `evaluate.py`: CLI evaluation / visualization script
- `ablation_registry.py`: ablation experiment definitions
- `run_experiments.py`: batch runner for baseline, ablations, and combined improvements
- `run_experiments.sh`: shell wrapper around `run_experiments.py`
- `summarize_experiments.py`: build ablation tables and training-curve plots
- `outputs/`: default output directory

## Recommended server environment

Install PyTorch with the CUDA command from the official PyTorch site first, then install the remaining packages:

```bash
pip install medpy scikit-image pillow tqdm matplotlib tensorboard
```

`torchvision` is not required by the current scripts.

## Dataset path

The scripts expect the raw patient folders under a path like:

```text
/path/to/BrainMRI/kaggle_3m
```

If your server path is different, pass it with `--images` or `IMAGES_DIR=...`.

## Full server run

From `Project_1_0506/brain-seg/pytorch`:

```bash
bash run_experiments.sh
```

The default runner now executes:

- `baseline`
- `bn`
- `bilinear`
- `dropout`
- `bce_dice`
- `adamw`
- `cosine`
- `strong_aug`
- `improved`

The shared default configuration is aligned with `train.ipynb` baseline:

- `IMAGE_SIZE=256`
- `EPOCHS=100`
- `STEPS_PER_EPOCH=0` for full epochs
- `BATCH_SIZE=16`
- `EVAL_BATCH_SIZE=16`
- `INIT_FEATURES=32`
- `VALIDATION_CASES=10`
- `VIS_IMAGES=200`
- `VIS_FREQ=10`

All of them can be overridden with environment variables:

```bash
IMAGE_SIZE=256 BATCH_SIZE=8 EPOCHS=30 WORKERS=8 bash run_experiments.sh
```

You can also run only a subset of experiments:

```bash
EXPERIMENTS=baseline,bn,bilinear,improved bash run_experiments.sh
```

If you need to resume after partial completion:

```bash
SKIP_EXISTING=1 bash run_experiments.sh
```

## Manual commands

Baseline:

```bash
python train.py \
  --images /path/to/BrainMRI/kaggle_3m \
  --output-root ./outputs \
  --experiment-name baseline \
  --epochs 100 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --image-size 256 \
  --validation-cases 10 \
  --init-features 32 \
  --optimizer adam \
  --loss dice \
  --lr 3e-4 \
  --vis-images 200 \
  --vis-freq 10 \
  --workers 4
```

Single ablation example:

```bash
python train.py \
  --images /path/to/BrainMRI/kaggle_3m \
  --output-root ./outputs \
  --experiment-name bn \
  --epochs 100 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --image-size 256 \
  --validation-cases 10 \
  --init-features 32 \
  --optimizer adam \
  --loss dice \
  --lr 3e-4 \
  --batch-norm \
  --vis-images 200 \
  --vis-freq 10 \
  --workers 4
```

Combined improved:

```bash
python train.py \
  --images /path/to/BrainMRI/kaggle_3m \
  --output-root ./outputs \
  --experiment-name improved \
  --epochs 100 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --image-size 256 \
  --validation-cases 10 \
  --init-features 32 \
  --optimizer adamw \
  --loss bce_dice \
  --bce-weight 0.4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --batch-norm \
  --bilinear \
  --dropout 0.1 \
  --aug-scale 0.10 \
  --aug-angle 20 \
  --vis-images 200 \
  --vis-freq 10 \
  --workers 4
```

Evaluation:

```bash
python evaluate.py \
  --images /path/to/BrainMRI/kaggle_3m \
  --experiment-dir ./outputs/baseline \
  --batch-size 16 \
  --workers 4
```

## Outputs

For each experiment directory:

- `config.json`: training configuration
- `history.csv`: epoch metrics
- `metrics.json`: best validation summary
- `weights/best.pt`: best checkpoint
- `evaluation/metrics.json`: final evaluation summary
- `evaluation/dsc_distribution.png`: patient-level DSC plot
- `evaluation/best_*.png`, `median_*.png`, `worst_*.png`: overlay examples

After all experiments finish, the runner also creates:

- `outputs/summary/ablation_results.csv`
- `outputs/summary/ablation_results.md`
- `outputs/summary/ablation_results.json`
- `outputs/summary/train_loss_curves.png`
- `outputs/summary/valid_loss_curves.png`
- `outputs/summary/valid_dsc_curves.png`
- `outputs/summary/best_dsc_bar.png`

## Suggested workflow for your report

1. Run `bash run_experiments.sh`.
2. Read the ablation table from `outputs/summary/ablation_results.csv`.
3. Insert the training curves from `outputs/summary/*.png`.
4. Use representative overlays from each experiment's `evaluation/` directory when needed.

## If GPU memory is limited

Reduce one or more of these:

- `BATCH_SIZE`
- `IMAGE_SIZE`
- `INIT_FEATURES`

Example:

```bash
IMAGE_SIZE=96 BATCH_SIZE=4 INIT_FEATURES=16 bash run_experiments.sh
```
