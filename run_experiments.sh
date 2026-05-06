#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGES_DIR="${IMAGES_DIR:-/opt/tiger/alpha-seed/0506_dwn/brain-seg/kaggle_3m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
INIT_FEATURES="${INIT_FEATURES:-32}"
VALIDATION_CASES="${VALIDATION_CASES:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
WORKERS="${WORKERS:-4}"
VIS_IMAGES="${VIS_IMAGES:-200}"
VIS_FREQ="${VIS_FREQ:-10}"
EXPERIMENTS="${EXPERIMENTS:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_experiments.py"
  --images "$IMAGES_DIR"
  --output-root "$OUTPUT_ROOT"
  --epochs "$EPOCHS"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --validate-every "$VALIDATE_EVERY"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --image-size "$IMAGE_SIZE"
  --validation-cases "$VALIDATION_CASES"
  --init-features "$INIT_FEATURES"
  --workers "$WORKERS"
  --vis-images "$VIS_IMAGES"
  --vis-freq "$VIS_FREQ"
  --experiments "$EXPERIMENTS"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
  CMD+=(--skip-existing)
fi

"${CMD[@]}"
