#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${UAVM_ROOT:-}" ]]; then
  if [[ -d /root/autodl-tmp/uavm_2026 ]]; then
    UAVM_ROOT=/root/autodl-tmp/uavm_2026
  elif [[ -d /media/jgzn/SSD_lexar/RZ/UAVM ]]; then
    UAVM_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
  else
    echo "UAVM_ROOT is not set and no default root was detected." >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_JSON_ROOT="${TRAIN_JSON_ROOT:-$UAVM_ROOT/runs/devsplit_v1/train_json}"
VAL_JSON_ROOT="${VAL_JSON_ROOT:-$UAVM_ROOT/runs/devsplit_v1/val_json}"
IMAGE_ROOT="${IMAGE_ROOT:-$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$UAVM_ROOT/runs/reloc3r_official_pairuav}"
RUN_NAME="${RUN_NAME:-pairuav_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"

RESOLUTION="${RESOLUTION:-(512,384)}"
MODEL_EXPR="${MODEL_EXPR:-Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')}"
CRITERION_EXPR="${CRITERION_EXPR:-PairUAVHeadingRangeLoss(heading_weight=1.0, range_weight=1.0, beta=0.1)}"
SEED="${SEED:-777}"
TRAIN_SIZE="${TRAIN_SIZE:-1024}"
VAL_SIZE="${VAL_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
AMP="${AMP:-1}"
PRETRAINED="${PRETRAINED:-}"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

TRAIN_DATASET="${TRAIN_SIZE} @ PairUAV(json_root='${TRAIN_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='train', resolution=${RESOLUTION}, seed=${SEED})"
TEST_DATASET="${VAL_SIZE} @ PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=${RESOLUTION}, seed=${SEED})"

CMD=(
  "$PYTHON_BIN" train.py
  --train_dataset "$TRAIN_DATASET"
  --test_dataset "$TEST_DATASET"
  --model "$MODEL_EXPR"
  --train_criterion "$CRITERION_EXPR"
  --test_criterion "$CRITERION_EXPR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --lr "$LR"
  --warmup_epochs "$WARMUP_EPOCHS"
  --eval_freq 1
  --save_freq 1
  --keep_freq 1
  --print_freq 20
  --amp "$AMP"
  --output_dir "$RUN_DIR"
)

if [[ -n "$PRETRAINED" ]]; then
  CMD+=(--pretrained "$PRETRAINED")
fi

printf 'RUN_DIR=%s\n' "$RUN_DIR"
printf 'TRAIN_DATASET=%s\n' "$TRAIN_DATASET"
printf 'TEST_DATASET=%s\n' "$TEST_DATASET"

exec "${CMD[@]}"
