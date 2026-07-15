#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$REPO_ROOT/configs/pairuav_official_metric_longer_v1.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$UAVM_ROOT/runs/reloc3r_official_pairuav}"
RUN_NAME="${RUN_NAME:-${RUN_FAMILY}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$ENV_FILE" "$RUN_DIR/run.env"

echo "RUN_DIR=$RUN_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "UAVM_ROOT=$UAVM_ROOT"
echo "PRETRAINED=${PRETRAINED:-}"

PYTHON_BIN="$PYTHON_BIN" \
UAVM_ROOT="$UAVM_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
RUN_NAME="$RUN_NAME" \
RESOLUTION="$RESOLUTION" \
MODEL_EXPR="$MODEL_EXPR" \
CRITERION_EXPR="$CRITERION_EXPR" \
SEED="$SEED" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
AMP="$AMP" \
PRETRAINED="${PRETRAINED:-}" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
STEP_CHECKPOINT_FREQ="$STEP_CHECKPOINT_FREQ" \
EVAL_FREQ="$EVAL_FREQ" \
TRAIN_MATCHER_FEATURES="${TRAIN_MATCHER_FEATURES:-}" \
VAL_MATCHER_FEATURES="${VAL_MATCHER_FEATURES:-}" \
TRAIN_BSCR_FEATURES="${TRAIN_BSCR_FEATURES:-}" \
VAL_BSCR_FEATURES="${VAL_BSCR_FEATURES:-}" \
FREEZE_EXCEPT_ANGLE_SPECIALIST="${FREEZE_EXCEPT_ANGLE_SPECIALIST:-0}" \
TRAINABLE_POLICY="${TRAINABLE_POLICY:-}" \
bash "$REPO_ROOT/scripts/train_pairuav_full_devsplit.sh" 2>&1 | tee "$RUN_DIR/train.log"
