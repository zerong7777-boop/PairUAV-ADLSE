#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint> [output_dir]" >&2
  exit 1
fi

CHECKPOINT="$1"
shift || true

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
VAL_JSON_ROOT="${VAL_JSON_ROOT:-$UAVM_ROOT/runs/devsplit_v1/val_json}"
IMAGE_ROOT="${IMAGE_ROOT:-$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$UAVM_ROOT/runs/reloc3r_official_pairuav}"
RUN_NAME="${RUN_NAME:-eval_pairuav_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${1:-$OUTPUT_ROOT/$RUN_NAME}"

RESOLUTION="${RESOLUTION:-(512,384)}"
MODEL_EXPR="${MODEL_EXPR:-Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')}"
SEED="${SEED:-777}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-1}"
RANGE_MIN="${RANGE_MIN:--132.0}"
RANGE_MAX="${RANGE_MAX:-132.0}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

TEST_DATASET="PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=${RESOLUTION}, seed=${SEED})"

CMD=(
  "$PYTHON_BIN" eval_pairuav.py
  --model "$MODEL_EXPR"
  --test_dataset "$TEST_DATASET"
  --checkpoint "$CHECKPOINT"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --amp "$AMP"
  --output_dir "$OUTPUT_DIR"
  --range_min "$RANGE_MIN"
  --range_max "$RANGE_MAX"
)

printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
printf 'TEST_DATASET=%s\n' "$TEST_DATASET"

exec "${CMD[@]}"
