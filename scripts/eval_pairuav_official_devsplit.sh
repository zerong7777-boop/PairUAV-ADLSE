#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint> <output_dir> [max_samples]" >&2
  exit 1
fi

CHECKPOINT="$1"
OUTPUT_DIR="$2"
MAX_SAMPLES="${3:-0}"

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
JSON_ROOT="${VAL_JSON_ROOT:-$UAVM_ROOT/runs/devsplit_v1/val_json}"
IMAGE_ROOT="${IMAGE_ROOT:-$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour}"
PHASE20_ROOT="${PHASE20_ROOT:-$UAVM_ROOT/experiments/paper_pillars/20_official_metric_aware_distance_calibration}"
MANIFEST="${MANIFEST:-$PHASE20_ROOT/manifests/devsplit_v1_official_metric_manifest.csv}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" infer_pairuav_with_progress.py \
  --json-root "$JSON_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model "${MODEL_EXPR:-Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')}" \
  --output-dir "$OUTPUT_DIR" \
  --split dev \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --amp "${AMP:-1}" \
  --max-samples "$MAX_SAMPLES" \
  --log-every "${LOG_EVERY:-200}" \
  --matcher-feature-manifest "${MATCHER_FEATURE_MANIFEST:-}" \
  --bscr-feature-manifest "${BSCR_FEATURE_MANIFEST:-}" \
  --diagnostics-output "${DIAGNOSTICS_OUTPUT:-$OUTPUT_DIR/diagnostics.csv}"

PRED="$OUTPUT_DIR/result.txt"
if [[ "$MAX_SAMPLES" != "0" ]]; then
  awk 'NR==1 || NR<='"$((MAX_SAMPLES + 1))" "$MANIFEST" > "$OUTPUT_DIR/manifest_limited.csv"
  MANIFEST_TO_USE="$OUTPUT_DIR/manifest_limited.csv"
else
  MANIFEST_TO_USE="$MANIFEST"
fi

"$PYTHON_BIN" "$PHASE20_ROOT/scripts/evaluate_official_metrics.py" \
  --manifest-csv "$MANIFEST_TO_USE" \
  --prediction "$PRED" \
  --output-json "$OUTPUT_DIR/official_metrics.json" \
  --per-sample-csv "$OUTPUT_DIR/official_per_sample.csv" \
  --bucket-csv "$OUTPUT_DIR/official_buckets.csv" \
  --id-column sample_id \
  --gt-angle-column gt_angle \
  --gt-distance-column gt_distance \
  --prediction-format txt

echo "OFFICIAL_METRICS=$OUTPUT_DIR/official_metrics.json"
