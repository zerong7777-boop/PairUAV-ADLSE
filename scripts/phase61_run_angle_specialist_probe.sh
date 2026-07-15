#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase61_rank1_compatible_angle_specialist_v1"
SURFACE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase54_8192_fixed_val811"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
STAMP="${PHASE61_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RANK1_CKPT="$UAVM_ROOT/synced_results/reloc3r_official_full_epoch_20260507/checkpoint-final.pth"
PHASE61_ENV_FILES="${PHASE61_ENV_FILES:-configs/pairuav_phase61_angle_specialist_smoke.env}"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT

mkdir -p "$PHASE_ROOT/eval" "$PHASE_ROOT/reports" "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "phase61 angle specialist probe started at $(date -Is)"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
echo "RANK1_CKPT=$RANK1_CKPT"
echo "PHASE61_ENV_FILES=$PHASE61_ENV_FILES"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

BASELINE_EVAL_DIR="$PHASE_ROOT/eval/rank1_baseline_${STAMP}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$RANK1_CKPT" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$BASELINE_EVAL_DIR"
cat "$BASELINE_EVAL_DIR/val_metrics_endpoint_range_span.json"

SUMMARY_MD="$PHASE_ROOT/REPORT.md"
cat > "$SUMMARY_MD" <<EOF
# Phase61 Rank1-Compatible Angle Specialist Report

Updated: $(date -Is)

## Scope

This phase probes a frozen-base internal angle specialist loaded from the rank1 official Reloc3r checkpoint.

Baseline:

- checkpoint: \`$RANK1_CKPT\`
- eval_dir: \`$BASELINE_EVAL_DIR\`

## Runs

| run | eval metrics | run dir |
|---|---|---|
EOF

for ENV_FILE in $PHASE61_ENV_FILES; do
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing env file: $ENV_FILE" >&2
    exit 2
  fi
  RUN_FAMILY="$(grep '^RUN_FAMILY=' "$ENV_FILE" | head -1 | cut -d= -f2)"
  if [[ -z "$RUN_FAMILY" ]]; then
    echo "RUN_FAMILY missing in $ENV_FILE" >&2
    exit 2
  fi
  RUN_NAME="${RUN_FAMILY}_${STAMP}"
  echo "=== Phase61 training $RUN_NAME from $ENV_FILE ==="
  RUN_NAME="$RUN_NAME" bash scripts/train_pairuav_official_metric_longer_5090.sh "$ENV_FILE"

  RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
  EVAL_DIR="$PHASE_ROOT/eval/${RUN_NAME}_val811"
  "$PYTHON_BIN" eval_pairuav.py \
    --model "Reloc3rRelpose(img_size=512, output_mode='pairuav_angle_specialist_heading_range', angle_specialist_hidden_dim=256, angle_specialist_dropout=0.05, angle_specialist_max_residual=0.10, angle_specialist_init_scale=0.0)" \
    --checkpoint "$RUN_DIR/checkpoint-final.pth" \
    --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
    --batch_size 4 \
    --num_workers 4 \
    --amp 1 \
    --output_dir "$EVAL_DIR"
  cat "$EVAL_DIR/val_metrics_endpoint_range_span.json"
  echo "| \`$RUN_NAME\` | \`$EVAL_DIR/val_metrics_endpoint_range_span.json\` | \`$RUN_DIR\` |" >> "$SUMMARY_MD"
done

cat >> "$SUMMARY_MD" <<EOF

## Gate

Promotion thresholds:

- weak continuation: angle MAE <= 0.1297
- serious promotion: angle MAE <= 0.125347
- distance protection: distance MAE <= 0.043750

Do not package hidden-test inference unless a bounded run clears the gate.
EOF

echo "phase61 angle specialist probe finished at $(date -Is)"
