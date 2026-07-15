#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase59_rank1_compatible_angle_mechanism_v1"
SURFACE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase54_8192_fixed_val811"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
STAMP="${PHASE59_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RANK1_CKPT="$UAVM_ROOT/synced_results/reloc3r_official_full_epoch_20260507/checkpoint-final.pth"
CONTROL_ENV_FILE="${CONTROL_ENV_FILE:-configs/pairuav_phase59_rank1_control_s64_lr3e8_b1.env}"
TRUE_SWAP_ENV_FILE="${TRUE_SWAP_ENV_FILE:-configs/pairuav_phase59_rank1_true_swap_s64_lr3e8_b1.env}"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT

mkdir -p "$PHASE_ROOT/eval" "$PHASE_ROOT/reports" "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "phase59 rank1 true-swap s64 b1 started at $(date -Is)"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
echo "RANK1_CKPT=$RANK1_CKPT"
echo "CONTROL_ENV_FILE=$CONTROL_ENV_FILE"
echo "TRUE_SWAP_ENV_FILE=$TRUE_SWAP_ENV_FILE"
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

CONTROL_RUN_NAME="phase59_rank1_control_s64_lr3e8_b1_${STAMP}"
RUN_NAME="$CONTROL_RUN_NAME" bash scripts/train_pairuav_official_metric_longer_5090.sh "$CONTROL_ENV_FILE"
CONTROL_RUN_DIR="$OUTPUT_ROOT/$CONTROL_RUN_NAME"
CONTROL_EVAL_DIR="$PHASE_ROOT/eval/${CONTROL_RUN_NAME}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$CONTROL_RUN_DIR/checkpoint-final.pth" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$CONTROL_EVAL_DIR"
cat "$CONTROL_EVAL_DIR/val_metrics_endpoint_range_span.json"
"$PYTHON_BIN" scripts/phase59_gate_report.py \
  --metrics-json "$CONTROL_EVAL_DIR/val_metrics_endpoint_range_span.json" \
  --run-name "$CONTROL_RUN_NAME" \
  --base-checkpoint "$RANK1_CKPT" \
  --run-dir "$CONTROL_RUN_DIR" \
  --eval-dir "$CONTROL_EVAL_DIR" \
  --out-dir "$PHASE_ROOT/reports/$CONTROL_RUN_NAME"

TRUE_SWAP_RUN_NAME="phase59_rank1_true_swap_s64_lr3e8_b1_${STAMP}"
RUN_NAME="$TRUE_SWAP_RUN_NAME" bash scripts/train_pairuav_official_metric_longer_5090.sh "$TRUE_SWAP_ENV_FILE"
TRUE_SWAP_RUN_DIR="$OUTPUT_ROOT/$TRUE_SWAP_RUN_NAME"
TRUE_SWAP_EVAL_DIR="$PHASE_ROOT/eval/${TRUE_SWAP_RUN_NAME}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$TRUE_SWAP_RUN_DIR/checkpoint-final.pth" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$TRUE_SWAP_EVAL_DIR"
cat "$TRUE_SWAP_EVAL_DIR/val_metrics_endpoint_range_span.json"
"$PYTHON_BIN" scripts/phase59_gate_report.py \
  --metrics-json "$TRUE_SWAP_EVAL_DIR/val_metrics_endpoint_range_span.json" \
  --run-name "$TRUE_SWAP_RUN_NAME" \
  --base-checkpoint "$RANK1_CKPT" \
  --run-dir "$TRUE_SWAP_RUN_DIR" \
  --eval-dir "$TRUE_SWAP_EVAL_DIR" \
  --out-dir "$PHASE_ROOT/reports/$TRUE_SWAP_RUN_NAME"

cat > "$PHASE_ROOT/REPORT.md" <<EOF
# Phase59 Rank1-Compatible Angle Mechanism Report

Updated: $(date -Is)

## Runs

- baseline_eval_dir: \`$BASELINE_EVAL_DIR\`
- control_run_dir: \`$CONTROL_RUN_DIR\`
- control_eval_dir: \`$CONTROL_EVAL_DIR\`
- control_report: \`$PHASE_ROOT/reports/$CONTROL_RUN_NAME\`
- true_swap_run_dir: \`$TRUE_SWAP_RUN_DIR\`
- true_swap_eval_dir: \`$TRUE_SWAP_EVAL_DIR\`
- true_swap_report: \`$PHASE_ROOT/reports/$TRUE_SWAP_RUN_NAME\`

## Decision Rule

Promote only if true-swap passes G1 and G3, and if its angle delta is better than both rank1 and same-step control.
EOF

echo "phase59 rank1 true-swap s64 b1 finished at $(date -Is)"
