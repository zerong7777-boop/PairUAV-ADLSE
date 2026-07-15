#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1"
SURFACE_ROOT="$PHASE_ROOT/surfaces/phase54_8192_fixed_val811"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
STAMP="${PHASE56_G5_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RANK1_CKPT="$UAVM_ROOT/synced_results/reloc3r_official_full_epoch_20260507/checkpoint-final.pth"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT

cd "$REPO_ROOT"

echo "phase56 G5 rank1-safe s200 b2 started at $(date -Is)"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
echo "RANK1_CKPT=$RANK1_CKPT"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

BASELINE_EVAL_DIR="$PHASE_ROOT/eval/g5_rank1_baseline_${STAMP}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$RANK1_CKPT" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$BASELINE_EVAL_DIR"
echo "phase56 G5 rank1 baseline eval completed at $(date -Is)"
cat "$BASELINE_EVAL_DIR/val_metrics_endpoint_range_span.json"

CONTROL_RUN_NAME="g5_rank1_control_s200_lr1e7_b2_${STAMP}"
RUN_NAME="$CONTROL_RUN_NAME" bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase56_g5_rank1_control_s200_lr1e7_b2.env
CONTROL_RUN_DIR="$OUTPUT_ROOT/$CONTROL_RUN_NAME"
CONTROL_EVAL_DIR="$PHASE_ROOT/eval/${CONTROL_RUN_NAME}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$CONTROL_RUN_DIR/checkpoint-final.pth" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$CONTROL_EVAL_DIR"
echo "phase56 G5 rank1 control s200 eval completed at $(date -Is)"
cat "$CONTROL_EVAL_DIR/val_metrics_endpoint_range_span.json"

M1A_RUN_NAME="g5_rank1_m1a_s200_lr1e7_b2_${STAMP}"
RUN_NAME="$M1A_RUN_NAME" bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase56_g5_rank1_m1a_s200_lr1e7_b2.env
M1A_RUN_DIR="$OUTPUT_ROOT/$M1A_RUN_NAME"
M1A_EVAL_DIR="$PHASE_ROOT/eval/${M1A_RUN_NAME}_val811"
"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$M1A_RUN_DIR/checkpoint-final.pth" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$M1A_EVAL_DIR"
echo "phase56 G5 rank1 M1-A s200 eval completed at $(date -Is)"
cat "$M1A_EVAL_DIR/val_metrics_endpoint_range_span.json"

echo "BASELINE_EVAL_DIR=$BASELINE_EVAL_DIR"
echo "CONTROL_RUN_DIR=$CONTROL_RUN_DIR"
echo "CONTROL_EVAL_DIR=$CONTROL_EVAL_DIR"
echo "M1A_RUN_DIR=$M1A_RUN_DIR"
echo "M1A_EVAL_DIR=$M1A_EVAL_DIR"
echo "phase56 G5 rank1-safe s200 b2 finished at $(date -Is)"
