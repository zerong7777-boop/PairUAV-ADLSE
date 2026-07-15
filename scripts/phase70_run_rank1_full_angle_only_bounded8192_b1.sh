#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase70_rank1_full_angle_only_finetune_v1"
SURFACE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase54_8192_fixed_val811"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
RUN_NAME="${RUN_NAME:-g6_rank1_full_angle_only_bounded8192_b1_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
EVAL_DIR="$PHASE_ROOT/eval/${RUN_NAME}_val811_raw"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT
export RUN_NAME

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "phase70 rank1 full angle-only bounded8192 started at $(date -Is)"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase70_rank1_full_angle_only_bounded8192_b1.env

echo "phase70 rank1 full angle-only bounded8192 train completed at $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$RUN_DIR/checkpoint-final.pth" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$EVAL_DIR"

echo "phase70 rank1 full angle-only bounded8192 eval completed at $(date -Is)"
echo "RUN_DIR=$RUN_DIR"
echo "EVAL_DIR=$EVAL_DIR"
cat "$EVAL_DIR/val_metrics_endpoint_range_span.json"
