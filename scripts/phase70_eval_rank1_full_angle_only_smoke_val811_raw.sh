#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
RUN_NAME="${RUN_NAME:-g5_rank1_full_angle_only_smoke_s64_b1_20260603_082323}"
CHECKPOINT="$UAVM_ROOT/runs/phase70_rank1_full_angle_only_finetune_v1/train_runs/$RUN_NAME/checkpoint-final.pth"
VAL_JSON_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase54_8192_fixed_val811/val_json"
IMAGE_ROOT="$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour"
OUTPUT_DIR="$UAVM_ROOT/runs/phase70_rank1_full_angle_only_finetune_v1/eval/${RUN_NAME}_val811_raw"

cd "$REPO_ROOT"

"$PYTHON_BIN" eval_pairuav.py \
  --checkpoint "$CHECKPOINT" \
  --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$IMAGE_ROOT', split='dev', resolution=(512,384), seed=777)" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "$OUTPUT_DIR"

cat "$OUTPUT_DIR/val_metrics_endpoint_range_span.json"
