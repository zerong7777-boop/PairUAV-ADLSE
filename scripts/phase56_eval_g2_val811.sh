#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1"
VAL_JSON_ROOT="$PHASE_ROOT/surfaces/phase48_4089_fixed/val_json"
IMAGE_ROOT="$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

cd "$REPO_ROOT"

run_eval() {
  local checkpoint="$1"
  local output_dir="$2"
  "$PYTHON_BIN" eval_pairuav.py \
    --checkpoint "$checkpoint" \
    --test_dataset "PairUAV(json_root='$VAL_JSON_ROOT', image_root='$IMAGE_ROOT', split='dev', resolution=(512,384), seed=777)" \
    --batch_size 4 \
    --num_workers 4 \
    --amp 1 \
    --output_dir "$output_dir"
}

run_eval \
  "$PHASE_ROOT/train_runs/g2_control_bounded4096_lab_b2_20260528_070106/checkpoint-final.pth" \
  "$PHASE_ROOT/eval/g2_control_bounded4096_lab_b2_val811"

run_eval \
  "$PHASE_ROOT/train_runs/g2_m1a_bounded4096_lab_b2_20260528_071546/checkpoint-final.pth" \
  "$PHASE_ROOT/eval/g2_m1a_bounded4096_lab_b2_val811"
