#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase82_official_style_adaptation_v1"
RUN_ROOT="$PHASE_ROOT/w2_reloc3r_trunk_reuse_512/train_runs"
EVAL_ROOT="$PHASE_ROOT/w2_reloc3r_trunk_reuse_512/eval_val811"
VAL_JSON_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed/val_json"
IMAGE_ROOT="$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

cd "$REPO_ROOT"
mkdir -p "$EVAL_ROOT"

TEST_DATASET="PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=(512,384), seed=777)"

echo "phase82 W2 val811 eval started at $(date -Is)"
echo "RUN_ROOT=$RUN_ROOT"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "TEST_DATASET=$TEST_DATASET"

for ckpt in "$RUN_ROOT"/pairuav_phase82_r*/checkpoint-final.pth
do
  run_dir="$(dirname "$ckpt")"
  run_name="$(basename "$run_dir")"
  out_dir="$EVAL_ROOT/${run_name}_val811"
  echo "evaluating $run_name"
  "$PYTHON_BIN" eval_pairuav.py \
    --model "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')" \
    --test_dataset "$TEST_DATASET" \
    --checkpoint "$ckpt" \
    --batch_size 16 \
    --num_workers 4 \
    --amp 1 \
    --output_dir "$out_dir"
  echo "completed eval $run_name at $(date -Is)"
done

echo "phase82 W2 val811 eval finished at $(date -Is)"

