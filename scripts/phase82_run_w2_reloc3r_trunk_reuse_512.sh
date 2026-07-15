#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase82_official_style_adaptation_v1"
SURFACE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed"
OUTPUT_ROOT="$PHASE_ROOT/w2_reloc3r_trunk_reuse_512/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "phase82 W2 Reloc3r trunk-reuse 512 matrix started at $(date -Is)"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

for cfg in \
  pairuav_phase82_r0_pose_trunk_full_512.env \
  pairuav_phase82_r1_no_pose_trunk_full_512.env \
  pairuav_phase82_r2_pose_trunk_headonly_512.env \
  pairuav_phase82_r3_no_pose_trunk_headonly_512.env
do
  run_base="${cfg%.env}"
  run_name="${run_base}_$(date +%Y%m%d_%H%M%S)"
  echo "running $cfg as $run_name"
  RUN_NAME="$run_name" bash scripts/train_pairuav_official_metric_longer_5090.sh "configs/$cfg"
  echo "completed $run_name at $(date -Is)"
done

echo "phase82 W2 Reloc3r trunk-reuse 512 matrix finished at $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

