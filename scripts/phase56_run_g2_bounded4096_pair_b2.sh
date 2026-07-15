#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1"
SURFACE_ROOT="$PHASE_ROOT/surfaces/phase48_4089_fixed"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT

cd "$REPO_ROOT"

echo "phase56 G2 bounded4096 pair b2 started at $(date -Is)"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

RUN_NAME="g2_control_bounded4096_lab_b2_$(date +%Y%m%d_%H%M%S)" \
bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase56_control_bounded4096_lab_b2.env

echo "phase56 G2 control b2 completed at $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

RUN_NAME="g2_m1a_bounded4096_lab_b2_$(date +%Y%m%d_%H%M%S)" \
bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase56_m1a_bounded4096_lab_b2.env

echo "phase56 G2 m1a b2 completed at $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "phase56 G2 bounded4096 pair b2 finished at $(date -Is)"
