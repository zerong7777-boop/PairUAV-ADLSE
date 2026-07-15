#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase70_rank1_full_angle_only_finetune_v1"
SURFACE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase54_8192_fixed_val811"
OUTPUT_ROOT="$PHASE_ROOT/train_runs"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"
RUN_NAME="${RUN_NAME:-g5_rank1_full_angle_only_smoke_s64_b1_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"

export UAVM_ROOT
export PYTHON_BIN
export TRAIN_JSON_ROOT="$SURFACE_ROOT/train_json"
export VAL_JSON_ROOT="$SURFACE_ROOT/val_json"
export OUTPUT_ROOT
export RUN_NAME

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "phase70 rank1 full angle-only smoke started at $(date -Is)"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

bash scripts/train_pairuav_official_metric_longer_5090.sh configs/pairuav_phase70_rank1_full_angle_only_smoke_s64_b1.env

echo "phase70 rank1 full angle-only smoke completed at $(date -Is)"
echo "RUN_DIR=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
