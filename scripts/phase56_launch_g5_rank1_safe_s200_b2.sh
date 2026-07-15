#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1"
STAMP="${PHASE56_G5_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="$PHASE_ROOT/g5_rank1_safe_s200_b2_${STAMP}.log"

cd "$REPO_ROOT"
mkdir -p "$PHASE_ROOT"

PHASE56_G5_STAMP="$STAMP" nohup bash scripts/phase56_run_g5_rank1_safe_s200_b2.sh > "$LOG" 2>&1 &
PID=$!

echo "PID=$PID"
echo "STAMP=$STAMP"
echo "LOG=$LOG"
