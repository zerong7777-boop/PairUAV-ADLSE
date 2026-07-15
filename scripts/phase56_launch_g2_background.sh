#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1"
LOG_PATH="$PHASE_ROOT/g2_bounded4096_pair.log"
PID_PATH="$PHASE_ROOT/g2_bounded4096_pair.pid"

mkdir -p "$PHASE_ROOT"
cd "$REPO_ROOT"

nohup bash scripts/phase56_run_g2_bounded4096_pair.sh > "$LOG_PATH" 2>&1 &
pid="$!"
echo "$pid" > "$PID_PATH"
echo "$pid"
