#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="$UAVM_ROOT/external/reloc3r_pairuav"
PHASE_ROOT="$UAVM_ROOT/runs/phase70_rank1_full_angle_only_finetune_v1"
LOG="$PHASE_ROOT/g5_rank1_full_angle_only_smoke_s64_b1.log"
PID_FILE="$PHASE_ROOT/g5_rank1_full_angle_only_smoke_s64_b1.pid"

mkdir -p "$PHASE_ROOT"
cd "$REPO_ROOT"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "already_running pid=$OLD_PID"
    exit 0
  fi
fi

nohup bash scripts/phase70_run_rank1_full_angle_only_smoke_s64_b1.sh > "$LOG" 2>&1 &
PID="$!"
echo "$PID" > "$PID_FILE"
echo "started pid=$PID log=$LOG"
