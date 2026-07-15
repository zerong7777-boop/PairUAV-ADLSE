#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [eval_max_samples]" >&2
  exit 1
fi

RUN_DIR="$1"
EVAL_MAX_SAMPLES="${2:-2048}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$RUN_DIR/checkpoint-final.pth"
EVAL_DIR="$RUN_DIR/official_dev_eval_max${EVAL_MAX_SAMPLES}"

if [[ ! -s "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

bash "$REPO_ROOT/scripts/eval_pairuav_official_devsplit.sh" "$CKPT" "$EVAL_DIR" "$EVAL_MAX_SAMPLES"

"${PYTHON_BIN:-python3}" - "$EVAL_DIR/official_metrics.json" "$RUN_DIR/selection_summary.json" <<'PY'
import json
import sys

metrics_path, out_path = sys.argv[1], sys.argv[2]
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)
summary = {
    "checkpoint": "checkpoint-final.pth",
    "eval_metrics": metrics_path,
    "final_score": metrics.get("final_score"),
    "distance_rel_error": metrics.get("distance_rel_error"),
    "angle_rel_error": metrics.get("angle_rel_error"),
    "num_prediction_rows": metrics.get("num_prediction_rows"),
    "selection_status": "smoke_eval_only" if metrics.get("num_prediction_rows", 0) < 204120 else "full_devsplit_eval",
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
print(json.dumps(summary, indent=2))
PY
