#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-$UAVM_ROOT/external/reloc3r_pairuav}"
PYTHON_BIN="${PYTHON_BIN:-/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python}"

PREV_RUN="${PREV_RUN:-/tmp/uavm_phase82_repro_audit/reloc3r_bounded10k_repro_v2_curve_lab_clean_bs4_full_20260608}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/uavm_phase82_repro_audit}"
RUN_NAME="${RUN_NAME:-reloc3r_10ksample_clean_resume_to5000_bs4_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
EVAL_ROOT="$RUN_DIR/eval_val811"

TRAIN_JSON_ROOT="${TRAIN_JSON_ROOT:-$UAVM_ROOT/runs/phase69_far_native_bounded10k_checkpoint_v1/surfaces/far_bounded10k_exclude_fixed_val811_seed69069/train_json}"
VAL_JSON_ROOT="${VAL_JSON_ROOT:-$UAVM_ROOT/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed/val_json}"
IMAGE_ROOT="${IMAGE_ROOT:-$UAVM_ROOT/official/UAVM_2026/pairUAV/train_tour}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-5}"
LOCAL_MILESTONE_STEPS="${LOCAL_MILESTONE_STEPS:-500,1500,2500}"
TOTAL_STEP_OFFSET="${TOTAL_STEP_OFFSET:-2500}"

MODEL_EXPR="${MODEL_EXPR:-Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')}"
CRITERION_EXPR="${CRITERION_EXPR:-PairUAVHeadingRangeLoss(heading_weight=1.0, range_weight=1.0, beta=0.1)}"

mkdir -p "$RUN_DIR" "$EVAL_ROOT"
cp "$PREV_RUN/checkpoint-last.pth" "$RUN_DIR/checkpoint-last.pth"

cd "$REPO_ROOT"

TRAIN_DATASET="PairUAV(json_root='${TRAIN_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='train', resolution=(512,384), seed=777)"
TEST_DATASET="PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=(512,384), seed=777)"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "RUN_DIR=$RUN_DIR"
  echo "PREV_RUN=$PREV_RUN"
  echo "TRAIN_JSON_ROOT=$TRAIN_JSON_ROOT"
  echo "VAL_JSON_ROOT=$VAL_JSON_ROOT"
  echo "IMAGE_ROOT=$IMAGE_ROOT"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "LR=$LR"
  echo "LOCAL_MILESTONE_STEPS=$LOCAL_MILESTONE_STEPS"
  echo "TOTAL_STEP_OFFSET=$TOTAL_STEP_OFFSET"
  echo "MODEL_EXPR=$MODEL_EXPR"
  echo "CRITERION_EXPR=$CRITERION_EXPR"
  echo "train_json_count=$(find "$TRAIN_JSON_ROOT" -type f -name '*.json' | wc -l)"
  echo "val_json_count=$(find "$VAL_JSON_ROOT" -type f -name '*.json' | wc -l)"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
} > "$RUN_DIR/run.env"

"$PYTHON_BIN" train.py \
  --train_dataset "$TRAIN_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --model "$MODEL_EXPR" \
  --train_criterion "$CRITERION_EXPR" \
  --test_criterion "$CRITERION_EXPR" \
  --epochs 2 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --warmup_epochs 0 \
  --max_train_steps 0 \
  --step_checkpoint_freq 0 \
  --milestone_steps "$LOCAL_MILESTONE_STEPS" \
  --milestone_model_only 1 \
  --eval_freq 0 \
  --save_freq 1 \
  --keep_freq 1 \
  --print_freq 20 \
  --amp 1 \
  --output_dir "$RUN_DIR"

IFS=',' read -r -a STEPS <<< "$LOCAL_MILESTONE_STEPS"
for step in "${STEPS[@]}"; do
  step="$(echo "$step" | tr -d ' ')"
  [[ -z "$step" ]] && continue
  step_padded="$(printf "%06d" "$step")"
  ckpt="$RUN_DIR/checkpoint-step${step_padded}.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing milestone checkpoint: $ckpt" >&2
    continue
  fi
  total_step=$((TOTAL_STEP_OFFSET + step))
  out_dir="$EVAL_ROOT/step$(printf "%06d" "$total_step")"
  "$PYTHON_BIN" eval_pairuav.py \
    --checkpoint "$ckpt" \
    --test_dataset "$TEST_DATASET" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --amp 1 \
    --output_dir "$out_dir"
done

if [[ -f "$RUN_DIR/checkpoint-final.pth" ]]; then
  "$PYTHON_BIN" eval_pairuav.py \
    --checkpoint "$RUN_DIR/checkpoint-final.pth" \
    --test_dataset "$TEST_DATASET" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --amp 1 \
    --output_dir "$EVAL_ROOT/final_total$(printf "%06d" $((TOTAL_STEP_OFFSET + 2500)))"
fi

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for metrics_path in sorted((run_dir / "eval_val811").glob("*/val_metrics_range_span.json")):
    label = metrics_path.parent.name
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    match = re.search(r"(\d+)$", label)
    total_steps = int(match.group(1)) if match else None
    rows.append({
        "label": label,
        "total_completed_steps": total_steps,
        "angle_mae_deg": payload.get("angle_mae_deg"),
        "distance_mae": payload.get("distance_mae"),
        "angle_rel_error": payload.get("angle_rel_error"),
        "distance_rel_error": payload.get("distance_rel_error"),
        "final_score_proxy": payload.get("final_score_proxy"),
        "samples": payload.get("samples"),
        "metrics_path": str(metrics_path),
    })

rows.sort(key=lambda row: (row["total_completed_steps"] is None, row["total_completed_steps"] or 10**12, row["label"]))
fieldnames = [
    "label", "total_completed_steps", "angle_mae_deg", "distance_mae",
    "angle_rel_error", "distance_rel_error", "final_score_proxy",
    "samples", "metrics_path",
]
csv_path = run_dir / "summary_curve_total_steps.csv"
json_path = run_dir / "summary_curve_total_steps.json"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(f"SUMMARY_CSV={csv_path}")
print(f"SUMMARY_JSON={json_path}")
PY

echo "finished_at=$(date --iso-8601=seconds)" >> "$RUN_DIR/run.env"
echo "RUN_DIR=$RUN_DIR"
