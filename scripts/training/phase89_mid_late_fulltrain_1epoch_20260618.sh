#!/usr/bin/env bash
set -euo pipefail

ROOT="${UAVM_ROOT:-/root/autodl-tmp/uavm_2026}"
PHASE="${ROOT}/runs/phase88_axiswise_interaction_position_v1"
WORKER="${ROOT}/external/reloc3r_pairuav"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

RUN_ID="phase89_Wstrip_T2_H8_mid_late_from_phase88_10k_fulltrain_1epoch_lr1e-5_bs4_pf20_20260618"
RUN_ROOT="${PHASE}/train_runs/${RUN_ID}"
EVAL_ROOT="${PHASE}/metrics/${RUN_ID}_val811"
RECORD_ROOT="${PHASE}/records"

INIT_CKPT="${PHASE}/train_runs/phase88_B2_H8_mid_late_10000_w0t2_lr1e-5_bs4/checkpoint-final.pth"
TRAIN_JSON_ROOT="${ROOT}/runs/devsplit_v1/train_json"
VAL_JSON_ROOT="${ROOT}/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed/val_json"
IMAGE_ROOT="${ROOT}/official/UAVM_2026/pairUAV/train_tour"

OUTPUT_MODE="pairuav_range_h0_heading_mid_late_heading_range"
MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='${OUTPUT_MODE}')"
CRITERION_EXPR="PairUAVOfficialMetricAwareLoss(heading_weight=1.0, range_weight=1.0, angle_floor_deg=1.0, distance_floor=1.0, absolute_heading_weight=0.05, absolute_range_weight=0.05)"
TRAIN_DATASET="PairUAV(json_root='${TRAIN_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='train', resolution=(512,384), seed=42)"
VAL_DATASET="PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=(512,384), seed=42)"

BATCH_SIZE=4
LR="1e-5"
EPOCHS=1
MAX_TRAIN_STEPS=0
MILESTONE_STEPS="50000,100000,150000,200000,250000,300000,350000,400000,450000"
STEP_CHECKPOINT_FREQ=50000

for required in "${WORKER}/train.py" "${WORKER}/eval_pairuav.py" "${INIT_CKPT}" "${TRAIN_JSON_ROOT}" "${VAL_JSON_ROOT}" "${IMAGE_ROOT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[phase89-fulltrain-1epoch] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${RUN_ROOT}" || -e "${EVAL_ROOT}" ]]; then
  echo "[phase89-fulltrain-1epoch] refusing to overwrite existing run or eval dir:" >&2
  echo "  run: ${RUN_ROOT}" >&2
  echo "  eval: ${EVAL_ROOT}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}" "${RECORD_ROOT}"

{
  echo "run_id=${RUN_ID}"
  echo "date_start=$(date -Is)"
  echo "root=${ROOT}"
  echo "worker=${WORKER}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "output_mode=${OUTPUT_MODE}"
  echo "model=${MODEL_EXPR}"
  echo "train_json=${TRAIN_JSON_ROOT}"
  echo "val_json=${VAL_JSON_ROOT}"
  echo "criterion=${CRITERION_EXPR}"
  echo "batch_size=${BATCH_SIZE}"
  echo "lr=${LR}"
  echo "epochs=${EPOCHS}"
  echo "max_train_steps=${MAX_TRAIN_STEPS}"
  echo "milestone_steps=${MILESTONE_STEPS}"
  echo "step_checkpoint_freq=${STEP_CHECKPOINT_FREQ}"
  echo "milestone_model_only=0"
  echo "step_checkpoint_model_only=0"
  echo "interpretation=full devsplit train_json, exactly 1 epoch from Phase88 H8-mid-late 10k checkpoint"
} | tee "${RUN_ROOT}/phase89_run.env"

cd "${WORKER}"

"${PYTHON_BIN}" train.py \
  --model "${MODEL_EXPR}" \
  --pretrained "${INIT_CKPT}" \
  --train_dataset "${TRAIN_DATASET}" \
  --test_dataset "${VAL_DATASET}" \
  --train_criterion "${CRITERION_EXPR}" \
  --test_criterion "${CRITERION_EXPR}" \
  --output_dir "${RUN_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --step_checkpoint_freq "${STEP_CHECKPOINT_FREQ}" \
  --step_checkpoint_keep_named 1 \
  --step_checkpoint_model_only 0 \
  --milestone_steps "${MILESTONE_STEPS}" \
  --milestone_model_only 0 \
  --num_workers 4 \
  --print_freq 20 \
  --eval_freq 0 \
  --save_freq 0 \
  --keep_freq 0 \
  --lr "${LR}" \
  --blr "${LR}" \
  --warmup_epochs 0 \
  --amp 1 2>&1 | tee "${RUN_ROOT}/train.log"

"${PYTHON_BIN}" eval_pairuav.py \
  --model "${MODEL_EXPR}" \
  --checkpoint "${RUN_ROOT}/checkpoint-final.pth" \
  --test_dataset "${VAL_DATASET}" \
  --output_dir "${EVAL_ROOT}" \
  --batch_size 4 \
  --num_workers 4 \
  --amp 1 2>&1 | tee "${EVAL_ROOT}/eval.log"

METRICS_JSON="${EVAL_ROOT}/val_metrics_range_span.json"
if [[ ! -f "${METRICS_JSON}" ]]; then
  echo "[phase89-fulltrain-1epoch] metrics json not found: ${METRICS_JSON}" >&2
  exit 4
fi

"${PYTHON_BIN}" "${PHASE}/scripts/phase88_collect_summary.py" \
  --phase-root "${PHASE}" \
  --append-run "${RUN_ID}" \
  --output-mode "${OUTPUT_MODE}" \
  --max-steps 459999 \
  --init-ckpt "${INIT_CKPT}" \
  --criterion-expr "${CRITERION_EXPR}" \
  --trainable-policy all \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --seed 42 \
  --metrics-json "${METRICS_JSON}"

echo "date_finished=$(date -Is)" | tee -a "${RUN_ROOT}/phase89_run.env"
