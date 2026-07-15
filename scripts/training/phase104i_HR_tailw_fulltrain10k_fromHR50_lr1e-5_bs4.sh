#!/usr/bin/env bash
set -euo pipefail

UAVM_ROOT="${UAVM_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
PHASE="${UAVM_ROOT}/runs/phase88_axiswise_interaction_position_v1"
WORKER="${UAVM_ROOT}/external/reloc3r_pairuav"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /media/jgzn/SSD_lexar/conda_envs/uavm/bin/python ]]; then
    PYTHON_BIN=/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

RUN_ID="${RUN_ID:-phase104i_HR_tailw_fulltrain10k_fromHR50_lr1e-5_bs4_lab0628a}"
RUN_ROOT="${PHASE}/train_runs/${RUN_ID}"
EVAL_ROOT="${PHASE}/metrics/${RUN_ID}_val811"
RECORD_ROOT="${PHASE}/records"

INIT_CKPT="${INIT_CKPT:-${PHASE}/train_runs/phase104h_HR_full_loss_all_fulltrain50k_from10klabel_lr1e-5_bs4_lab0628a/checkpoint-final.pth}"
TRAIN_JSON_ROOT="${TRAIN_JSON_ROOT:-${UAVM_ROOT}/runs/devsplit_v1/train_json}"
VAL_JSON_ROOT="${VAL_JSON_ROOT:-${UAVM_ROOT}/runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed/val_json}"
IMAGE_ROOT="${IMAGE_ROOT:-${UAVM_ROOT}/official/UAVM_2026/pairUAV/train_tour}"

OUTPUT_MODE="pairuav_phase104e_paaer_hard_heading_range"
MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='${OUTPUT_MODE}')"
CRITERION_EXPR="${CRITERION_EXPR:-__import__('phase104i_tail_loss').PairUAVTailWeightedOfficialLoss(heading_weight=1.0, range_weight=1.0, angle_floor_deg=1.0, distance_floor=1.0, absolute_heading_weight=0.05, absolute_range_weight=0.10, tail_start=80.0, tail_end=120.0, tail_max_weight=3.0)}"
TRAIN_DATASET="PairUAV(json_root='${TRAIN_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='train', resolution=(512,384), seed=42)"
VAL_DATASET="PairUAV(json_root='${VAL_JSON_ROOT}', image_root='${IMAGE_ROOT}', split='dev', resolution=(512,384), seed=42)"

BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
PRINT_FREQ="${PRINT_FREQ:-20}"
CLEAN_STEP_CKPTS="${CLEAN_STEP_CKPTS:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"

for required in \
  "${WORKER}/train.py" \
  "${WORKER}/eval_pairuav.py" \
  "${WORKER}/phase104i_tail_loss.py" \
  "${INIT_CKPT}" \
  "${TRAIN_JSON_ROOT}" \
  "${VAL_JSON_ROOT}" \
  "${IMAGE_ROOT}" \
  "${PHASE}/scripts/phase88_collect_summary.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "[phase104i-tail] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${RUN_ROOT}" || -e "${EVAL_ROOT}" ]]; then
  echo "[phase104i-tail] refusing to overwrite existing run or eval dir:" >&2
  echo "  run: ${RUN_ROOT}" >&2
  echo "  eval: ${EVAL_ROOT}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}" "${RECORD_ROOT}"

{
  echo "run_id=${RUN_ID}"
  echo "date_start=$(date -Is)"
  echo "uavm_root=${UAVM_ROOT}"
  echo "worker=${WORKER}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "output_mode=${OUTPUT_MODE}"
  echo "model=${MODEL_EXPR}"
  echo "train_json=${TRAIN_JSON_ROOT}"
  echo "val_json=${VAL_JSON_ROOT}"
  echo "criterion=${CRITERION_EXPR}"
  echo "trainable_policy=all"
  echo "batch_size=${BATCH_SIZE}"
  echo "lr=${LR}"
  echo "epochs=${EPOCHS}"
  echo "max_train_steps=${MAX_TRAIN_STEPS}"
  echo "print_freq=${PRINT_FREQ}"
  echo "clean_step_ckpts=${CLEAN_STEP_CKPTS}"
  echo "skip_eval=${SKIP_EVAL}"
  echo "interpretation=Phase104i range-tail weighted official-like continuation from HR50 final checkpoint"
} | tee "${RUN_ROOT}/phase104i_run.env"

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
  --num_workers 4 \
  --print_freq "${PRINT_FREQ}" \
  --eval_freq 0 \
  --save_freq 0 \
  --keep_freq 0 \
  --lr "${LR}" \
  --blr "${LR}" \
  --warmup_epochs 0 \
  --amp 1 \
  --trainable_policy all 2>&1 | tee "${RUN_ROOT}/train.log"

if [[ "${SKIP_EVAL}" == "1" ]]; then
  if [[ "${CLEAN_STEP_CKPTS}" == "1" ]]; then
    rm -f "${RUN_ROOT}"/checkpoint-step*.pth "${RUN_ROOT}"/checkpoint-last.pth
  fi
  echo "date_finished=$(date -Is)" | tee -a "${RUN_ROOT}/phase104i_run.env"
  exit 0
fi

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
  echo "[phase104i-tail] metrics json not found: ${METRICS_JSON}" >&2
  exit 4
fi

"${PYTHON_BIN}" "${PHASE}/scripts/phase88_collect_summary.py" \
  --phase-root "${PHASE}" \
  --append-run "${RUN_ID}" \
  --output-mode "${OUTPUT_MODE}" \
  --max-steps "${MAX_TRAIN_STEPS}" \
  --init-ckpt "${INIT_CKPT}" \
  --criterion-expr "${CRITERION_EXPR}" \
  --trainable-policy all \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --seed 42 \
  --metrics-json "${METRICS_JSON}"

if [[ "${CLEAN_STEP_CKPTS}" == "1" ]]; then
  echo "[phase104i-tail] cleaning non-final step checkpoints under ${RUN_ROOT}"
  rm -f "${RUN_ROOT}"/checkpoint-step*.pth "${RUN_ROOT}"/checkpoint-last.pth
fi

echo "date_finished=$(date -Is)" | tee -a "${RUN_ROOT}/phase104i_run.env"
