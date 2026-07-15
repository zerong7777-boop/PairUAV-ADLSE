#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"
TRAIN_LIMIT="${TRAIN_LIMIT:-50000}"
DEV_LIMIT="${DEV_LIMIT:-10000}"
ENABLE_CHEAP="${ENABLE_CHEAP:-true}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/features" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

TRAIN_FEATURES="${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_train_bounded.csv"
DEV_FEATURES="${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_dev_bounded.csv"
BOUNDED_FEATURES="${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_bounded.csv"
BOUNDED_MANIFEST="${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_bounded.csv"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_build_features.py \
  --project-root "${PROJECT_ROOT}" \
  --split train \
  --limit "${TRAIN_LIMIT}" \
  --stride 1 \
  --enable-cheap-image-features "${ENABLE_CHEAP}" \
  --cached-matcher-jsonl "${PROJECT_ROOT}/experiments/phase26_b_selective_correspondence_reasoning/features/train_bscr_features.jsonl" \
  --out "${TRAIN_FEATURES}" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v3_feature_train_bounded_metrics.json"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_build_features.py \
  --project-root "${PROJECT_ROOT}" \
  --split dev \
  --limit "${DEV_LIMIT}" \
  --stride 1 \
  --enable-cheap-image-features "${ENABLE_CHEAP}" \
  --cached-matcher-jsonl "${PROJECT_ROOT}/experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl" \
  --out "${DEV_FEATURES}" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v3_feature_dev_bounded_metrics.json"

head -n 1 "${TRAIN_FEATURES}" > "${BOUNDED_FEATURES}"
tail -n +2 "${TRAIN_FEATURES}" >> "${BOUNDED_FEATURES}"
tail -n +2 "${DEV_FEATURES}" >> "${BOUNDED_FEATURES}"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_build_manifest.py \
  --feature-manifest "${BOUNDED_FEATURES}" \
  --out "${BOUNDED_MANIFEST}" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v3_bounded_build_metrics.json"

echo "Phase27 A evidence-state v3 bounded build complete: ${OUT_ROOT}"
