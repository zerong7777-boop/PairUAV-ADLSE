#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav"
PROJECT_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM"
PYTHON="/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python"
OUT_DIR="${PROJECT_ROOT}/experiments/phase27_a_validation_spine"
EVIDENCE="${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_calibrated_v2.csv"
REFERENCE="${PROJECT_ROOT}/experiments/phase26_c1_bounded_matched_eval/subsets/phase13_overlap_v1/eval_official_manifest.csv"

cd "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON}" scripts/phase27_a_validation_spine_run.py \
  --mode bounded \
  --project-root "${PROJECT_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --evidence-manifest "${EVIDENCE}" \
  --reference-surface "${REFERENCE}"
