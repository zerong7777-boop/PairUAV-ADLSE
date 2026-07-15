#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"

cd "${REPO_ROOT}"

mkdir -p "${OUT_ROOT}/schema" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_build_manifest.py \
  --project-root "${PROJECT_ROOT}" \
  --split eval \
  --out "${OUT_ROOT}/manifests/a_evidence_state_manifest.csv" \
  --schema-out "${OUT_ROOT}/schema/manifest_schema.json"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_validate.py \
  --project-root "${PROJECT_ROOT}" \
  --manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest.csv" \
  --metrics-json "${OUT_ROOT}/metrics/a_evidence_state_metrics.json" \
  --metrics-csv "${OUT_ROOT}/metrics/a_evidence_state_metrics.csv"

echo "Phase27 A evidence-state bounded validation complete: ${OUT_ROOT}"
