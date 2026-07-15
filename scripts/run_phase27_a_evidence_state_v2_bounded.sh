#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/schema" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_build_manifest_v2.py \
  --project-root "${PROJECT_ROOT}" \
  --split eval \
  --out "${OUT_ROOT}/manifests/a_evidence_state_manifest_v2.csv" \
  --schema-out "${OUT_ROOT}/schema/manifest_schema_v2.json" \
  --coverage-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v2_build_metrics.json"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_validate_v2.py \
  --project-root "${PROJECT_ROOT}" \
  --manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest_v2.csv" \
  --metrics-json "${OUT_ROOT}/metrics/a_evidence_state_manifest_v2_metrics.json" \
  --metrics-csv "${OUT_ROOT}/metrics/a_evidence_state_manifest_v2_metrics.csv"

echo "Phase27 A evidence-state v2 bounded validation complete: ${OUT_ROOT}"
