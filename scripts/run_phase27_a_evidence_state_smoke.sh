#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"

cd "${REPO_ROOT}"

mkdir -p "${OUT_ROOT}/schema" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_inspect_inputs.py \
  --project-root "${PROJECT_ROOT}" \
  --out "${OUT_ROOT}/schema/input_inventory.json"

PYTHONPATH=. "${PYTHON}" -m unittest tests.test_phase27_a_evidence_state_manifest -v

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_build_manifest.py \
  --project-root "${PROJECT_ROOT}" \
  --split train \
  --limit 128 \
  --out "${OUT_ROOT}/manifests/a_evidence_state_manifest_smoke.csv" \
  --schema-out "${OUT_ROOT}/schema/manifest_schema.json" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_metrics_smoke.json"

echo "Phase27 A evidence-state smoke complete: ${OUT_ROOT}"
