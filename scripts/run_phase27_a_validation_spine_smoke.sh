#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav"
PYTHON="/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python"
OUT_DIR="/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_validation_spine_smoke"

cd "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON}" -m unittest \
  tests.test_phase27_a_validation_spine_keys \
  tests.test_phase27_a_validation_spine_registry \
  tests.test_phase27_a_validation_spine_manifest_slices \
  tests.test_phase27_a_validation_spine_suites \
  -v

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_validation_spine_run.py \
  --mode smoke \
  --project-root /media/jgzn/SSD_lexar/RZ/UAVM \
  --out-dir "${OUT_DIR}"
