#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/scripts:${PYTHONPATH:-}"

"${PYTHON}" -m unittest \
  tests.test_phase27_a_taxonomy_redesign_v2_schema \
  tests.test_phase27_a_taxonomy_redesign_v2_rules \
  tests.test_phase27_a_taxonomy_redesign_v2_manifest \
  tests.test_phase27_a_taxonomy_redesign_v2_metrics

echo "phase27_a_taxonomy_redesign_v2_smoke PASS"
