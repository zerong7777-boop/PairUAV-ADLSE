#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/schema" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

PYTHONPATH=. "${PYTHON}" -m unittest tests.test_phase27_a_evidence_state_manifest_v2 -v

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_build_manifest_v2.py \
  --project-root "${PROJECT_ROOT}" \
  --split train \
  --limit 128 \
  --out "${OUT_ROOT}/manifests/a_evidence_state_manifest_v2_smoke.csv" \
  --schema-out "${OUT_ROOT}/schema/manifest_schema_v2.json" \
  --coverage-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v2_smoke_metrics.json"

PYTHONPATH=. "${PYTHON}" - <<'PY'
import csv
from pathlib import Path

path = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v2_smoke.csv")
base_cols = [
    "base_ordinary_control_anchor",
    "base_high_evidence_anchor",
    "base_hard_trainable",
    "base_low_observable",
    "base_ambiguous_unreliable",
    "base_unknown_insufficient_features",
]
risk_cols = [
    "heading_risk",
    "range_risk",
    "ambiguous_scale_tag",
    "semantic_geometry_conflict_tag",
    "target_regime_shift_tag",
    "matcher_fallback_tag",
    "weak_spatial_support_tag",
]
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "smoke manifest has no rows"
for idx, row in enumerate(rows):
    assert sum(int(row[col]) for col in base_cols) == 1, (idx, row)
    for col in risk_cols:
        assert row[col] in {"0", "1"}, (idx, col, row[col])
print(f"v2 smoke invariant check passed: {len(rows)} rows")
PY

echo "Phase27 A evidence-state v2 smoke complete: ${OUT_ROOT}"
