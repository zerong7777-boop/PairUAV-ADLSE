#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/features" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

PYTHONPATH=. "${PYTHON}" -m unittest tests.test_phase27_a_evidence_state_v3_features -v

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_build_features.py \
  --project-root "${PROJECT_ROOT}" \
  --split train \
  --limit 128 \
  --stride 1 \
  --enable-cheap-image-features true \
  --cached-matcher-jsonl "${PROJECT_ROOT}/experiments/phase26_b_selective_correspondence_reasoning/features/train_bscr_features.jsonl" \
  --out "${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_smoke.csv" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v3_feature_smoke_metrics.json"

PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_build_manifest.py \
  --feature-manifest "${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_smoke.csv" \
  --out "${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_smoke.csv" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_manifest_v3_smoke_metrics.json"

PYTHONPATH=. "${PYTHON}" - <<'PY'
import csv
from pathlib import Path

path = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_smoke.csv")
base_cols = [
    "base_ordinary_control_anchor",
    "base_high_evidence_anchor",
    "base_hard_trainable",
    "base_low_observable",
    "base_ambiguous_unreliable",
    "base_unknown_insufficient_features",
]
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "empty v3 smoke manifest"
for i, row in enumerate(rows):
    assert sum(int(row[col]) for col in base_cols) == 1, (i, row.get("pair_id"))
print(f"v3 smoke base-regime invariant passed: {len(rows)} rows")
PY

echo "Phase27 A evidence-state v3 smoke complete: ${OUT_ROOT}"
