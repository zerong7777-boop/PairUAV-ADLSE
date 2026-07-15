#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/jgzn/SSD_lexar/RZ/UAVM}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/reloc3r_pairuav}"
PYTHON="${PYTHON:-/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest}"
FEATURE_MANIFEST="${FEATURE_MANIFEST:-${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_bounded.csv}"
SMOKE_AXES="${SMOKE_AXES:-${OUT_ROOT}/features/a_evidence_state_v3_calibrated_axes_smoke.csv}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_calibrated_smoke.csv}"
SMOKE_METRICS="${SMOKE_METRICS:-${OUT_ROOT}/metrics/a_evidence_state_feature_calibration_smoke_metrics.json}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/features" "${OUT_ROOT}/manifests" "${OUT_ROOT}/metrics"

echo "Phase27 A v3 calibration smoke: running unit tests"
PYTHONPATH=. "${PYTHON}" -m unittest tests.test_phase27_a_evidence_state_v3_calibration -v

echo "Phase27 A v3 calibration smoke: building 512-row calibrated axes and manifest"
PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_calibrate_manifest.py \
  --feature-manifest "${FEATURE_MANIFEST}" \
  --out-axes "${SMOKE_AXES}" \
  --out-manifest "${SMOKE_MANIFEST}" \
  --metrics-out "${SMOKE_METRICS}" \
  --limit 512 \
  --fit-split train

echo "Phase27 A v3 calibration smoke: checking leakage audit, outputs, and base-regime invariant"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import csv
import json
from pathlib import Path

axes_path = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/features/a_evidence_state_v3_calibrated_axes_smoke.csv")
manifest_path = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_calibrated_smoke.csv")
metrics_path = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/metrics/a_evidence_state_feature_calibration_smoke_metrics.json")
for path in (axes_path, manifest_path, metrics_path):
    assert path.exists(), f"missing smoke output: {path}"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
assert metrics["row_count"] == 512, metrics["row_count"]
assert metrics["leakage_audit"]["passed"] is True
assert metrics["leakage_audit_passed"] is True
assert metrics["exactly_one_base_regime_passed"] is True
assert metrics["training_started"] is False
assert metrics["submission_created"] is False

base_cols = [
    "base_ordinary_control_anchor",
    "base_high_evidence_anchor",
    "base_hard_trainable",
    "base_low_observable",
    "base_ambiguous_unreliable",
    "base_unknown_insufficient_features",
]
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 512, len(rows)
for index, row in enumerate(rows):
    assert sum(int(row[col]) for col in base_cols) == 1, (index, row.get("pair_id"))

with axes_path.open("r", encoding="utf-8", newline="") as handle:
    axes_rows = list(csv.DictReader(handle))
assert len(axes_rows) == 512, len(axes_rows)
print(f"calibration leakage audit passed; exactly-one-base-regime passed: {len(rows)} rows")
PY

echo "NO TRAINING STARTED; NO SUBMISSION CREATED"
echo "Phase27 A evidence-state v3 calibration smoke complete: ${OUT_ROOT}"
