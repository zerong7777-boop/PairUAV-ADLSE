#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav"
PROJECT_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM"
PYTHON="/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python"
OUT_ROOT="${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest"

cd "${REPO_ROOT}"

echo "Phase27 A v3 calibration-v2 smoke: running unit tests"
PYTHONPATH=. "${PYTHON}" -m unittest tests.test_phase27_a_evidence_state_v3_calibration_v2 -v

echo "Phase27 A v3 calibration-v2 smoke: building 512-row artifacts"
PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_calibrate_manifest_v2.py \
  --feature-manifest "${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_bounded.csv" \
  --out-axes "${OUT_ROOT}/features/a_evidence_state_v3_calibrated_axes_v2_smoke.csv" \
  --out-manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_calibrated_v2_smoke.csv" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_feature_calibration_v2_smoke_metrics.json" \
  --limit 512 \
  --fit-split train

echo "Phase27 A v3 calibration-v2 smoke: checking invariants"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
from pathlib import Path

out_root = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest")
paths = [
    out_root / "features/a_evidence_state_v3_calibrated_axes_v2_smoke.csv",
    out_root / "manifests/a_evidence_state_manifest_v3_calibrated_v2_smoke.csv",
    out_root / "metrics/a_evidence_state_feature_calibration_v2_smoke_metrics.json",
]
missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"missing smoke outputs: {missing}")
with (out_root / "metrics/a_evidence_state_feature_calibration_v2_smoke_metrics.json").open("r", encoding="utf-8") as handle:
    metrics = json.load(handle)
if metrics.get("row_count") != 512:
    raise SystemExit(f"unexpected smoke row_count: {metrics.get('row_count')}")
if metrics.get("leakage_audit", {}).get("passed") is not True:
    raise SystemExit("leakage audit failed")
if metrics.get("exactly_one_base_regime_passed") is not True:
    raise SystemExit("exactly-one-base-regime failed")
if metrics.get("canonical_pair_id_nonempty_fraction", 0.0) <= 0.0:
    raise SystemExit("canonical pair-id non-empty check failed")
if metrics.get("training_started") is not False:
    raise SystemExit("training_started flag is not false")
if metrics.get("submission_created") is not False:
    raise SystemExit("submission_created flag is not false")
print("calibration-v2 smoke metrics ok:", {
    "row_count": metrics.get("row_count"),
    "ordinary_control_anchor_fraction": metrics.get("ordinary_control_anchor_fraction"),
    "canonical_pair_id_nonempty_fraction": metrics.get("canonical_pair_id_nonempty_fraction"),
})
print("NO TRAINING STARTED; NO SUBMISSION CREATED")
PY

echo "Phase27 A evidence-state v3 calibration-v2 smoke complete: ${OUT_ROOT}"
