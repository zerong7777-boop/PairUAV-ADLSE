#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav"
PROJECT_ROOT="/media/jgzn/SSD_lexar/RZ/UAVM"
PYTHON="/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python"
OUT_ROOT="${PROJECT_ROOT}/experiments/phase27_a_evidence_state_manifest"

cd "${REPO_ROOT}"

echo "Phase27 A v3 calibration bounded: building full calibrated axes and manifest"
PYTHONPATH=. "${PYTHON}" scripts/phase27_a_evidence_state_v3_calibrate_manifest.py \
  --feature-manifest "${OUT_ROOT}/features/a_evidence_state_v3_feature_manifest_bounded.csv" \
  --raw-v3-manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_bounded.csv" \
  --v2-manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest_v2.csv" \
  --out-axes "${OUT_ROOT}/features/a_evidence_state_v3_calibrated_axes.csv" \
  --out-manifest "${OUT_ROOT}/manifests/a_evidence_state_manifest_v3_calibrated.csv" \
  --metrics-out "${OUT_ROOT}/metrics/a_evidence_state_feature_calibration_build_metrics.json" \
  --fit-split train

echo "Phase27 A v3 calibration bounded: checking outputs and no-train/no-submission flags"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
from pathlib import Path

out_root = Path("/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest")
paths = [
    out_root / "features/a_evidence_state_v3_calibrated_axes.csv",
    out_root / "manifests/a_evidence_state_manifest_v3_calibrated.csv",
    out_root / "metrics/a_evidence_state_feature_calibration_build_metrics.json",
]
missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"missing bounded calibration outputs: {missing}")

with (out_root / "metrics/a_evidence_state_feature_calibration_build_metrics.json").open("r", encoding="utf-8") as handle:
    metrics = json.load(handle)

row_count = int(metrics.get("row_count", 0))
if row_count < 60000:
    raise SystemExit(f"bounded calibration row_count below 60000: {row_count}")
if metrics.get("training_started") is not False:
    raise SystemExit("training_started flag is not false")
if metrics.get("submission_created") is not False:
    raise SystemExit("submission_created flag is not false")
if metrics.get("exactly_one_base_regime_passed") is not True:
    raise SystemExit("exactly_one_base_regime_passed is not true")
if metrics.get("leakage_audit", {}).get("passed") is not True:
    raise SystemExit("leakage audit did not pass")

print(f"bounded calibration row_count={row_count}")
print("NO TRAINING STARTED; NO SUBMISSION CREATED")
PY

echo "Phase27 A evidence-state v3 calibration bounded complete: ${OUT_ROOT}"
