#!/usr/bin/env bash
set -euo pipefail

REPO=/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
PROJECT=/media/jgzn/SSD_lexar/RZ/UAVM
PY=/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python
BASE=$PROJECT/experiments/phase27_a_evidence_state_manifest

cd "$REPO"
PYTHONPATH=. "$PY" scripts/phase27_a_evidence_state_v3_build_overlap_validation.py \
  --project-root "$PROJECT" \
  --v3-calibrated-manifest "$BASE/manifests/a_evidence_state_manifest_v3_calibrated_v2.csv" \
  --v3-calibrated-axes "$BASE/features/a_evidence_state_v3_calibrated_axes_v2.csv" \
  --v2-reference-manifest "$BASE/manifests/a_evidence_state_manifest_v2.csv" \
  --v2-reference-metrics "$BASE/metrics/a_evidence_state_manifest_v2_metrics.json" \
  --v2-calibration-metrics "$BASE/metrics/a_evidence_state_feature_calibration_v2_metrics.json" \
  --identity-audit-json "$BASE/metrics/a_evidence_state_calibration_v3_identity_audit_metrics.json" \
  --matched-surface-out "$BASE/manifests/a_evidence_state_calibration_v3_matched_reference_surface.csv" \
  --control-metrics-json "$BASE/metrics/a_evidence_state_calibration_v3_control_preservation_metrics.json" \
  --full-regression-json "$BASE/metrics/a_evidence_state_calibration_v3_full_surface_regression_metrics.json" \
  --combined-metrics-json "$BASE/metrics/a_evidence_state_calibration_v3_overlap_metrics.json" \
  --report-out "$BASE/reports/a_evidence_state_calibration_v3_overlap_validation_report.md" \
  --mode bounded

test -s "$BASE/metrics/a_evidence_state_calibration_v3_identity_audit_metrics.json"
test -s "$BASE/manifests/a_evidence_state_calibration_v3_matched_reference_surface.csv"
test -s "$BASE/metrics/a_evidence_state_calibration_v3_control_preservation_metrics.json"
test -s "$BASE/metrics/a_evidence_state_calibration_v3_full_surface_regression_metrics.json"
test -s "$BASE/metrics/a_evidence_state_calibration_v3_overlap_metrics.json"
test -s "$BASE/reports/a_evidence_state_calibration_v3_overlap_validation_report.md"

echo BOUNDED_EVAL_NO_TRAINING_NO_SUBMISSION
