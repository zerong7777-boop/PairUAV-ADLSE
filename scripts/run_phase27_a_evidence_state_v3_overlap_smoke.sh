#!/usr/bin/env bash
set -euo pipefail

REPO=/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
PY=/media/jgzn/SSD_lexar/RZ/danzi/ESS/.venv-fcclip/bin/python
WORK=/tmp/phase27_overlap_smoke
OUT=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/metrics

rm -rf "$WORK"
mkdir -p "$WORK" "$OUT" /tmp/phase27_overlap_smoke_reports

cat > "$WORK/v2_reference.csv" <<'CSV'
pair_id,query_name,reference_name,group_id,base_regime,base_ordinary_control_anchor,risk_tags
0881/33_02,image-33,image-02,0881,ordinary_control_anchor,1,
0881/11_12,image-11,image-12,0881,ambiguous_unreliable,0,conflict
CSV

cat > "$WORK/v3_manifest.csv" <<'CSV'
source_split,json_path,group_id,pair_id,pair_key,canonical_pair_id,image_a,image_b,image_a_name,image_b_name,feature_complete,observable_adequate,image_quality_adequate,pair_identity_valid,base_regime,base_ordinary_control_anchor,risk_tags,control_centrality_score,scale_risk_axis,layout_risk_axis,conflict_risk_axis
train,/tmp/0881/33_02.json,0881,0881/33_02,image-33.jpeg|image-02.jpeg,0881/33_02,0881/image-33.jpeg,0881/image-02.jpeg,image-33.jpeg,image-02.jpeg,1,1,1,1,ordinary_control_anchor,1,,0.7,0.1,0.1,0.1
train,/tmp/0881/12_11.json,0881,0881/12_11,image-12.jpeg|image-11.jpeg,0881/12_11,0881/image-12.jpeg,0881/image-11.jpeg,image-12.jpeg,image-11.jpeg,1,1,1,1,ambiguous_unreliable,0,conflict,0.2,0.8,0.8,0.8
CSV

cp "$WORK/v3_manifest.csv" "$WORK/v3_axes.csv"
cat > "$WORK/v2_reference_metrics.json" <<'JSON'
{"row_count": 2}
JSON
cat > "$WORK/v2_calibration_metrics.json" <<'JSON'
{
  "row_count": 60000,
  "leakage_audit": {"passed": true},
  "exactly_one_base_regime_passed": true,
  "ordinary_control_anchor_fraction": 0.3204,
  "high_evidence_anchor_fraction": 0.123,
  "max_base_regime_fraction": 0.5553,
  "quota_only_suspected": false,
  "base_regime_counts": {"ordinary_control_anchor": 19224},
  "base_regime_fractions": {"ordinary_control_anchor": 0.3204}
}
JSON

cd "$REPO"
PYTHONPATH=. "$PY" -m unittest tests.test_phase27_a_evidence_state_v3_overlap_identity -v
PYTHONPATH=. "$PY" scripts/phase27_a_evidence_state_v3_build_overlap_validation.py \
  --project-root /media/jgzn/SSD_lexar/RZ/UAVM \
  --v3-calibrated-manifest "$WORK/v3_manifest.csv" \
  --v3-calibrated-axes "$WORK/v3_axes.csv" \
  --v2-reference-manifest "$WORK/v2_reference.csv" \
  --v2-reference-metrics "$WORK/v2_reference_metrics.json" \
  --v2-calibration-metrics "$WORK/v2_calibration_metrics.json" \
  --identity-audit-json "$OUT/a_evidence_state_calibration_v3_overlap_smoke_metrics.json" \
  --matched-surface-out "$WORK/matched_surface.csv" \
  --control-metrics-json "$WORK/control_metrics.json" \
  --full-regression-json "$WORK/full_regression.json" \
  --combined-metrics-json "$WORK/combined_metrics.json" \
  --report-out /tmp/phase27_overlap_smoke_reports/report.md \
  --mode smoke

"$PY" - "$WORK/control_metrics.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1], encoding="utf-8"))
assert metrics["matched_row_count"] > 0, metrics
print("synthetic overlap nonzero:", metrics["matched_row_count"])
PY

echo SMOKE_ONLY_NO_TRAINING_NO_SUBMISSION
