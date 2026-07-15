#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
SRC=$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv
BASE=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
STRESS_DIR=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces
OUT=$ROOT/experiments/phase27_a_v3_1_shared_outcome_surface_acquisition_bias_audit

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/metrics" "$OUT/tables" "$OUT/reports"

python3 -m scripts.phase27_a_v3_1_artifact_discovery --root "$ROOT" --output-dir "$OUT"

python3 - <<PY
import csv
from pathlib import Path
src = Path("$SRC")
out = Path("$OUT")
candidate = out / "manifests" / "candidate_rows.csv"
with src.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
fields = rows[0].keys() if rows else []
candidate.parent.mkdir(parents=True, exist_ok=True)
with candidate.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("candidate", len(rows))
PY

STRESS_ARGS=()
for f in "$STRESS_DIR"/phase21_stress*_joinable_baseline_surface.csv; do
  name=$(basename "$f" .csv)
  name=${name#phase21_}
  name=${name%_joinable_baseline_surface}
  name=${name//[^A-Za-z0-9_]/_}
  STRESS_ARGS+=(--stress "$name=$f")
done

python3 -m scripts.phase27_a_v3_1_shared_surface_builder \
  --candidate "$OUT/manifests/candidate_rows.csv" \
  --baseline "$BASE" \
  "${STRESS_ARGS[@]}" \
  --output-dir "$OUT"

SURF=$OUT/manifests/a_v3_1_shared_outcome_surface_manifest.csv
python3 -m scripts.phase27_a_v3_1_coverage_bias_audit --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_outcome_consistency --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_candidate_predictability --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_stable_control_audit --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_hard_ambiguity_shared_decomposition --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_report --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"
python3 -m scripts.phase27_a_v3_1_write_record --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"

test -f "$OUT/input_manifest.json"
test -f "$OUT/manifests/a_v3_1_shared_outcome_surface_manifest.csv"
test -f "$OUT/metrics/a_v3_1_shared_surface_coverage_metrics.json"
test -f "$OUT/metrics/a_v3_1_shared_outcome_consistency_metrics.json"
test -f "$OUT/metrics/a_v3_1_shared_candidate_predictability_metrics.json"
test -f "$OUT/metrics/a_v3_1_shared_stable_control_metrics.json"
test -f "$OUT/tables/a_v3_1_join_bias_by_target_group.csv"
test -f "$OUT/tables/a_v3_1_candidate_predictability_by_target_group.csv"
test -f "$OUT/tables/a_v3_1_hard_ambiguity_shared_decomposition.csv"
test -f "$OUT/reports/a_v3_1_shared_surface_coverage_report.md"
test -f "$OUT/reports/a_v3_1_shared_outcome_consistency_report.md"
test -f "$OUT/reports/a_v3_1_shared_candidate_predictability_report.md"
test -f "$OUT/reports/a_v3_1_shared_stable_control_report.md"
test -f "$OUT/reports/a_v3_1_route_verdict.md"
test -f "$OUT/phase27_a_v3_1_shared_outcome_surface_experiment_record.md"

echo A_V3_1_SHARED_SURFACE_BOUNDED_COMPLETE
