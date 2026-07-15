#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition
CAND=$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
PAIRWISE=$ROOT/experiments/phase27_a_v3_2a_canonical_pair_identity_contract_join_audit/tables/pairwise_join_matrix.csv

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports"

python3 - <<PY
from scripts.phase27_a_v3_2b_common import write_json
from pathlib import Path
root = Path("$ROOT")
stress = sorted(str(p) for p in (root / "experiments/phase27_a_validation_spine/baseline_surfaces").glob("phase21_stress*_joinable_baseline_surface.csv"))
write_json("$OUT/input_manifest.json", {
  "candidate": "$CAND",
  "full_dev": "$FULL",
  "pairwise": "$PAIRWISE",
  "stress": stress,
  "scope": "validation-only fixed-manifest shared outcome surface; no training; no silent dedup"
})
PY

python3 -m scripts.phase27_a_v3_2b_fixed_manifest --candidate "$CAND" --full-dev "$FULL" --pairwise "$PAIRWISE" --mode bounded_clean_candidate_full --limit 10000 --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_runner_capability --repo-root "$REPO" --output-dir "$OUT"

ART=(--fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --baseline-surface "$FULL")
for f in "$ROOT"/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_stress*_joinable_baseline_surface.csv; do
  name=$(basename "$f" .csv)
  name=${name#phase21_}
  name=${name%_joinable_baseline_surface}
  ART+=(--stress-surface "$name=$f")
done

python3 -m scripts.phase27_a_v3_2b_outcome_reexport "${ART[@]}" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_shared_wide_surface --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --baseline "$OUT/tables/baseline_on_fixed_manifest_bounded.csv" --stress-long "$OUT/tables/stress_on_fixed_manifest_bounded_long.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_identity_coverage_bias_audit --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --shared-wide "$OUT/tables/baseline_stress_shared_wide_surface_bounded.csv" --outcome-metrics "$OUT/metrics/outcome_reexport_metrics.json" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_repair_feasibility --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --stress-duplicates "$OUT/tables/stress_duplicate_groups.csv" --identity-metrics "$OUT/metrics/shared_surface_identity_metrics.json" --coverage-metrics "$OUT/metrics/shared_surface_coverage_bias_metrics.json" --runner-capability "$OUT/metrics/runner_capability.json" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_write_record --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"

for f in \
  input_manifest.json \
  manifests/fixed_shared_pair_manifest_bounded.csv \
  manifests/fixed_shared_pair_manifest_bounded.metrics.json \
  tables/baseline_on_fixed_manifest_bounded.csv \
  tables/stress_on_fixed_manifest_bounded_long.csv \
  tables/baseline_stress_shared_wide_surface_bounded.csv \
  tables/stress_duplicate_groups.csv \
  tables/stress_repair_candidate_table.csv \
  tables/joined_unjoined_bias_table.csv \
  metrics/runner_capability.json \
  metrics/shared_surface_identity_metrics.json \
  metrics/shared_surface_coverage_bias_metrics.json \
  metrics/repair_feasibility_metrics.json \
  metrics/a_v3_2b_route_verdict.json \
  reports/runner_capability_report.md \
  reports/shared_surface_identity_audit.md \
  reports/shared_surface_coverage_bias_report.md \
  reports/stress_identity_failure_report.md \
  reports/stress_duplicate_group_summary.md \
  reports/stress_missing_composite_identity_report.md \
  reports/repair_feasibility_verdict.md \
  reports/a_v3_2b_route_verdict.md \
  phase27_a_v3_2b_fixed_manifest_shared_surface_experiment_record.md; do
  test -f "$OUT/$f"
done

echo A_V3_2B_BOUNDED_COMPLETE

