#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

python3 -m unittest \
  tests.test_phase27_a_v3_2b_common \
  tests.test_phase27_a_v3_2b_fixed_manifest \
  tests.test_phase27_a_v3_2b_runner_capability \
  tests.test_phase27_a_v3_2b_outcome_reexport \
  tests.test_phase27_a_v3_2b_shared_wide_surface \
  tests.test_phase27_a_v3_2b_identity_coverage_bias_audit \
  tests.test_phase27_a_v3_2b_repair_feasibility -v

OUT=/tmp/phase27_a_v3_2b_smoke
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/candidate.csv" <<'CSV'
canonical_pair_id,source_image_key,target_image_key,pair_key,target_key,group_id,scene_key,split_key,evidence_sufficient_candidate,READY_CONTROL_PRESERVATION
a,S.JPG,T.JPG,s|t,g,g,g,dev,True,False
b,S2.JPG,T2.JPG,s2|t2,g,g,g,dev,False,True
CSV
cat > "$OUT/full.csv" <<'CSV'
canonical_pair_id,source_image_a,source_image_b,baseline_angle_abs_error,baseline_distance_abs_error,baseline_surface_source
a,S.JPG,T.JPG,1,2,smoke
b,S2.JPG,T2.JPG,1,2,smoke
CSV
cat > "$OUT/stress.csv" <<'CSV'
canonical_pair_id,source_row_index,source_image_a,source_image_b,baseline_angle_abs_error,baseline_distance_abs_error,baseline_surface_source
a,0,S.JPG,T.JPG,2,3,smoke
b,1,S2.JPG,T2.JPG,2,3,smoke
CSV
cat > "$OUT/pairwise.csv" <<'CSV'
left_artifact,right_artifact,key_strategy,key_role,intersection_count,promotion_eligible
candidate,full_dev,canonical_pair_id,promotion_key,2,true
CSV

python3 -m scripts.phase27_a_v3_2b_fixed_manifest --candidate "$OUT/candidate.csv" --full-dev "$OUT/full.csv" --pairwise "$OUT/pairwise.csv" --mode bounded_clean_candidate_full --limit 2 --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_runner_capability --repo-root /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_outcome_reexport --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --baseline-surface "$OUT/full.csv" --stress-surface stress_smoke="$OUT/stress.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_shared_wide_surface --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --baseline "$OUT/tables/baseline_on_fixed_manifest_bounded.csv" --stress-long "$OUT/tables/stress_on_fixed_manifest_bounded_long.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_identity_coverage_bias_audit --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --shared-wide "$OUT/tables/baseline_stress_shared_wide_surface_bounded.csv" --outcome-metrics "$OUT/metrics/outcome_reexport_metrics.json" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_repair_feasibility --fixed-manifest "$OUT/manifests/fixed_shared_pair_manifest_bounded.csv" --stress-duplicates "$OUT/tables/stress_duplicate_groups.csv" --identity-metrics "$OUT/metrics/shared_surface_identity_metrics.json" --coverage-metrics "$OUT/metrics/shared_surface_coverage_bias_metrics.json" --runner-capability "$OUT/metrics/runner_capability.json" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2b_write_record --output-dir "$OUT"

test -f "$OUT/reports/a_v3_2b_route_verdict.md"
test -f "$OUT/phase27_a_v3_2b_fixed_manifest_shared_surface_experiment_record.md"
echo A_V3_2B_SMOKE_OK

