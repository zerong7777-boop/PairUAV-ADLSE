#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

python3 -m unittest \
  tests.test_phase27_a_v3_1_shared_surface_common \
  tests.test_phase27_a_v3_1_artifact_discovery \
  tests.test_phase27_a_v3_1_shared_surface_builder \
  tests.test_phase27_a_v3_1_coverage_bias_audit \
  tests.test_phase27_a_v3_1_shared_outcome_consistency \
  tests.test_phase27_a_v3_1_shared_candidate_predictability \
  tests.test_phase27_a_v3_1_shared_stable_control_audit \
  tests.test_phase27_a_v3_1_hard_ambiguity_shared_decomposition -v

OUT=/tmp/phase27_a_v3_1_shared_surface_smoke
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/candidate.csv" <<'CSV'
canonical_pair_id,target_key,group_id,ambiguity_candidate,control_candidate,multi_modal_ambiguous,semantic_geometric_conflict,stress_sensitive_ambiguous,tail_error_unreliable
p1,t1,g1,true,false,true,false,false,false
p2,t1,g1,false,false,false,false,false,false
p3,t2,g2,false,false,false,false,false,false
p4,t3,g3,false,true,false,false,false,false
dup,t4,g4,false,false,false,false,false,false
CSV
cat > "$OUT/baseline.csv" <<'CSV'
canonical_pair_id,full_dev_join_status,baseline_angle_abs_error,baseline_distance_rel_error,baseline_joint_error_score,baseline_heading_hard,baseline_range_hard,baseline_joint_hard
p1,joined,1,2,3,true,false,true
p2,joined,1,2,3,false,false,false
dup,joined,1,2,3,false,false,false
dup,joined,1,2,3,false,false,false
CSV
cat > "$OUT/stress.csv" <<'CSV'
canonical_pair_id,stress_join_status,stress_baseline_angle_abs_error,stress_baseline_distance_rel_error,stress_baseline_final_score,stress_heading_sensitive,stress_range_sensitive,stress_joint_sensitive
p1,joined,2,4,6,true,false,true
p3,joined,2,4,6,false,true,true
CSV

python3 -m scripts.phase27_a_v3_1_shared_surface_builder --candidate "$OUT/candidate.csv" --baseline "$OUT/baseline.csv" --stress main="$OUT/stress.csv" --output-dir "$OUT"
SURF="$OUT/manifests/a_v3_1_shared_outcome_surface_manifest.csv"
python3 -m scripts.phase27_a_v3_1_coverage_bias_audit --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_outcome_consistency --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_candidate_predictability --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_shared_stable_control_audit --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_hard_ambiguity_shared_decomposition --input "$SURF" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_report --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_1_write_record --output-dir "$OUT"

test -f "$OUT/manifests/a_v3_1_shared_outcome_surface_manifest.csv"
test -f "$OUT/reports/a_v3_1_route_verdict.md"
echo A_V3_1_SHARED_SURFACE_SMOKE_OK
