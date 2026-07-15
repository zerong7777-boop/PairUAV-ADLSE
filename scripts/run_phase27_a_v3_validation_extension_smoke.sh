#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

python3 -m unittest \
  tests.test_phase27_a_v3_validation_extension_common \
  tests.test_phase27_a_v3_outcome_consistency_audit \
  tests.test_phase27_a_v3_hard_ambiguity_decomposition \
  tests.test_phase27_a_v3_candidate_outcome_predictability \
  tests.test_phase27_a_v3_stable_control_join_bias_audit \
  tests.test_phase27_a_v3_b_diagnostic_slices -v

OUT=/tmp/phase27_a_v3_validation_extension_smoke
rm -rf "$OUT"
mkdir -p "$OUT"
CSV="$OUT/smoke.csv"
cat > "$CSV" <<'CSV'
canonical_pair_id,evidence_sufficient_candidate,heading_observable_candidate,range_observable_candidate,semantic_geometric_conflict_candidate,local_alignment_needed_candidate,ambiguity_candidate,ordinary_candidate,control_candidate,baseline_error_score,heading_error_score,range_error_score,stress_sensitivity_score,full_dev_join_status,stress_join_status,baseline_heading_hard,baseline_range_hard,baseline_joint_hard,stress_heading_sensitive,stress_range_sensitive,stress_joint_sensitive,tail_error_high,READY_CONTROL_PRESERVATION,evidence_sufficient_heading_hard,evidence_sufficient_range_hard,evidence_sufficient_joint_hard,multi_modal_ambiguous,semantic_geometric_conflict,stress_sensitive_ambiguous,tail_error_unreliable,target_key,group_id
p1,true,true,false,true,true,true,false,false,0.8,0.8,0.2,0.1,joined,joined,true,false,true,false,false,false,false,false,true,false,true,true,true,false,false,t1,g1
p2,false,false,true,false,false,true,false,false,0.2,0.1,0.7,0.9,joined,joined,false,true,false,true,true,true,true,false,false,true,false,false,false,true,true,t2,g1
p3,false,false,false,false,false,false,true,true,0.1,0.1,0.1,0.1,missing,missing,false,false,false,false,false,false,false,true,false,false,false,false,false,false,false,t3,g2
CSV

python3 -m scripts.phase27_a_v3_outcome_consistency_audit --input "$CSV" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_hard_ambiguity_decomposition --input "$CSV" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_candidate_outcome_predictability --input "$CSV" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_stable_control_join_bias_audit --input "$CSV" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_b_diagnostic_slices --input "$CSV" --output-dir "$OUT"
python3 - <<'PY'
from scripts.phase27_a_v3_b_diagnostic_slices import write_training_policy_readiness_verdict
write_training_policy_readiness_verdict('/tmp/phase27_a_v3_validation_extension_smoke', {
    'outcome': {'verdict': 'unresolved-blocker'},
    'join_bias': {'verdict': 'join-bias-acceptable-for-analysis'},
    'predictability': {'useful_pair_count': 1},
    'stable_control': {'verdict': 'control-preservation-safe'},
})
PY

echo A_V3_VALIDATION_EXTENSION_SMOKE_OK
