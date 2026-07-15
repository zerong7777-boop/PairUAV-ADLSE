#!/usr/bin/env bash
set -euo pipefail
cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
python3 -m unittest \
  tests.test_phase27_a_taxonomy_redesign_v3_schema \
  tests.test_phase27_a_taxonomy_redesign_v3_candidates \
  tests.test_phase27_a_taxonomy_redesign_v3_outcomes \
  tests.test_phase27_a_taxonomy_redesign_v3_readiness \
  tests.test_phase27_a_taxonomy_redesign_v3_metrics -v
python3 - <<'PY'
from pathlib import Path
from scripts.phase27_a_taxonomy_redesign_v3_candidates import build_candidate_manifest
from scripts.phase27_a_taxonomy_redesign_v3_outcomes import build_outcome_manifest
from scripts.phase27_a_taxonomy_redesign_v3_readiness import build_readiness_manifest
from scripts.phase27_a_taxonomy_redesign_v3_metrics import write_metrics_bundle
from scripts.phase27_a_taxonomy_redesign_v3_io import write_csv_dicts

out = Path('/tmp/phase27_a_taxonomy_redesign_v3_smoke')
(out / 'manifests').mkdir(parents=True, exist_ok=True)
rows = [{
    'canonical_pair_id': 'a::b',
    'evidence_sufficiency_score': '0.8',
    'heading_observability_score': '0.8',
    'range_observability_score': '0.3',
    'semantic_geometric_conflict_score': '0.8',
    'match_sufficiency_score': '0.8',
    'ambiguity_tail_risk_score': '0.8',
    'control_stability_score': '0.2',
    'layout_scale_risk_score': '0.1',
}]
full = [{'canonical_pair_id': 'a::b', 'baseline_angle_rel_error': '0.9', 'baseline_distance_rel_error': '0.2'}]
stress = [{'canonical_pair_id': 'a::b', 'baseline_angle_rel_error': '1.8', 'baseline_distance_rel_error': '0.2'}]
candidate = build_candidate_manifest(rows)
outcome = build_outcome_manifest(candidate, full, stress)
readiness = build_readiness_manifest(candidate, outcome)
write_csv_dicts(out / 'manifests' / 'training_readiness_verdict_manifest.csv', readiness, list(readiness[0].keys()))
write_metrics_bundle(readiness, out)
PY
echo SMOKE_OK
