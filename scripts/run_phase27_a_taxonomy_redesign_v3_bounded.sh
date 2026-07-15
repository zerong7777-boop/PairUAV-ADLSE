#!/usr/bin/env bash
set -euo pipefail
cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
python3 - <<'PY'
from pathlib import Path
from scripts.phase27_a_taxonomy_redesign_v3_io import find_input_paths, read_csv_dicts, write_csv_dicts, write_input_manifest
from scripts.phase27_a_taxonomy_redesign_v3_candidates import build_candidate_manifest
from scripts.phase27_a_taxonomy_redesign_v3_outcomes import build_outcome_manifest
from scripts.phase27_a_taxonomy_redesign_v3_readiness import build_readiness_manifest
from scripts.phase27_a_taxonomy_redesign_v3_metrics import write_metrics_bundle
from scripts.phase27_a_taxonomy_redesign_v3_report import write_report
from scripts.phase27_a_taxonomy_redesign_v3_write_record import write_record

root = Path('/media/jgzn/SSD_lexar/RZ/UAVM')
out = root / 'experiments' / 'phase27_a_taxonomy_redesign_v3'
(out / 'manifests').mkdir(parents=True, exist_ok=True)
(out / 'metrics').mkdir(parents=True, exist_ok=True)
(out / 'reports').mkdir(parents=True, exist_ok=True)
paths = write_input_manifest(root, out / 'input_manifest.json')
v2_manifest = Path(paths['taxonomy_v2_dir']) / 'phase27_a_taxonomy_redesign_v2_manifest.csv'
source_rows = read_csv_dicts(str(v2_manifest)) if v2_manifest.exists() else read_csv_dicts(paths['candidate_manifest'])
candidate = build_candidate_manifest(source_rows)
candidate_fields = list(candidate[0].keys()) if candidate else []
write_csv_dicts(out / 'manifests' / 'non_leaking_candidate_manifest.csv', candidate, candidate_fields)
full_rows = read_csv_dicts(paths['full_dev_surface']) if paths.get('full_dev_surface') else []
stress_rows = []
for stress_path in paths.get('stress_surfaces', [])[:1]:
    stress_rows.extend(read_csv_dicts(stress_path))
validation_fields = [
    'baseline_error_score', 'heading_error_score', 'range_error_score',
    'stress_sensitivity_score', 'checkpoint_disagreement_score', 'tail_outlier_flag',
    'full_dev_joined', 'stress_joined',
]
candidate_for_outcome = []
for idx, row in enumerate(candidate):
    merged = dict(row)
    if idx < len(source_rows):
        for field in validation_fields:
            if field in source_rows[idx]:
                merged[field] = source_rows[idx][field]
    candidate_for_outcome.append(merged)
outcome = build_outcome_manifest(candidate_for_outcome, full_rows, stress_rows)
outcome_fields = list(outcome[0].keys()) if outcome else []
write_csv_dicts(out / 'manifests' / 'validation_only_outcome_attribution_manifest.csv', outcome, outcome_fields)
readiness = build_readiness_manifest(candidate, outcome)
readiness_fields = list(readiness[0].keys()) if readiness else []
write_csv_dicts(out / 'manifests' / 'training_readiness_verdict_manifest.csv', readiness, readiness_fields)
write_metrics_bundle(readiness, out)
write_report(out)
write_record(out, 'bash scripts/run_phase27_a_taxonomy_redesign_v3_bounded.sh')
PY
echo BOUNDED_V3_COMPLETE
