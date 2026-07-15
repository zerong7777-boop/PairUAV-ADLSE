#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
IN=$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv
OUT=$ROOT/experiments/phase27_a_v3_validation_extension_outcome_consistency_audit

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/metrics" "$OUT/tables" "$OUT/reports"

python3 - <<PY
from scripts.phase27_a_v3_validation_extension_common import write_json
write_json('$OUT/input_manifest.json', {
  'main_manifest': '$IN',
  'non_leaking_candidate_manifest': '$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/non_leaking_candidate_manifest.csv',
  'validation_only_outcome_attribution_manifest': '$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/validation_only_outcome_attribution_manifest.csv',
  'a_v3_metrics_dir': '$ROOT/experiments/phase27_a_taxonomy_redesign_v3/metrics',
  'validation_spine_baseline_surfaces': '$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces',
  'scope': 'validation-only; no training/finetune/sampler/gate labels'
})
PY

python3 -m scripts.phase27_a_v3_outcome_consistency_audit --input "$IN" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_hard_ambiguity_decomposition --input "$IN" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_candidate_outcome_predictability --input "$IN" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_stable_control_join_bias_audit --input "$IN" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_b_diagnostic_slices --input "$IN" --output-dir "$OUT"
python3 - <<PY
import json
from pathlib import Path
from scripts.phase27_a_v3_b_diagnostic_slices import write_training_policy_readiness_verdict
out = Path('$OUT')
def read(name):
    with (out / 'metrics' / name).open(encoding='utf-8') as f:
        return json.load(f)
write_training_policy_readiness_verdict(out, {
    'outcome': read('a_v3_outcome_surface_consistency_audit.json'),
    'join_bias': read('a_v3_join_bias_extension_metrics.json'),
    'predictability': read('a_v3_candidate_to_outcome_predictability_metrics.json'),
    'stable_control': read('a_v3_stable_control_stress_audit.json'),
})
PY
python3 -m scripts.phase27_a_v3_validation_extension_report --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"
python3 -m scripts.phase27_a_v3_validation_extension_write_record --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"

test -f "$OUT/reports/phase27_a_v3_validation_extension_summary.md"
test -f "$OUT/phase27_a_v3_validation_extension_experiment_record.md"
echo A_V3_VALIDATION_EXTENSION_BOUNDED_COMPLETE
