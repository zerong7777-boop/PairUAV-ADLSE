#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2a_canonical_pair_identity_contract_join_audit
CAND=$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/contract" "$OUT/metrics" "$OUT/tables" "$OUT/reports"

python3 - <<PY
from scripts.phase27_a_v3_2a_identity_common import write_json
from pathlib import Path
root = Path("$ROOT")
stress = sorted(str(p) for p in (root / "experiments/phase27_a_validation_spine/baseline_surfaces").glob("phase21_stress*_joinable_baseline_surface.csv"))
write_json("$OUT/input_manifest.json", {
  "candidate": "$CAND",
  "full_dev": "$FULL",
  "stress": stress,
  "scope": "validation-only identity contract and join audit; no training; no fuzzy overlap; no silent dedup"
})
PY

ART=(--artifact candidate="$CAND" --artifact full_dev="$FULL")
for f in "$ROOT"/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_stress*_joinable_baseline_surface.csv; do
  name=$(basename "$f" .csv)
  name=${name#phase21_}
  name=${name%_joinable_baseline_surface}
  ART+=(--artifact "$name=$f")
done

python3 -m scripts.phase27_a_v3_2a_identity_contract --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_artifact_profile "${ART[@]}" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_pairwise_join_matrix "${ART[@]}" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_duplicate_missing_exports "${ART[@]}" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_repair_candidates --profile "$OUT/tables/artifact_identity_profile.csv" --pairwise "$OUT/tables/pairwise_join_matrix.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_report --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_write_record --output-dir "$OUT" --input-manifest "$OUT/input_manifest.json"

for f in \
  input_manifest.json \
  contract/canonical_pair_identity_contract.md \
  metrics/identity_profile_metrics.json \
  metrics/pairwise_join_matrix_metrics.json \
  metrics/identity_repair_candidate_metrics.json \
  tables/artifact_identity_profile.csv \
  tables/pairwise_join_matrix.csv \
  tables/duplicate_blocked_pairs.csv \
  tables/missing_key_rows.csv \
  tables/disjoint_universe_summary.csv \
  tables/identity_repair_candidates.csv \
  reports/identity_join_audit_report.md \
  reports/duplicate_resolution_report.md \
  reports/identity_repair_candidate_report.md \
  reports/a_v3_2a_route_verdict.md \
  phase27_a_v3_2a_identity_join_audit_experiment_record.md; do
  test -f "$OUT/$f"
done

echo A_V3_2A_IDENTITY_JOIN_BOUNDED_COMPLETE
