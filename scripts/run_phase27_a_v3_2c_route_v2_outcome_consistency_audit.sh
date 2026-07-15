#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
SOURCE=$ROOT/experiments/phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat
OUT=$ROOT/experiments/phase27_a_v3_2c_route_v2_outcome_consistency_audit
PY=/home/jgzn/新加卷/myenv/bin/python
export ROOT REPO SOURCE OUT PY

case "$OUT" in
  "$ROOT"/experiments/phase27_a_v3_2c_route_v2_outcome_consistency_audit) ;;
  *) echo "Refusing to remove unexpected OUT=$OUT" >&2; exit 2 ;;
esac

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/source"

test -x "$PY"
test -f "$SOURCE/manifests/fixed_manifest_reacquired_state_512.csv"
test -f "$SOURCE/tables/route_v2_repeat_0_on_fixed_manifest.csv"
test -f "$SOURCE/metrics/deterministic_repeatability_metrics.json"
test -f "$SOURCE/identity/metrics/fixed_manifest_eval_identity_audit_tiny.json"

cp "$SOURCE/manifests/fixed_manifest_reacquired_state_512.csv" "$OUT/manifests/fixed_manifest_reacquired_state_512.csv"
cp "$SOURCE/tables/route_v2_repeat_0_on_fixed_manifest.csv" "$OUT/source/route_v2_repeat_0_on_fixed_manifest.csv"
cp "$SOURCE/metrics/deterministic_repeatability_metrics.json" "$OUT/source/deterministic_repeatability_metrics.json"
cp "$SOURCE/identity/metrics/fixed_manifest_eval_identity_audit_tiny.json" "$OUT/source/fixed_manifest_eval_identity_audit_tiny.json"

"$PY" -m scripts.phase27_a_v3_2c_route_v2_outcome_consistency_audit \
  --manifest "$OUT/manifests/fixed_manifest_reacquired_state_512.csv" \
  --baseline "$OUT/source/route_v2_repeat_0_on_fixed_manifest.csv" \
  --repeat-metrics "$OUT/source/deterministic_repeatability_metrics.json" \
  --identity-metrics "$OUT/source/fixed_manifest_eval_identity_audit_tiny.json" \
  --output-dir "$OUT"

"$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
metrics = json.loads((out / "metrics" / "route_v2_outcome_consistency_metrics.json").read_text(encoding="utf-8"))
states = list(csv.DictReader((out / "tables" / "route_v2_state_wise_outcome_consistency.csv").open(encoding="utf-8")))
pairs = list(csv.DictReader((out / "tables" / "route_v2_pairwise_state_separation.csv").open(encoding="utf-8")))
lines = [
    "# Phase27 A-v3.2c Route-v2 Outcome-Consistency Audit Record",
    "",
    f"status: `{metrics['verdict']}`",
    "",
    f"- source_dir: `{os.environ['SOURCE']}`",
    f"- output_dir: `{out}`",
    f"- manifest_rows: `{metrics['manifest_rows']}`",
    f"- joined_ok_count: `{metrics['joined_ok_count']}`",
    f"- joined_fraction: `{metrics['joined_fraction']:.6f}`",
    f"- state_count: `{metrics['state_count']}`",
    f"- repeatability_verdict: `{metrics['repeatability_verdict']}`",
    f"- identity_verdict: `{metrics['identity_verdict']}`",
    f"- intervention_consistency_status: `{metrics['intervention_consistency_status']}`",
    f"- max_state_mean_gap: `{metrics['max_state_mean_gap']:.12g}`",
    f"- max_abs_cliff_delta: `{metrics['max_abs_cliff_delta']:.12g}`",
    f"- reason: `{metrics['reason']}`",
    "",
    "## State Summary",
]
for row in states:
    lines.append(
        f"- {row['state']}: count={row['count']}, mean={row['mean_error']}, "
        f"median={row['median_error']}, top_quartile_fraction={row['top_quartile_error_fraction']}"
    )
lines.extend(["", "## Pairwise Separation"])
for row in pairs:
    lines.append(
        f"- {row['state_a']} vs {row['state_b']}: mean_delta={row['mean_a_minus_b']}, "
        f"cliff_delta={row['cliff_delta_a_harder_than_b']}"
    )
lines.extend([
    "",
    "No training, finetuning, sample weighting, threshold tuning, B/C gate, full eval, submission packaging, fuzzy join, silent deduplication, or leaderboard probing was run.",
])
(out / "phase27_a_v3_2c_route_v2_outcome_consistency_audit_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

for f in \
  manifests/fixed_manifest_reacquired_state_512.csv \
  source/route_v2_repeat_0_on_fixed_manifest.csv \
  source/deterministic_repeatability_metrics.json \
  source/fixed_manifest_eval_identity_audit_tiny.json \
  tables/route_v2_shared_baseline_outcome_surface.csv \
  tables/route_v2_state_wise_outcome_consistency.csv \
  tables/route_v2_pairwise_state_separation.csv \
  tables/route_v2_target_bias_report.csv \
  metrics/route_v2_outcome_consistency_metrics.json \
  reports/route_v2_outcome_consistency_report.md \
  phase27_a_v3_2c_route_v2_outcome_consistency_audit_record.md; do
  test -f "$OUT/$f"
done

echo A_V3_2C_ROUTE_V2_OUTCOME_CONSISTENCY_AUDIT_COMPLETE
