#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2c_bounded_outcome_consistency
SOURCE=$ROOT/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition/manifests/fixed_shared_pair_manifest_bounded.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
IMAGE_ROOT=$ROOT/official/UAVM_2026/pairUAV/train_tour
CKPT=$ROOT/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth
PY=/home/jgzn/新加卷/myenv/bin/python
LIMIT=${LIMIT:-512}
export OUT CKPT IMAGE_ROOT LIMIT SOURCE FULL

case "$OUT" in
  "$ROOT"/experiments/phase27_a_v3_2c_bounded_outcome_consistency) ;;
  *) echo "Refusing to remove unexpected OUT=$OUT" >&2; exit 2 ;;
esac

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs"

test -f "$SOURCE"
test -f "$FULL"
test -d "$IMAGE_ROOT"
test -f "$CKPT"
test -x "$PY"

"$PY" -m unittest discover -s tests -p 'test_phase27_a_v3_2c_*.py' -v

"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_tiny \
  --source-manifest "$SOURCE" \
  --full-dev-surface "$FULL" \
  --limit "$LIMIT" \
  --output-dir "$OUT"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
payload = {
    "source_manifest": os.environ["SOURCE"],
    "full_dev_surface": os.environ["FULL"],
    "image_root": os.environ["IMAGE_ROOT"],
    "checkpoint": os.environ["CKPT"],
    "limit": int(os.environ["LIMIT"]),
    "scope": "bounded fixed-manifest model forward outcome-consistency audit; no training; no full eval",
}
(out / "input_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

for variant in baseline stress64030429 stress64030429_22181448 stress64030429_94572967 stress64030429_99516045; do
  cat > "$OUT/configs/${variant}.json" <<JSON
{"variant_id":"$variant","mode":"bounded_model_forward","no_training":true,"limit":$LIMIT}
JSON
  "$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_eval_runner \
    --fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv" \
    --image-root "$IMAGE_ROOT" \
    --checkpoint "$CKPT" \
    --output-csv "$OUT/tables/${variant}_on_fixed_manifest_bounded.csv" \
    --variant-id "$variant" \
    --variant-config "$OUT/configs/${variant}.json" \
    --mode model_forward_tiny \
    --batch-size 4 \
    --num-workers 0 \
    --device auto \
    --max-samples 0
done

"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_identity_audit \
  --fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv" \
  --outcome "baseline=$OUT/tables/baseline_on_fixed_manifest_bounded.csv" \
  --outcome "stress64030429=$OUT/tables/stress64030429_on_fixed_manifest_bounded.csv" \
  --outcome "stress64030429_22181448=$OUT/tables/stress64030429_22181448_on_fixed_manifest_bounded.csv" \
  --outcome "stress64030429_94572967=$OUT/tables/stress64030429_94572967_on_fixed_manifest_bounded.csv" \
  --outcome "stress64030429_99516045=$OUT/tables/stress64030429_99516045_on_fixed_manifest_bounded.csv" \
  --output-dir "$OUT"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
metrics = json.loads((out / "metrics" / "fixed_manifest_eval_identity_audit_tiny.json").read_text(encoding="utf-8"))
if metrics.get("verdict") != "fixed-manifest-runner-smoke-pass":
    (out / "reports" / "bounded_go_no_go_verdict.md").write_text(
        "# A-v3.2c Bounded Outcome-Consistency Go/No-Go Verdict\n\n"
        "verdict: `a-v3-2c-bounded-validation-blocked-identity`\n"
        "reason: `identity_audit_failed_before_outcome_consistency`\n",
        encoding="utf-8",
    )
    raise SystemExit(1)
PY

"$PY" -m scripts.phase27_a_v3_2c_outcome_consistency_audit \
  --manifest "$OUT/manifests/fixed_manifest_tiny.csv" \
  --baseline "$OUT/tables/baseline_on_fixed_manifest_bounded.csv" \
  --stress "stress64030429=$OUT/tables/stress64030429_on_fixed_manifest_bounded.csv" \
  --stress "stress64030429_22181448=$OUT/tables/stress64030429_22181448_on_fixed_manifest_bounded.csv" \
  --stress "stress64030429_94572967=$OUT/tables/stress64030429_94572967_on_fixed_manifest_bounded.csv" \
  --stress "stress64030429_99516045=$OUT/tables/stress64030429_99516045_on_fixed_manifest_bounded.csv" \
  --output-dir "$OUT" \
  --min-shared-fraction 0.90 \
  --min-prediction-success-fraction 0.95

"$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
metrics = json.loads((out / "metrics" / "outcome_consistency_metrics.json").read_text(encoding="utf-8"))
states = list(csv.DictReader((out / "tables" / "state_wise_outcome_report.csv").open(encoding="utf-8")))
lines = [
    "# Phase27 A-v3.2c Bounded Outcome-Consistency Experiment Record",
    "",
    f"status: `{metrics['verdict']}`",
    "",
    f"- limit: `{os.environ['LIMIT']}`",
    f"- checkpoint: `{os.environ['CKPT']}`",
    f"- image_root: `{os.environ['IMAGE_ROOT']}`",
    f"- source_manifest: `{os.environ['SOURCE']}`",
    f"- full_dev_surface: `{os.environ['FULL']}`",
    f"- output_dir: `{out}`",
    "",
    "## Metrics",
    "",
    f"- manifest_row_count: `{metrics['manifest_row_count']}`",
    f"- variant_count: `{metrics['variant_count']}`",
    f"- shared_outcome_count: `{metrics['shared_outcome_count']}`",
    f"- shared_fraction: `{metrics['shared_fraction']:.6f}`",
    f"- min_prediction_success_fraction: `{metrics['min_prediction_success_fraction']:.6f}`",
    f"- reason: `{metrics['reason']}`",
    "",
    "## State Headline",
]
for row in states[:20]:
    lines.append(
        f"- {row['state']}: count={row['count']}, shared={row['shared_outcome_count']}, "
        f"mean_error={row['mean_error']}, stress_delta_mean={row['stress_delta_mean']}, note={row['verdict_note']}"
    )
lines.extend([
    "",
    "No training, finetuning, sample weighting, curriculum, checkpoint selection, threshold tuning, B/C gate construction, full eval, submission packaging, fuzzy join, silent deduplication, or leaderboard probing was run.",
])
(out / "phase27_a_v3_2c_bounded_outcome_consistency_experiment_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

for f in \
  input_manifest.json \
  manifests/fixed_manifest_tiny.csv \
  metrics/fixed_manifest_tiny_metrics.json \
  metrics/fixed_manifest_eval_identity_audit_tiny.json \
  metrics/outcome_consistency_metrics.json \
  tables/baseline_on_fixed_manifest_bounded.csv \
  tables/stress64030429_on_fixed_manifest_bounded.csv \
  tables/stress64030429_22181448_on_fixed_manifest_bounded.csv \
  tables/stress64030429_94572967_on_fixed_manifest_bounded.csv \
  tables/stress64030429_99516045_on_fixed_manifest_bounded.csv \
  tables/shared_outcome_surface.csv \
  tables/state_wise_outcome_report.csv \
  tables/variant_summary.csv \
  tables/target_coverage_bias.csv \
  reports/fixed_manifest_eval_identity_audit_tiny.md \
  reports/outcome_consistency_report.md \
  reports/bounded_go_no_go_verdict.md \
  phase27_a_v3_2c_bounded_outcome_consistency_experiment_record.md; do
  test -f "$OUT/$f"
done

cat "$OUT/reports/bounded_go_no_go_verdict.md"
echo A_V3_2C_BOUNDED_OUTCOME_CONSISTENCY_COMPLETE
