#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2c_route_v2_multitarget_baseline_audit
SOURCE=$ROOT/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition/manifests/fixed_shared_pair_manifest_bounded.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
EVIDENCE=$ROOT/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_calibrated_v2.csv
TRAINING=$ROOT/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv
IMAGE_ROOT=$ROOT/official/UAVM_2026/pairUAV/train_tour
MODEL_PATH=$ROOT/official/UAVM_2026/models/dino_resnet
CKPT=$ROOT/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth
PY=/home/jgzn/新加卷/myenv/bin/python
PER_TARGET=${PER_TARGET:-512}
export ROOT REPO OUT SOURCE FULL EVIDENCE TRAINING IMAGE_ROOT MODEL_PATH CKPT PY PER_TARGET

case "$OUT" in
  "$ROOT"/experiments/phase27_a_v3_2c_route_v2_multitarget_baseline_audit) ;;
  *) echo "Refusing to remove unexpected OUT=$OUT" >&2; exit 2 ;;
esac

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs" "$OUT/identity"

test -x "$PY"
test -f "$SOURCE"
test -f "$FULL"
test -f "$EVIDENCE"
test -f "$TRAINING"
test -d "$IMAGE_ROOT"
test -d "$MODEL_PATH"
test -f "$CKPT"

"$PY" -m scripts.phase27_a_v3_2c_route_v2_multitarget_manifest_builder \
  --source-manifest "$SOURCE" \
  --full-dev-surface "$FULL" \
  --evidence-manifest "$EVIDENCE" \
  --training-manifest "$TRAINING" \
  --per-target "$PER_TARGET" \
  --output-manifest "$OUT/manifests/fixed_manifest_route_v2_multitarget.csv" \
  --metrics-json "$OUT/metrics/route_v2_multitarget_manifest_metrics.json"

cat > "$OUT/configs/route_v2_multitarget_baseline.json" <<JSON
{"variant_id":"route_v2_multitarget_baseline","mode":"route_v2_multitarget_baseline_audit","no_training":true,"per_target":$PER_TARGET}
JSON

"$PY" -m scripts.phase27_a_v3_2c_route_v2_fixed_manifest_eval_runner \
  --fixed-manifest "$OUT/manifests/fixed_manifest_route_v2_multitarget.csv" \
  --checkpoint "$CKPT" \
  --image-root "$IMAGE_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-csv "$OUT/tables/route_v2_multitarget_baseline_on_fixed_manifest.csv" \
  --diagnostics-json "$OUT/metrics/route_v2_multitarget_baseline_diagnostics.json" \
  --variant-id route_v2_multitarget_baseline \
  --variant-config "$OUT/configs/route_v2_multitarget_baseline.json" \
  --batch-size 16 \
  --max-samples 0 \
  --device auto

mkdir -p "$OUT/identity/tables" "$OUT/identity/metrics" "$OUT/identity/reports"
"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_identity_audit \
  --fixed-manifest "$OUT/manifests/fixed_manifest_route_v2_multitarget.csv" \
  --outcome "route_v2_multitarget_baseline=$OUT/tables/route_v2_multitarget_baseline_on_fixed_manifest.csv" \
  --output-dir "$OUT/identity"

"$PY" -m scripts.phase27_a_v3_2c_route_v2_outcome_consistency_audit \
  --manifest "$OUT/manifests/fixed_manifest_route_v2_multitarget.csv" \
  --baseline "$OUT/tables/route_v2_multitarget_baseline_on_fixed_manifest.csv" \
  --repeat-metrics "$ROOT/experiments/phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat/metrics/deterministic_repeatability_metrics.json" \
  --identity-metrics "$OUT/identity/metrics/fixed_manifest_eval_identity_audit_tiny.json" \
  --output-dir "$OUT"

"$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
manifest = json.loads((out / "metrics" / "route_v2_multitarget_manifest_metrics.json").read_text(encoding="utf-8"))
diag = json.loads((out / "metrics" / "route_v2_multitarget_baseline_diagnostics.json").read_text(encoding="utf-8"))
metrics = json.loads((out / "metrics" / "route_v2_outcome_consistency_metrics.json").read_text(encoding="utf-8"))
states = list(csv.DictReader((out / "tables" / "route_v2_state_wise_outcome_consistency.csv").open(encoding="utf-8")))
targets = list(csv.DictReader((out / "tables" / "route_v2_target_bias_report.csv").open(encoding="utf-8")))
lines = [
    "# Phase27 A-v3.2c Route-v2 Multitarget Baseline Audit Record",
    "",
    f"status: `{metrics['verdict']}`",
    "",
    f"- output_dir: `{out}`",
    f"- per_target: `{os.environ['PER_TARGET']}`",
    f"- manifest_rows: `{manifest['row_count']}`",
    f"- target_count: `{manifest['target_count']}`",
    f"- target_distribution: `{json.dumps(manifest['target_distribution'], sort_keys=True)}`",
    f"- state_count: `{manifest['state_count']}`",
    f"- state_distribution: `{json.dumps(manifest['state_distribution'], sort_keys=True)}`",
    f"- strict_load_missing: `{diag['missing_key_count']}`",
    f"- strict_load_unexpected: `{diag['unexpected_key_count']}`",
    f"- joined_fraction: `{metrics['joined_fraction']:.6f}`",
    f"- max_state_mean_gap: `{metrics['max_state_mean_gap']:.12g}`",
    f"- max_abs_cliff_delta: `{metrics['max_abs_cliff_delta']:.12g}`",
    f"- intervention_consistency_status: `{metrics['intervention_consistency_status']}`",
    f"- reason: `{metrics['reason']}`",
    "",
    "## State Summary",
]
for row in states:
    lines.append(
        f"- {row['state']}: count={row['count']}, mean={row['mean_error']}, "
        f"median={row['median_error']}, top_quartile_fraction={row['top_quartile_error_fraction']}"
    )
lines.extend(["", "## Target Summary"])
for row in targets:
    lines.append(
        f"- {row['target_key']}: count={row['count']}, mean={row['mean_error']}, "
        f"states={row['state_distribution']}"
    )
lines.extend([
    "",
    "No training, finetuning, sample weighting, threshold tuning, B/C gate, full eval, submission packaging, fuzzy join, silent deduplication, or leaderboard probing was run.",
])
(out / "phase27_a_v3_2c_route_v2_multitarget_baseline_audit_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

for f in \
  manifests/fixed_manifest_route_v2_multitarget.csv \
  metrics/route_v2_multitarget_manifest_metrics.json \
  metrics/route_v2_multitarget_baseline_diagnostics.json \
  metrics/route_v2_outcome_consistency_metrics.json \
  identity/reports/runner_go_no_go_verdict.md \
  tables/route_v2_multitarget_baseline_on_fixed_manifest.csv \
  tables/route_v2_shared_baseline_outcome_surface.csv \
  tables/route_v2_state_wise_outcome_consistency.csv \
  tables/route_v2_pairwise_state_separation.csv \
  tables/route_v2_target_bias_report.csv \
  reports/route_v2_outcome_consistency_report.md \
  phase27_a_v3_2c_route_v2_multitarget_baseline_audit_record.md; do
  test -f "$OUT/$f"
done

echo A_V3_2C_ROUTE_V2_MULTITARGET_BASELINE_AUDIT_COMPLETE
