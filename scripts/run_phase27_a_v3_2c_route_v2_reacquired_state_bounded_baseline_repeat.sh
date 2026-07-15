#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
SOURCE=$ROOT/experiments/phase27_a_v3_2c_bounded_outcome_consistency_reacquired_state/manifests/fixed_manifest_reacquired_state.csv
OUT=$ROOT/experiments/phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat
IMAGE_ROOT=$ROOT/official/UAVM_2026/pairUAV/train_tour
MODEL_PATH=$ROOT/official/UAVM_2026/models/dino_resnet
CKPT=$ROOT/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth
PY=/home/jgzn/新加卷/myenv/bin/python
LIMIT=${LIMIT:-512}
REPEATS=${REPEATS:-3}
export OUT CKPT IMAGE_ROOT MODEL_PATH LIMIT REPEATS SOURCE

case "$OUT" in
  "$ROOT"/experiments/phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat) ;;
  *) echo "Refusing to remove unexpected OUT=$OUT" >&2; exit 2 ;;
esac

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs" "$OUT/identity"

test -f "$SOURCE"
test -d "$IMAGE_ROOT"
test -d "$MODEL_PATH"
test -f "$CKPT"
test -x "$PY"

"$PY" /tmp/make_manifest_subset.py \
  --src "$SOURCE" \
  --dst "$OUT/manifests/fixed_manifest_reacquired_state_${LIMIT}.csv" \
  --limit "$LIMIT"

cat > "$OUT/configs/route_v2_repeat_same_config.json" <<JSON
{"variant_id":"route_v2_repeat_same_config","mode":"route_v2_reacquired_state_bounded_baseline_repeat","no_training":true,"limit":$LIMIT}
JSON

for idx in $(seq 0 $((REPEATS - 1))); do
  "$PY" scripts/phase27_a_v3_2c_route_v2_fixed_manifest_eval_runner.py \
    --fixed-manifest "$OUT/manifests/fixed_manifest_reacquired_state_${LIMIT}.csv" \
    --checkpoint "$CKPT" \
    --image-root "$IMAGE_ROOT" \
    --model-path "$MODEL_PATH" \
    --output-csv "$OUT/tables/route_v2_repeat_${idx}_on_fixed_manifest.csv" \
    --diagnostics-json "$OUT/metrics/route_v2_repeat_${idx}_diagnostics.json" \
    --variant-id route_v2_repeat_same_config \
    --variant-config "$OUT/configs/route_v2_repeat_same_config.json" \
    --batch-size 16 \
    --max-samples 0 \
    --device auto
done

mkdir -p "$OUT/identity/tables" "$OUT/identity/metrics" "$OUT/identity/reports"
"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_identity_audit \
  --fixed-manifest "$OUT/manifests/fixed_manifest_reacquired_state_${LIMIT}.csv" \
  --outcome "route_v2_repeat_0=$OUT/tables/route_v2_repeat_0_on_fixed_manifest.csv" \
  --outcome "route_v2_repeat_1=$OUT/tables/route_v2_repeat_1_on_fixed_manifest.csv" \
  --outcome "route_v2_repeat_2=$OUT/tables/route_v2_repeat_2_on_fixed_manifest.csv" \
  --output-dir "$OUT/identity"

AUDIT_ARGS=()
for idx in $(seq 0 $((REPEATS - 1))); do
  AUDIT_ARGS+=(--run "route_v2_repeat_${idx}=$OUT/tables/route_v2_repeat_${idx}_on_fixed_manifest.csv")
done
"$PY" -m scripts.phase27_a_v3_2c_deterministic_repeatability_audit \
  --output-dir "$OUT" \
  --load-diagnostics "$OUT/metrics/route_v2_repeat_0_diagnostics.json" \
  "${AUDIT_ARGS[@]}"

"$PY" -m scripts.phase27_a_v3_2c_route_v2_reacquired_state_baseline_report \
  --manifest "$OUT/manifests/fixed_manifest_reacquired_state_${LIMIT}.csv" \
  --baseline "$OUT/tables/route_v2_repeat_0_on_fixed_manifest.csv" \
  --output-dir "$OUT"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
repeat = json.loads((out / "metrics" / "deterministic_repeatability_metrics.json").read_text(encoding="utf-8"))
state = json.loads((out / "metrics" / "route_v2_state_wise_baseline_metrics.json").read_text(encoding="utf-8"))
identity = json.loads((out / "identity" / "metrics" / "fixed_manifest_eval_identity_audit_tiny.json").read_text(encoding="utf-8"))
verdict = "route-v2-reacquired-state-baseline-repeat-pass" if repeat["verdict"] == "deterministic-repeatability-pass" and identity["verdict"] == "fixed-manifest-runner-smoke-pass" and state["state_count"] >= 2 else "route-v2-reacquired-state-baseline-repeat-weak"
overall_mean_error = state.get("overall_mean_error")
overall_mean_error_text = "" if overall_mean_error is None else f"{overall_mean_error:.12g}"
lines = [
    "# Phase27 A-v3.2c Route-v2 Reacquired-State Bounded Baseline Repeat Record",
    "",
    f"status: `{verdict}`",
    "",
    f"- limit: `{os.environ['LIMIT']}`",
    f"- repeats: `{os.environ['REPEATS']}`",
    f"- checkpoint: `{os.environ['CKPT']}`",
    f"- model_path: `{os.environ['MODEL_PATH']}`",
    f"- manifest: `{out / 'manifests' / ('fixed_manifest_reacquired_state_' + os.environ['LIMIT'] + '.csv')}`",
    "",
    "## Metrics",
    "",
    f"- identity_verdict: `{identity['verdict']}`",
    f"- repeatability_verdict: `{repeat['verdict']}`",
    f"- min_same_prediction_fraction: `{repeat['min_same_prediction_fraction']:.6f}`",
    f"- max_heading_delta: `{repeat['max_heading_delta']:.6f}`",
    f"- state_report_verdict: `{state['verdict']}`",
    f"- state_count: `{state['state_count']}`",
    f"- overall_mean_error: `{overall_mean_error_text}`",
    "",
    "No training, finetuning, stress protocol, threshold tuning, full eval, B/C gate, submission packaging, fuzzy join, or silent deduplication was run.",
]
(out / "phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

for f in \
  manifests/fixed_manifest_reacquired_state_${LIMIT}.csv \
  tables/route_v2_repeat_0_on_fixed_manifest.csv \
  tables/route_v2_repeat_1_on_fixed_manifest.csv \
  tables/route_v2_repeat_2_on_fixed_manifest.csv \
  tables/repeatability_delta_summary.csv \
  tables/route_v2_state_wise_baseline_report.csv \
  metrics/deterministic_repeatability_metrics.json \
  metrics/route_v2_state_wise_baseline_metrics.json \
  reports/deterministic_repeatability_audit_report.md \
  reports/route_v2_state_wise_baseline_report.md \
  identity/reports/runner_go_no_go_verdict.md \
  phase27_a_v3_2c_route_v2_reacquired_state_bounded_baseline_repeat_record.md; do
  test -f "$OUT/$f"
done

echo A_V3_2C_ROUTE_V2_REACQUIRED_STATE_BOUNDED_BASELINE_REPEAT_COMPLETE
