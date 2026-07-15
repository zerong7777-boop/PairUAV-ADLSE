#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2c_fixed_manifest_pairuav_eval_runner_tiny_smoke
SOURCE=$ROOT/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition/manifests/fixed_shared_pair_manifest_bounded.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs"

python3 -m unittest \
  tests.test_phase27_a_v3_2c_fixed_manifest_eval_runner \
  tests.test_phase27_a_v3_2c_fixed_manifest_identity_audit -v

python3 -m scripts.phase27_a_v3_2c_fixed_manifest_tiny \
  --source-manifest "$SOURCE" \
  --full-dev-surface "$FULL" \
  --limit 16 \
  --output-dir "$OUT"

for variant in baseline stress64030429 stress64030429_22181448 stress64030429_94572967 stress64030429_99516045; do
  cat > "$OUT/configs/${variant}.json" <<JSON
{"variant_id":"$variant","mode":"dry_run_identity_only","no_model_forward":true}
JSON
  python3 -m scripts.phase27_a_v3_2c_fixed_manifest_eval_runner \
    --fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv" \
    --image-root "$ROOT" \
    --checkpoint "" \
    --output-csv "$OUT/tables/${variant}_on_fixed_manifest_tiny.csv" \
    --variant-id "$variant" \
    --variant-config "$OUT/configs/${variant}.json" \
    --mode dry_run_identity_only \
    --batch-size 1 \
    --num-workers 0 \
    --device cpu \
    --max-samples 0
done

AUDIT_ARGS=(--fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv")
for variant in baseline stress64030429 stress64030429_22181448 stress64030429_94572967 stress64030429_99516045; do
  AUDIT_ARGS+=(--outcome "$variant=$OUT/tables/${variant}_on_fixed_manifest_tiny.csv")
done
python3 -m scripts.phase27_a_v3_2c_fixed_manifest_identity_audit "${AUDIT_ARGS[@]}" --output-dir "$OUT"

for f in \
  manifests/fixed_manifest_tiny.csv \
  metrics/fixed_manifest_tiny_metrics.json \
  tables/baseline_on_fixed_manifest_tiny.csv \
  tables/stress64030429_on_fixed_manifest_tiny.csv \
  tables/stress64030429_22181448_on_fixed_manifest_tiny.csv \
  tables/stress64030429_94572967_on_fixed_manifest_tiny.csv \
  tables/stress64030429_99516045_on_fixed_manifest_tiny.csv \
  tables/fixed_manifest_eval_identity_audit_tiny.csv \
  metrics/fixed_manifest_eval_identity_audit_tiny.json \
  reports/fixed_manifest_eval_identity_audit_tiny.md \
  reports/runner_go_no_go_verdict.md; do
  test -f "$OUT/$f"
done

cat "$OUT/reports/runner_go_no_go_verdict.md"
echo A_V3_2C_TINY_SMOKE_OK

