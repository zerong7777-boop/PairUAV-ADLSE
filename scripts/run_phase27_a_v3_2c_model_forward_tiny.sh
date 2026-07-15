#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
OUT=$ROOT/experiments/phase27_a_v3_2c_model_forward_tiny
SOURCE=$ROOT/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition/manifests/fixed_shared_pair_manifest_bounded.csv
FULL=$ROOT/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
IMAGE_ROOT=$ROOT/official/UAVM_2026/pairUAV/train_tour
CKPT=$ROOT/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth
PY=/home/jgzn/新加卷/myenv/bin/python
export OUT CKPT IMAGE_ROOT

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs"

test -f "$CKPT"
test -d "$IMAGE_ROOT"
test -x "$PY"

"$PY" -m unittest discover -s tests -p 'test_phase27_a_v3_2c_fixed_manifest_*.py' -v

"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_tiny \
  --source-manifest "$SOURCE" \
  --full-dev-surface "$FULL" \
  --limit 16 \
  --output-dir "$OUT"

"$PY" - <<PY
from scripts.phase27_a_v3_2c_fixed_manifest_tiny import write_json
write_json("$OUT/input_manifest.json", {
  "source_manifest": "$SOURCE",
  "full_dev_surface": "$FULL",
  "image_root": "$IMAGE_ROOT",
  "checkpoint": "$CKPT",
  "mode": "model_forward_tiny",
  "scope": "tiny fixed-manifest model forward; no training; no 10k"
})
PY

for variant in baseline stress64030429 stress64030429_22181448 stress64030429_94572967 stress64030429_99516045; do
  cat > "$OUT/configs/${variant}.json" <<JSON
{"variant_id":"$variant","mode":"model_forward_tiny","no_training":true}
JSON
  "$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_eval_runner \
    --fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv" \
    --image-root "$IMAGE_ROOT" \
    --checkpoint "$CKPT" \
    --output-csv "$OUT/tables/${variant}_on_fixed_manifest_tiny.csv" \
    --variant-id "$variant" \
    --variant-config "$OUT/configs/${variant}.json" \
    --mode model_forward_tiny \
    --batch-size 4 \
    --num-workers 0 \
    --device auto \
    --max-samples 0
done

AUDIT_ARGS=(--fixed-manifest "$OUT/manifests/fixed_manifest_tiny.csv")
for variant in baseline stress64030429 stress64030429_22181448 stress64030429_94572967 stress64030429_99516045; do
  AUDIT_ARGS+=(--outcome "$variant=$OUT/tables/${variant}_on_fixed_manifest_tiny.csv")
done
"$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_identity_audit "${AUDIT_ARGS[@]}" --output-dir "$OUT"

"$PY" - <<'PY'
import csv, json
import os
import sys
from collections import Counter
from pathlib import Path
out=Path(os.environ["OUT"])
rows=[]
for path in sorted((out/"tables").glob("*_on_fixed_manifest_tiny.csv")):
    if path.name.startswith("fixed_manifest"):
        continue
    data=list(csv.DictReader(path.open()))
    rows.append((path.stem, len(data), dict(Counter(r.get("row_status","") for r in data)), sum(1 for r in data if r.get("prediction_heading"))))
lines=["# A-v3.2c Model Forward Tiny Report",""]
for name,count,status,pred in rows:
    lines.append(f"- {name}: rows={count}, predictions={pred}, row_status={json.dumps(status, sort_keys=True)}")
ok = all(count == 16 and pred == 16 and status == {"ok": 16} for name, count, status, pred in rows)
verdict = "model-forward-tiny-pass" if ok else "model-forward-tiny-fail"
reason = "all_variants_have_16_predictions_and_ok_status" if ok else "one_or_more_variants_missing_predictions_or_ok_status"
lines.append("")
lines.append(f"verdict: `{verdict}`")
lines.append(f"reason: `{reason}`")
lines.append("")
lines.append("No training, finetuning, 10k eval, outcome-consistency audit, B/C gate, fuzzy join, or silent deduplication was run.")
(out/"reports"/"model_forward_tiny_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(out/"reports"/"model_forward_tiny_verdict.md").write_text(
    f"# A-v3.2c Model Forward Tiny Verdict\n\nverdict: `{verdict}`\nreason: `{reason}`\n",
    encoding="utf-8",
)
record=[
  "# Phase27 A-v3.2c Model Forward Tiny Experiment Record",
  "",
  f"status: `{verdict}`",
  "",
  f"- checkpoint: `{os.environ['CKPT']}`",
  f"- image_root: `{os.environ['IMAGE_ROOT']}`",
  "- manifest_rows: `16`",
  "",
  "## Output Status",
  *[f"- {name}: rows={count}, predictions={pred}, row_status={json.dumps(status, sort_keys=True)}" for name,count,status,pred in rows],
  "",
  "No training, finetuning, sample weighting, curriculum, checkpoint selection, submission packaging, 10k eval, outcome-consistency audit, or B/C gate creation was run.",
]
(out/"phase27_a_v3_2c_model_forward_tiny_experiment_record.md").write_text("\n".join(record)+"\n", encoding="utf-8")
if not ok:
    sys.exit(1)
PY

for f in \
  input_manifest.json \
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
  reports/runner_go_no_go_verdict.md \
  reports/model_forward_tiny_report.md \
  reports/model_forward_tiny_verdict.md \
  phase27_a_v3_2c_model_forward_tiny_experiment_record.md; do
  test -f "$OUT/$f"
done

cat "$OUT/reports/runner_go_no_go_verdict.md"
echo A_V3_2C_MODEL_FORWARD_TINY_COMPLETE
