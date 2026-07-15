#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
SOURCE=$ROOT/experiments/phase27_a_v3_2c_bounded_outcome_consistency_reacquired_state/manifests/fixed_manifest_reacquired_state.csv
OUT=$ROOT/experiments/phase27_a_v3_2c_deterministic_repeatability_audit
IMAGE_ROOT=$ROOT/official/UAVM_2026/pairUAV/train_tour
CKPT=$ROOT/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth
PY=/home/jgzn/新加卷/myenv/bin/python
LIMIT=${LIMIT:-128}
REPEATS=${REPEATS:-3}
export OUT CKPT IMAGE_ROOT LIMIT REPEATS SOURCE

case "$OUT" in
  "$ROOT"/experiments/phase27_a_v3_2c_deterministic_repeatability_audit) ;;
  *) echo "Refusing to remove unexpected OUT=$OUT" >&2; exit 2 ;;
esac

cd "$REPO"
rm -rf "$OUT"
mkdir -p "$OUT/manifests" "$OUT/tables" "$OUT/metrics" "$OUT/reports" "$OUT/configs"

test -f "$SOURCE"
test -d "$IMAGE_ROOT"
test -f "$CKPT"
test -x "$PY"

"$PY" - <<'PY'
import csv
import os
from pathlib import Path

source = Path(os.environ["SOURCE"])
out = Path(os.environ["OUT"])
limit = int(os.environ["LIMIT"])
rows = []
with source.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for i, row in enumerate(reader):
        if i >= limit:
            break
        rows.append(row)
target = out / "manifests" / "fixed_manifest_repeatability.csv"
with target.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
PY

cat > "$OUT/configs/repeat_same_config.json" <<JSON
{"variant_id":"repeat_same_config","mode":"deterministic_repeatability","no_training":true,"limit":$LIMIT}
JSON

"$PY" - <<'PY'
import json
import os
import torch
from pathlib import Path
from reloc3r.reloc3r_relpose import Reloc3rRelpose
from scripts.phase27_a_v3_2c_fixed_manifest_eval_runner import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Reloc3rRelpose(img_size=512, output_mode="pairuav_heading_range")
result = load_checkpoint(model, os.environ["CKPT"], device)
payload = {
    "missing_key_count": len(result.missing_keys),
    "unexpected_key_count": len(result.unexpected_keys),
    "missing_key_sample": list(result.missing_keys[:50]),
    "unexpected_key_sample": list(result.unexpected_keys[:50]),
}
out = Path(os.environ["OUT"])
(out / "metrics" / "checkpoint_load_diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

for idx in $(seq 0 $((REPEATS - 1))); do
  "$PY" -m scripts.phase27_a_v3_2c_fixed_manifest_eval_runner \
    --fixed-manifest "$OUT/manifests/fixed_manifest_repeatability.csv" \
    --image-root "$IMAGE_ROOT" \
    --checkpoint "$CKPT" \
    --output-csv "$OUT/tables/repeat_${idx}_on_fixed_manifest.csv" \
    --variant-id repeat_same_config \
    --variant-config "$OUT/configs/repeat_same_config.json" \
    --mode model_forward_tiny \
    --batch-size 4 \
    --num-workers 0 \
    --device auto \
    --max-samples 0
done

AUDIT_ARGS=()
for idx in $(seq 0 $((REPEATS - 1))); do
  AUDIT_ARGS+=(--run "repeat_${idx}=$OUT/tables/repeat_${idx}_on_fixed_manifest.csv")
done

"$PY" -m scripts.phase27_a_v3_2c_deterministic_repeatability_audit \
  --output-dir "$OUT" \
  --load-diagnostics "$OUT/metrics/checkpoint_load_diagnostics.json" \
  "${AUDIT_ARGS[@]}"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
metrics = json.loads((out / "metrics" / "deterministic_repeatability_metrics.json").read_text(encoding="utf-8"))
lines = [
    "# Phase27 A-v3.2c Deterministic Repeatability Experiment Record",
    "",
    f"status: `{metrics['verdict']}`",
    "",
    f"- limit: `{os.environ['LIMIT']}`",
    f"- repeats: `{os.environ['REPEATS']}`",
    f"- checkpoint: `{os.environ['CKPT']}`",
    f"- image_root: `{os.environ['IMAGE_ROOT']}`",
    f"- manifest: `{out / 'manifests' / 'fixed_manifest_repeatability.csv'}`",
    "",
    "## Metrics",
    "",
    f"- min_same_prediction_fraction: `{metrics['min_same_prediction_fraction']:.6f}`",
    f"- max_heading_delta: `{metrics['max_heading_delta']:.6f}`",
    f"- max_range_delta_p95: `{metrics['max_range_delta_p95']:.6f}`",
    f"- missing_key_count: `{metrics.get('load_diagnostics', {}).get('missing_key_count', '')}`",
    f"- unexpected_key_count: `{metrics.get('load_diagnostics', {}).get('unexpected_key_count', '')}`",
    f"- reason: `{metrics['reason']}`",
    "",
    "No training, finetuning, sample weighting, threshold tuning, full eval, B/C gate, submission packaging, fuzzy join, or silent deduplication was run.",
]
(out / "phase27_a_v3_2c_deterministic_repeatability_experiment_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

for f in \
  manifests/fixed_manifest_repeatability.csv \
  configs/repeat_same_config.json \
  metrics/checkpoint_load_diagnostics.json \
  metrics/deterministic_repeatability_metrics.json \
  tables/repeatability_delta_summary.csv \
  tables/repeatability_pair_delta_sample.csv \
  reports/deterministic_repeatability_audit_report.md \
  phase27_a_v3_2c_deterministic_repeatability_experiment_record.md; do
  test -f "$OUT/$f"
done

cat "$OUT/reports/deterministic_repeatability_audit_report.md"
echo A_V3_2C_DETERMINISTIC_REPEATABILITY_AUDIT_COMPLETE
