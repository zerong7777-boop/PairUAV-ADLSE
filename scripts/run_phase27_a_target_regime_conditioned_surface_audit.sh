#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO=$ROOT/external/reloc3r_pairuav
SOURCE=$ROOT/experiments/phase27_a_v3_2c_route_v2_multitarget_baseline_audit
OUT=$ROOT/experiments/phase27_a_target_regime_conditioned_evidence_state_validation_surface
PY=/home/jgzn/新加卷/myenv/bin/python
EXPECTED_OUT=$ROOT/experiments/phase27_a_target_regime_conditioned_evidence_state_validation_surface

if [[ "$OUT" != "$EXPECTED_OUT" ]]; then
  echo "Refusing to remove unexpected OUT path: $OUT" >&2
  exit 1
fi

if [[ -e "$OUT" ]]; then
  rm -rf "$OUT"
fi

mkdir -p "$OUT/source"
cp "$SOURCE/manifests/fixed_manifest_route_v2_multitarget.csv" "$OUT/source/"
cp "$SOURCE/tables/route_v2_multitarget_baseline_on_fixed_manifest.csv" "$OUT/source/"
cp "$SOURCE/tables/route_v2_shared_baseline_outcome_surface.csv" "$OUT/source/"
cp "$SOURCE/metrics/route_v2_multitarget_baseline_diagnostics.json" "$OUT/source/"
cp "$SOURCE/identity/reports/runner_go_no_go_verdict.md" "$OUT/source/"

cd "$REPO"
"$PY" -m unittest discover -s tests -p 'test_phase27_a_target_regime_conditioned_surface_audit.py' -v
"$PY" scripts/phase27_a_target_regime_conditioned_surface_audit.py \
  --shared-surface "$SOURCE/tables/route_v2_shared_baseline_outcome_surface.csv" \
  --output-dir "$OUT"

"$PY" - "$OUT" "$SOURCE" <<'PYRECORD'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
source = Path(sys.argv[2])
metrics = json.loads(
    (out / "metrics" / "target_regime_conditioned_surface_metrics.json").read_text(
        encoding="utf-8"
    )
)
diagnostics = json.loads(
    (
        out / "source" / "route_v2_multitarget_baseline_diagnostics.json"
    ).read_text(encoding="utf-8")
)

record = f"""# Phase27 A Target-Regime-Conditioned Evidence-State Validation Surface

status: `{metrics["verdict"]}`
source_dir: `{source}`
output_dir: `{out}`

## Checkpoint/model diagnostics copied from source

```json
{json.dumps(diagnostics, indent=2, sort_keys=True)}
```

## Target-regime definition

{metrics["target_regime_definition"]}

## Target-regime distribution

```json
{json.dumps(metrics["target_regime_distribution"], indent=2, sort_keys=True)}
```

## Evidence-state distribution

```json
{json.dumps(metrics["evidence_state_distribution"], indent=2, sort_keys=True)}
```

## Metrics

- joined_fraction: `{metrics["joined_fraction"]}`
- target_regime_mean_gap: `{metrics["target_regime_mean_gap"]}`
- max_abs_state_residual_mean: `{metrics["max_abs_state_residual_mean"]}`
- verdict: `{metrics["verdict"]}`
- reason: `{metrics["reason"]}`

## Forbidden actions confirmation

{metrics["forbidden_actions_confirmation"]}.
"""
(out / "phase27_a_target_regime_conditioned_evidence_state_validation_surface_record.md").write_text(
    record, encoding="utf-8"
)
PYRECORD

for artifact in \
  tables/global_target_report.csv \
  tables/global_evidence_state_report.csv \
  tables/target_regime_report.csv \
  tables/target_regime_evidence_state_surface.csv \
  tables/target_centered_residual_report.csv \
  tables/cell_coverage_report.csv \
  metrics/target_regime_conditioned_surface_metrics.json \
  metrics/leakage_deployability_audit.json \
  reports/target_regime_conditioned_surface_report.md \
  reports/go_no_go_verdict.md \
  phase27_a_target_regime_conditioned_evidence_state_validation_surface_record.md
do
  test -f "$OUT/$artifact" || { echo "MISSING:$artifact" >&2; exit 1; }
done

echo A_TARGET_REGIME_CONDITIONED_EVIDENCE_STATE_VALIDATION_SURFACE_COMPLETE
