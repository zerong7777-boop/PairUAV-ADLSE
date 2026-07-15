#!/usr/bin/env python3
"""Write Phase27 A evidence-state manifest v2 report and record."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--record-out", required=True, type=Path)
    return parser.parse_args()


def _read_manifest_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _dict_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"## {title}", "", "| Key | Value |", "|---|---:|"]
    for key, value in rows.items():
        lines.append(f"| `{key}` | `{_fmt(value)}` |")
    return "\n".join(lines)


def _per_base_table(metrics: dict[str, Any]) -> str:
    per_base = metrics.get("joined_validation", {}).get("per_base_validation", {})
    lines = [
        "## Per-Base Joined Validation",
        "",
        "| Base Regime | Count | Hard Labels | Control Labels | B-SCR Delta Mean | Gate-Off Delta Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime, row in per_base.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                regime,
                row.get("count", 0),
                row.get("hard_label_count", 0),
                row.get("control_label_count", 0),
                _fmt(row.get("bscr_delta_mean")),
                _fmt(row.get("gate_off_delta_mean")),
            )
        )
    return "\n".join(lines)


def _body(metrics: dict[str, Any], manifest: Path, project_root: Path) -> str:
    joined = metrics.get("joined_validation", {})
    v1 = metrics.get("v1_comparison", {})
    coverage = metrics.get("coverage_audit", {})
    reasons = metrics.get("verdict_reasons", [])
    feature_rows = coverage.get("feature_rows", {})
    identity_rows = coverage.get("identity_rows", {})
    return f"""Date: `{datetime.now(timezone.utc).isoformat()}`

Task: `20260417-uavm-pairuav-codabench`

Phase: `phase-27-a-evidence-state-manifest-redesign-v2`

Claim level: `{metrics.get("claim_level")}`

Verdict: `{metrics.get("verdict")}`

## Inputs

- Manifest: `{manifest}`
- Metrics: `{project_root / 'experiments/phase27_a_evidence_state_manifest/metrics/a_evidence_state_manifest_v2_metrics.json'}`
- Construction source: frozen B-SCR feature packets only.
- Validation-only sources: Phase26 per-sample official metrics and Phase27 v1 metrics.

No training, model change, checkpoint update, loss change, sample weighting, inference change, or submission package was performed.

## Key Results

- Bounded rows: `{metrics.get("row_count")}`
- Joined validation pairs: `{joined.get("joined_pair_count")}`
- Hard validation count: `{joined.get("hard_count")}`
- Control validation count: `{joined.get("control_count")}`
- Hard B-SCR delta mean: `{_fmt(joined.get("hard_bscr_delta_mean"))}`
- Control B-SCR delta mean: `{_fmt(joined.get("control_bscr_delta_mean"))}`
- v2 `ordinary_control_anchor`: `{metrics.get("base_regime_counts", {}).get("ordinary_control_anchor")}`
- v2 `ordinary_control_anchor` fraction: `{_fmt(metrics.get("ordinary_control_anchor_fraction"))}`
- v1 `ordinary_control`: `{v1.get("ordinary_control_count")}`
- v1 `ordinary_control` fraction: `{_fmt(v1.get("ordinary_control_fraction"))}`
- v2 improved anchor coverage over v1: `{v1.get("improved_anchor_coverage")}`

{_dict_table("Base-Regime Counts", metrics.get("base_regime_counts", {}))}

{_dict_table("Risk-Tag Counts", metrics.get("risk_tag_counts", {}))}

{_per_base_table(metrics)}

## Coverage Audit

- Train feature rows: `{feature_rows.get("train")}`
- Eval feature rows: `{feature_rows.get("eval")}`
- Train identity rows: `{identity_rows.get("train")}`
- Dev identity rows: `{identity_rows.get("dev")}`
- Train feature-to-identity fraction: `{_fmt(coverage.get("train_feature_to_identity_fraction"))}`
- Dev feature-to-identity fraction: `{_fmt(coverage.get("dev_feature_to_identity_fraction"))}`
- Feasible beyond eval packet: `{coverage.get("feasible_beyond_eval_packet")}`
- Missing observable feature reason: `{coverage.get("missing_observable_feature_reason")}`

## Leakage and Invariants

- Leakage audit passed: `{metrics.get("leakage_audit", {}).get("passed")}`
- Forbidden construction columns: `{metrics.get("leakage_audit", {}).get("forbidden_columns")}`
- Invariant violations: `{metrics.get("invariant_violations")}`

## Interpretation

v2 fixes the v1 structural failure. The ordinary/control anchor is no longer residual: it covers `13/33` bounded rows versus v1's `1/33`, and the exact-one-base-regime invariant holds.

The result is still not ready for A training-policy implementation because observable feature coverage is sparse: frozen B-SCR packets cover `545` train rows and `33` eval rows compared with `1,839,996` train identity rows and `204,120` dev identity rows.

## Verdict Reasons

{chr(10).join(f'- {reason}' for reason in reasons)}

## Decision

Do not start A-side training-policy implementation yet.

Recommended knowledge-review decision: `manifest-needs-another-redesign` or `manifest-ready-only-after-feature-coverage-expansion`, depending on whether the next phase is treated as manifest redesign or non-leaky feature-generation expansion.
"""


def main() -> int:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest_count = _read_manifest_count(args.manifest)
    if manifest_count != metrics.get("row_count"):
        raise SystemExit(f"manifest row count mismatch: {manifest_count} != {metrics.get('row_count')}")
    report = "# Phase27 A Evidence-State Manifest v2 Report\n\n" + _body(metrics, args.manifest, args.project_root)
    record = "# exp-20260507 Phase27 A Evidence-State Manifest Redesign v2\n\n" + _body(metrics, args.manifest, args.project_root)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    args.record_out.write_text(record, encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "record": str(args.record_out), "verdict": metrics.get("verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
