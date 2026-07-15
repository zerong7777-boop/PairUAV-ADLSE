#!/usr/bin/env python3
"""Write Phase27 A evidence-state manifest report and experiment record."""

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


def _state_table(metrics: dict[str, Any]) -> str:
    counts = metrics.get("state_counts", {})
    fractions = metrics.get("state_fractions", {})
    lines = ["| State | Count | Fraction |", "|---|---:|---:|"]
    for state, count in counts.items():
        lines.append(f"| `{state}` | {count} | {_fmt(fractions.get(state, 0.0))} |")
    return "\n".join(lines)


def _validation_table(metrics: dict[str, Any]) -> str:
    per_state = metrics.get("joined_validation", {}).get("per_state_validation", {})
    lines = [
        "| State | Count | Hard Labels | Control Labels | B-SCR Delta Mean | Gate-Off Delta Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state, row in per_state.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                state,
                row.get("count", 0),
                row.get("hard_label_count", 0),
                row.get("control_label_count", 0),
                _fmt(row.get("bscr_delta_mean")),
                _fmt(row.get("gate_off_delta_mean")),
            )
        )
    return "\n".join(lines)


def _common_body(metrics: dict[str, Any], manifest: Path, project_root: Path) -> str:
    joined = metrics.get("joined_validation", {})
    reasons = metrics.get("verdict_reasons", [])
    return f"""Date: `{datetime.now(timezone.utc).isoformat()}`

Task: `20260417-uavm-pairuav-codabench`

Phase: `phase-27-a-evidence-state-manifest`

Claim level: `{metrics.get("claim_level")}`

Verdict: `{metrics.get("verdict")}`

## Inputs

- Manifest: `{manifest}`
- Metrics: `{project_root / 'experiments/phase27_a_evidence_state_manifest/metrics/a_evidence_state_metrics.json'}`
- Construction source: frozen B-SCR feature packets only.
- Validation-only sources: Phase26 per-sample official metrics and slice metrics.

No training, model change, checkpoint update, or submission package was performed.

## Manifest Coverage

- Row count: `{metrics.get("row_count")}`
- Joined validation pairs: `{joined.get("joined_pair_count")}`
- Hard validation count: `{joined.get("hard_count")}`
- Control validation count: `{joined.get("control_count")}`
- Hard B-SCR delta mean: `{_fmt(joined.get("hard_bscr_delta_mean"))}`
- Control B-SCR delta mean: `{_fmt(joined.get("control_bscr_delta_mean"))}`

{_state_table(metrics)}

## Per-State Validation

{_validation_table(metrics)}

## Leakage Audit

- Passed: `{metrics.get("leakage_audit", {}).get("passed")}`
- Forbidden construction columns: `{metrics.get("leakage_audit", {}).get("forbidden_columns")}`
- Naive semantic proxy comparison: `{joined.get("naive_semantic_proxy_comparison")}`

## Interpretation

The manifest validates the repeated B-route failure pattern at the joined-record level: hard validation rows improve on average while control validation rows degrade on average. However, the current evidence-state construction does not provide enough ordinary/control coverage to protect control cases.

The primary rejection reason is:

{chr(10).join(f'- {reason}' for reason in reasons)}

## Decision

Do not proceed to A-side training-policy implementation from this manifest version.

Recommended next route decision: `manifest-needs-redesign`.
"""


def main() -> int:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    row_count = _read_manifest_count(args.manifest)
    if row_count != metrics.get("row_count"):
        raise SystemExit(f"manifest row count mismatch: manifest={row_count} metrics={metrics.get('row_count')}")
    report = "# Phase27 A Evidence-State Manifest Report\n\n" + _common_body(
        metrics, args.manifest, args.project_root
    )
    record = "# exp-20260507 Phase27 A Evidence-State Manifest\n\n" + _common_body(
        metrics, args.manifest, args.project_root
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    args.record_out.write_text(record, encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "record": str(args.record_out), "verdict": metrics.get("verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
