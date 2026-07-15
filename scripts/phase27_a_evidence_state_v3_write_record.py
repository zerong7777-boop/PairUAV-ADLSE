#!/usr/bin/env python3
"""Write Phase27 v3 coverage report and experiment record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--record-out", required=True, type=Path)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(title: str, data: dict[str, Any]) -> str:
    lines = [f"## {title}", "", "| Key | Value |", "|---|---:|"]
    for key, value in data.items():
        lines.append(f"| `{key}` | `{_fmt(value)}` |")
    return "\n".join(lines)


def _body(metrics: dict[str, Any]) -> str:
    reasons = metrics.get("verdict_reasons", [])
    feature = metrics.get("feature_coverage", {})
    return f"""Date: `{datetime.now(timezone.utc).isoformat()}`

Task: `20260417-uavm-pairuav-codabench`

Phase: `phase-27-a-evidence-state-manifest-v3-coverage`

Claim level: `{metrics.get("claim_level")}`

Verdict: `{metrics.get("verdict")}`

No training, model change, checkpoint update, loss change, sample weighting, inference change, or submission package was performed.

## Key Results

- Total bounded rows: `{feature.get("total_rows")}`
- Train rows: `{feature.get("source_split_counts", {}).get("train")}`
- Dev rows: `{feature.get("source_split_counts", {}).get("dev")}`
- Identity feature rows: `{feature.get("identity_rows")}`
- Cheap image feature rows: `{feature.get("cheap_image_rows")}`
- Cached matcher rows: `{feature.get("cached_matcher_rows")}`
- Missing image rows: `{feature.get("missing_image_rows")}`
- Ordinary-control anchor fraction: `{_fmt(metrics.get("ordinary_control_anchor_fraction"))}`
- Max base-regime fraction: `{_fmt(metrics.get("max_base_regime_fraction"))}`

{_table("Base-Regime Counts", metrics.get("base_regime_counts", {}))}

{_table("Risk-Tag Counts", metrics.get("risk_tag_counts", {}))}

## Coverage Interpretation

v3 solved the v2 coverage blocker mechanically:

- train coverage reached `50,000` rows;
- dev coverage reached `10,000` rows;
- cheap image features succeeded on all `60,000` rows;
- image path failure count was `0`.

However, v3 failed the state-stability gate. The cheap-proxy mapping collapsed most rows into `high_evidence_anchor`, leaving only `21/60,000` rows as `ordinary_control_anchor`.

## Verdict Reasons

{chr(10).join(f'- {reason}' for reason in reasons)}

## Decision

Do not start A-side training-policy implementation.

The next redesign should not focus on coverage plumbing. Coverage plumbing works. The next redesign must recalibrate regime assignment for cheap proxy features so that large-coverage states remain non-collapsed and preserve ordinary/control anchors.
"""


def main() -> int:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = "# Phase27 A Evidence-State Manifest v3 Coverage Report\n\n" + _body(metrics)
    record = "# exp-20260507 Phase27 A Evidence-State Manifest v3 Coverage\n\n" + _body(metrics)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    args.record_out.write_text(record, encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "record": str(args.record_out), "verdict": metrics.get("verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
