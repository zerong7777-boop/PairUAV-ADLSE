#!/usr/bin/env python3
"""Write Phase27 A feature-calibration-v2 report and experiment record."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEC_PATH = "docs/superpowers/specs/2026-05-07-uavm-a-evidence-state-feature-calibration-redesign-v2-spec-zh.md"
PLAN_PATH = "docs/superpowers/plans/2026-05-07-uavm-a-evidence-state-feature-calibration-redesign-v2-implementation.md"
EXPERIMENT_ID = "exp-20260507-phase27-a-evidence-state-feature-calibration-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--axes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--record-out", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def count_lines(counts: dict[str, Any], fractions: dict[str, Any]) -> str:
    return "\n".join(
        f"- `{key}`: `{counts[key]}` (`{fractions.get(key, 0.0):.6f}`)"
        for key in sorted(counts)
    )


def report_text(metrics: dict[str, Any], args: argparse.Namespace, commit: str) -> str:
    overlap = metrics["canonical_v2_overlap_comparison"]
    raw = metrics["raw_v3_comparison"]
    v1 = metrics["v1_vs_v2_comparison"]
    reasons = "\n".join(f"- {reason}" for reason in metrics.get("verdict_reasons", [])) or "- none"
    return f"""# Phase27 A Evidence-State Feature Calibration-v2 Report

Date: `{datetime.now(timezone.utc).isoformat()}`

Experiment: `{EXPERIMENT_ID}`

Verdict: `{metrics['verdict']}`

## Scope

This run validates calibration-v2 on the existing 60k v3 feature surface. It uses adequacy gates, explicit risk axes, control centrality, and canonical pair ids. It does not train, infer, change checkpoints, or create a submission.

## Inputs

- Spec: `{SPEC_PATH}`
- Plan: `{PLAN_PATH}`
- Axes: `{args.axes}`
- Manifest: `{args.manifest}`
- Metrics: `{args.metrics}`
- Code commit: `{commit}`

## Main Metrics

- Row count: `{metrics['row_count']}`
- Leakage audit passed: `{metrics['leakage_audit']['passed']}`
- Exactly-one-base-regime passed: `{metrics['exactly_one_base_regime_passed']}`
- `ordinary_control_anchor`: `{metrics['ordinary_control_anchor_fraction']:.6f}`
- `high_evidence_anchor`: `{metrics['high_evidence_anchor_fraction']:.6f}`
- Max base-regime fraction: `{metrics['max_base_regime_fraction']:.6f}`
- Quota-only suspected: `{metrics['quota_only_suspected']}`
- Training started: `{metrics['training_started']}`
- Submission created: `{metrics['submission_created']}`

## Base Regimes

{count_lines(metrics['base_regime_counts'], metrics['base_regime_fractions'])}

## v1-vs-v2

- v1 ordinary/control fraction: `{v1['v1_ordinary_control_anchor_fraction']:.6f}`
- v2 ordinary/control fraction: `{v1['v2_ordinary_control_anchor_fraction']:.6f}`
- v1 high-evidence fraction: `{v1['v1_high_evidence_anchor_fraction']:.6f}`
- v2 high-evidence fraction: `{v1['v2_high_evidence_anchor_fraction']:.6f}`
- v1 max base fraction: `{v1['v1_max_base_regime_fraction']:.6f}`
- v2 max base fraction: `{v1['v2_max_base_regime_fraction']:.6f}`

## Raw v3 Collapse Check

- Raw max base-regime fraction: `{raw['raw_max_base_regime_fraction']:.6f}`
- v2 max base-regime fraction: `{raw['v2_max_base_regime_fraction']:.6f}`
- Collapse remains fixed: `{raw['collapse_remains_fixed']}`

## Canonical v2-Overlap

- Status: `{overlap['status']}`
- Reference key count: `{overlap['reference_key_count']}`
- v2 key count: `{overlap['v2_key_count']}`
- Overlap count: `{overlap['overlap_count']}`
- Regressed below reference: `{overlap['regressed_below_reference']}`

## Verdict Reasons

{reasons}

## Boundary

Calibration-v2 satisfies the distribution-level goals that v1 failed: ordinary/control is above `15%`, high-evidence remains below `70%`, raw-v3 collapse remains fixed, and quota-only recovery is not suspected. However, the current 60k surface has zero canonical overlap with the v2 33-row matcher-derived reference. Therefore the required control-preservation gate cannot be evaluated, and A training-policy planning is still blocked.

## Next Action

Enter knowledge-review. The likely decision is `calibration-v2-needs-redesign` or a narrower validation-surface redesign, not A training-policy planning.
"""


def record_text(metrics: dict[str, Any], args: argparse.Namespace, commit: str) -> str:
    reasons = "; ".join(metrics.get("verdict_reasons", [])) or "none"
    return f"""+++
object_kind = "experiment"
experiment_id = "{EXPERIMENT_ID}"
status = "complete"
code_commit = "{commit}"
code_snapshot_id = "remote-working-tree-phase27-feature-calibration-v2"
config_fingerprint = "phase27_a_feature_calibration_v2|fit_split=train|rows={metrics['row_count']}"
machine_id = "lab-tailscale"
remote_run_path = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest"
primary_artifact = "{args.metrics}"
primary_artifact_reason = "Bounded calibration-v2 metrics contain adequacy, centrality, raw-v3, v1-vs-v2, and canonical-overlap gates."
result_summary = "{metrics['verdict']}: {reasons}"
metric_keys = ["row_count", "ordinary_control_anchor_fraction", "high_evidence_anchor_fraction", "max_base_regime_fraction", "quota_only_suspected", "canonical_v2_overlap_comparison", "verdict"]
created_at = "{datetime.now(timezone.utc).isoformat()}"
updated_at = "{datetime.now(timezone.utc).isoformat()}"
tags = ["uavm", "phase27", "evidence-state", "feature-calibration-v2", "bounded-eval"]
linked_task_ids = ["20260417-uavm-pairuav-codabench"]
source_refs = ["{SPEC_PATH}", "{PLAN_PATH}", "{args.metrics}", "{args.manifest}", "{args.axes}"]
+++
# Phase27 A Evidence-State Feature Calibration-v2 Experiment

## Purpose

Test whether adequacy gates plus control centrality recover ordinary/control anchors without returning to raw v3 high-evidence collapse.

## Method

The run consumed the existing 60k v3 feature surface, built calibration-v2 axes and manifest, then validated leakage, exactly-one-base-regime, v1-vs-v2 movement, raw-v3 collapse, quota-only suspicion, and canonical v2-overlap.

## Result

- Verdict: `{metrics['verdict']}`
- Row count: `{metrics['row_count']}`
- `ordinary_control_anchor`: `{metrics['ordinary_control_anchor_fraction']:.6f}`
- `high_evidence_anchor`: `{metrics['high_evidence_anchor_fraction']:.6f}`
- Max base-regime fraction: `{metrics['max_base_regime_fraction']:.6f}`
- Quota-only suspected: `{metrics['quota_only_suspected']}`
- Canonical overlap status: `{metrics['canonical_v2_overlap_comparison']['status']}`
- Canonical overlap count: `{metrics['canonical_v2_overlap_comparison']['overlap_count']}`
- Training started: `{metrics['training_started']}`
- Submission created: `{metrics['submission_created']}`

## Boundary of Conclusion

Calibration-v2 is promising as a distribution-level manifest redesign, but it cannot be promoted because the required v2-overlap control-preservation gate is unevaluable on the current 60k surface.

## Next Action

Proceed to knowledge-review. Do not start A training-policy implementation from this result.
"""


def main() -> int:
    args = parse_args()
    metrics = read_json(args.metrics)
    commit = git_commit()
    report = Path(args.report_out)
    record = Path(args.record_out)
    ensure_parent(report)
    ensure_parent(record)
    report.write_text(report_text(metrics, args, commit), encoding="utf-8")
    record.write_text(record_text(metrics, args, commit), encoding="utf-8")
    print(f"wrote report: {report}")
    print(f"wrote record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
