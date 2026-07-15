#!/usr/bin/env python3
"""Write Phase27 A v3 feature-calibration report and experiment record."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEC_PATH = "docs/superpowers/specs/2026-05-07-uavm-a-evidence-state-feature-calibration-redesign-spec-zh.md"
PLAN_PATH = "docs/superpowers/plans/2026-05-07-uavm-a-evidence-state-feature-calibration-implementation.md"
EXPERIMENT_ID = "exp-20260507-phase27-a-evidence-state-feature-calibration-v1"


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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def format_counts(counts: dict[str, Any], fractions: dict[str, Any]) -> str:
    lines = []
    for key in sorted(counts):
        value = counts[key]
        frac = fractions.get(key, 0.0)
        lines.append(f"- `{key}`: `{value}` (`{frac:.6f}`)")
    return "\n".join(lines)


def report_text(metrics: dict[str, Any], args: argparse.Namespace, commit: str) -> str:
    raw = metrics["raw_v3_comparison"]
    v2 = metrics["v2_overlap_comparison"]
    reasons = metrics.get("verdict_reasons", [])
    reason_text = "\n".join(f"- {reason}" for reason in reasons) if reasons else "- none"
    return f"""# Phase27 A Evidence-State Feature Calibration Report

Date: `{datetime.now(timezone.utc).isoformat()}`

Experiment: `{EXPERIMENT_ID}`

Verdict: `{metrics['verdict']}`

## Scope

This run validates the Phase27 A evidence-state feature-calibration layer on the existing v3 60k coverage surface. It does not train, change model weights, run inference, create a checkpoint, or create a submission.

## Inputs

- Spec: `{SPEC_PATH}`
- Plan: `{PLAN_PATH}`
- Axes input/output: `{args.axes}`
- Calibrated manifest: `{args.manifest}`
- Metrics: `{args.metrics}`
- Code commit: `{commit}`

## Calibration Outputs

- Row count: `{metrics['row_count']}`
- Axes row count: `{metrics['axes_row_count']}`
- Leakage audit passed: `{metrics['leakage_audit']['passed']}`
- Exactly-one-base-regime passed: `{metrics['exactly_one_base_regime_passed']}`
- Training started: `{metrics['training_started']}`
- Submission created: `{metrics['submission_created']}`

## Base Regimes

{format_counts(metrics['base_regime_counts'], metrics['base_regime_fractions'])}

## Raw v3 Comparison

- Raw `high_evidence_anchor`: `{raw['raw_high_evidence_anchor_count']}` (`{raw['raw_high_evidence_anchor_fraction']:.6f}`)
- Calibrated `high_evidence_anchor`: `{raw['calibrated_high_evidence_anchor_count']}` (`{raw['calibrated_high_evidence_anchor_fraction']:.6f}`)
- Raw `ordinary_control_anchor`: `{raw['raw_ordinary_control_anchor_count']}` (`{raw['raw_ordinary_control_anchor_fraction']:.6f}`)
- Calibrated `ordinary_control_anchor`: `{raw['calibrated_ordinary_control_anchor_count']}` (`{raw['calibrated_ordinary_control_anchor_fraction']:.6f}`)
- Raw max base-regime fraction: `{raw['raw_max_base_regime_fraction']:.6f}`
- Calibrated max base-regime fraction: `{raw['calibrated_max_base_regime_fraction']:.6f}`
- Collapse improved: `{raw['collapse_improved']}`

## v2-Overlap Check

- Status: `{v2['status']}`
- Overlap count: `{v2['overlap_count']}`
- v2 ordinary/control count on overlap: `{v2['v2_ordinary_control_count']}`
- calibrated ordinary/control count on v2 ordinary rows: `{v2['calibrated_ordinary_on_v2_ordinary_count']}`
- regressed below v2: `{v2['regressed_below_v2']}`

## Train/Dev Shift

- Max train/dev base-regime fraction shift: `{metrics['max_train_dev_base_fraction_shift']:.6f}`

## Verdict Reasons

{reason_text}

## Boundary of Conclusion

The calibration layer fixed the raw v3 regime-collapse symptom by reducing max base-regime fraction from about `0.9520` to `{metrics['max_base_regime_fraction']:.6f}` and reducing `high_evidence_anchor` from `57,122/60,000` to `{raw['calibrated_high_evidence_anchor_count']}/60,000`. However, it did not meet the control-anchor requirement: `ordinary_control_anchor` is `{metrics['ordinary_control_anchor_fraction']:.6f}`, below the required `0.15`, and v2-overlap is not reliable. This blocks A training-policy planning.

## Next Action

Enter knowledge-review with verdict candidate `calibration-needs-redesign` or `A-route-rejected-for-now`. Do not proceed to A training-policy implementation from this result.
"""


def record_text(metrics: dict[str, Any], args: argparse.Namespace, commit: str) -> str:
    reasons = "; ".join(metrics.get("verdict_reasons", [])) or "none"
    return f"""+++
object_kind = "experiment"
experiment_id = "{EXPERIMENT_ID}"
status = "complete"
code_commit = "{commit}"
code_snapshot_id = "remote-working-tree-phase27-feature-calibration-v1"
config_fingerprint = "phase27_a_feature_calibration_v1|fit_split=train|rows={metrics['row_count']}"
machine_id = "lab-tailscale"
remote_run_path = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest"
primary_artifact = "{args.metrics}"
primary_artifact_reason = "Bounded calibration metrics contain the verdict, collapse comparison, v2-overlap check, and readiness gates."
result_summary = "{metrics['verdict']}: {reasons}"
metric_keys = ["row_count", "ordinary_control_anchor_fraction", "high_evidence_anchor_fraction", "max_base_regime_fraction", "v2_overlap_comparison", "verdict"]
created_at = "{datetime.now(timezone.utc).isoformat()}"
updated_at = "{datetime.now(timezone.utc).isoformat()}"
tags = ["uavm", "phase27", "evidence-state", "feature-calibration", "bounded-eval"]
linked_task_ids = ["20260417-uavm-pairuav-codabench"]
source_refs = ["{SPEC_PATH}", "{PLAN_PATH}", "{args.metrics}", "{args.manifest}", "{args.axes}"]
+++
# Phase27 A Evidence-State Feature Calibration Experiment

## Purpose

Validate whether a non-leaky distribution-aware calibration layer can keep v3's 60k coverage while avoiding raw cheap-proxy regime collapse and preserving ordinary/control anchor behavior.

## Why This Mattered

Phase27 v3 solved coverage but collapsed into `high_evidence_anchor` (`57,122/60,000`) and only produced `21/60,000` ordinary/control anchors. The current experiment tests whether calibration can make the A evidence-state manifest usable before any training-policy design.

## Method

The run consumed the existing v3 bounded feature manifest, fit feature-axis calibration on train rows, wrote calibrated axes and a calibrated manifest, then validated leakage, base-regime balance, raw-v3 comparison, train/dev shift, and v2-overlap behavior.

## Run Conditions

- Remote repo: `/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav`
- Project root: `{args.project_root}`
- Metrics: `{args.metrics}`
- Axes: `{args.axes}`
- Manifest: `{args.manifest}`
- Training started: `{metrics['training_started']}`
- Submission created: `{metrics['submission_created']}`

## Result

- Verdict: `{metrics['verdict']}`
- Row count: `{metrics['row_count']}`
- `ordinary_control_anchor`: `{metrics['ordinary_control_anchor_fraction']:.6f}`
- `high_evidence_anchor`: `{metrics['high_evidence_anchor_fraction']:.6f}`
- Max base-regime fraction: `{metrics['max_base_regime_fraction']:.6f}`
- Raw collapse improved: `{metrics['raw_v3_comparison']['collapse_improved']}`
- v2-overlap status: `{metrics['v2_overlap_comparison']['status']}`

## Boundary of Conclusion

Calibration improved raw-v3 collapse but failed the control-anchor success criteria. This result does not authorize A training-policy planning.

## Next Action

Proceed to knowledge-review. The likely decision is `calibration-needs-redesign` unless the team chooses to reject the A route for now.
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
