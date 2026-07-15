#!/usr/bin/env python3
"""Write Phase27 calibration-v3 overlap validation record and docs/ai summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {"value": payload}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_record(args: argparse.Namespace, combined: dict[str, Any], identity: dict[str, Any], control: dict[str, Any], full: dict[str, Any]) -> str:
    verdict = combined.get("verdict", "")
    return f"""+++
object_kind = "experiment"
experiment_id = "exp-20260507-phase27-a-evidence-state-calibration-v3-overlap-validation"
task_id = "20260417-uavm-pairuav-codabench"
phase_id = "phase-27-a-evidence-state-feature-calibration-redesign-v3"
artifact_kind = "bounded-eval"
created_at = "2026-05-07T00:00:00+08:00"
updated_at = "2026-05-07T00:00:00+08:00"
result_summary = "{verdict}: matched overlap rows = {control.get('matched_row_count')}"
metric_keys = ["verdict", "failure_classification", "matched_row_count", "preservation_rate", "full_surface_regression_passed"]
tags = ["uavm", "phase27", "evidence-state", "calibration-v3", "overlap-validation", "bounded-eval"]
+++
# Phase27 A Evidence-State Calibration-v3 Overlap Validation Experiment

## Scope

This bounded eval tested whether calibration-v2's distribution-level success can be validated on an overlap-capable reference surface. It did not train, finetune, run inference, change checkpoints, or create a submission.

## Inputs

- V3 calibrated manifest: `{args.v3_manifest}`
- V3 calibrated axes: `{args.v3_axes}`
- V2 reference manifest: `{args.v2_reference}`
- V2 calibration metrics: `{args.v2_calibration_metrics}`

## Outputs

- Identity audit: `{args.identity_audit}`
- Matched surface: `{args.matched_surface}`
- Control metrics: `{args.control_metrics}`
- Full regression metrics: `{args.full_regression}`
- Combined metrics: `{args.combined_metrics}`
- Report: `{args.report}`

## Key Results

- Verdict: `{verdict}`
- Failure classification: `{identity.get('failure_classification')}`
- Reference rows: `{identity.get('reference', {}).get('row_count')}`
- V3 rows: `{identity.get('v3', {}).get('row_count')}`
- Matched rows: `{control.get('matched_row_count')}`
- Reference ordinary/control rows: `{control.get('reference_ordinary_control_count')}`
- Matched reference ordinary/control rows: `{control.get('matched_reference_ordinary_control_count')}`
- Preservation rate: `{control.get('preservation_rate')}`
- Promotion eligible overlap: `{control.get('promotion_eligible_overlap')}`
- Full-surface regression passed: `{full.get('gates', {}).get('full_surface_regression_passed')}`

## Overlap Counts

- Raw pair-id overlap: `{identity.get('overlap', {}).get('raw_pair_id_overlap_count')}`
- Canonical same-order overlap: `{identity.get('overlap', {}).get('canonical_same_order_overlap_count')}`
- Canonical flipped overlap: `{identity.get('overlap', {}).get('canonical_flipped_overlap_count')}`
- Canonical orderless overlap: `{identity.get('overlap', {}).get('canonical_orderless_overlap_count')}`
- Group-only overlap: `{identity.get('overlap', {}).get('group_only_overlap_count')}`
- Image-name-only overlap: `{identity.get('overlap', {}).get('image_name_only_overlap_count')}`

## Interpretation

Calibration-v3 did not falsify calibration-v2 distribution health. The full 60k surface still passes the distribution and leakage gates. The blocker is the reference surface: the 33-row matcher-derived reference has zero raw, canonical, flipped, orderless, and group overlap with the 60k v3 calibrated surface. The 31 image-name-only overlaps are diagnostic only and cannot support row-level control-preservation claims.

## Verdict

`{verdict}`

The A evidence-state route must not proceed to A training-policy implementation from this result. The next step is to acquire or build a larger read-only reference surface that shares the same pair identity namespace as the 60k v3 surface, or explicitly park the A route if that reference surface is not worth building.
"""


def make_docs_ai(record: str) -> str:
    return "# Phase27 A Evidence-State Calibration-v3 Overlap Validation\n\n" + "\n".join(record.splitlines()[18:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-metrics", required=True)
    parser.add_argument("--identity-audit", required=True)
    parser.add_argument("--control-metrics", required=True)
    parser.add_argument("--full-regression", required=True)
    parser.add_argument("--matched-surface", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--v3-manifest", required=True)
    parser.add_argument("--v3-axes", required=True)
    parser.add_argument("--v2-reference", required=True)
    parser.add_argument("--v2-calibration-metrics", required=True)
    parser.add_argument("--record-out", required=True)
    parser.add_argument("--docs-ai-out", required=True)
    args = parser.parse_args()

    combined = read_json(Path(args.combined_metrics))
    identity = read_json(Path(args.identity_audit))
    control = read_json(Path(args.control_metrics))
    full = read_json(Path(args.full_regression))
    record = make_record(args, combined, identity, control, full)
    write_text(Path(args.record_out), record)
    write_text(Path(args.docs_ai_out), make_docs_ai(record))
    print(f"wrote_record={args.record_out}")
    print(f"wrote_docs_ai={args.docs_ai_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
