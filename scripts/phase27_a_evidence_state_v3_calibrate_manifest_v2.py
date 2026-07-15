#!/usr/bin/env python3
"""Build Phase27 A evidence-state v3 calibration-v2 axes and manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_v3_calibration_v2 import (
    ADEQUACY_FIELDS,
    AXIS_NAMES,
    BASE_REGIMES,
    CALIBRATION_V2_VERSION,
    FEATURE_COLUMNS,
    assign_evidence_state_v2,
    audit_calibration_v2_columns,
    canonical_pair_id,
    compute_adequacy,
    compute_calibrated_axes_v2,
    fit_calibration_v2,
    summarize_calibration_v2,
)


IDENTITY_COLUMNS = [
    "source_split",
    "json_path",
    "group_id",
    "pair_id",
    "pair_key",
    "canonical_pair_id",
    "image_a",
    "image_b",
    "image_a_name",
    "image_b_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-axes", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--apply-splits", default=None)
    parser.add_argument("--v1-manifest", default=None)
    parser.add_argument("--v2-manifest", default=None)
    return parser.parse_args()


def load_rows(path: Path, limit: int | None, apply_splits: set[str] | None) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            split = str(row.get("source_split", "")).strip()
            if apply_splits is not None and split not in apply_splits:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows, fieldnames


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8f}"
    if value is None:
        return ""
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def exactly_one_base_regime(row: dict[str, Any]) -> bool:
    total = 0
    for regime in BASE_REGIMES:
        try:
            total += int(row.get(f"base_{regime}", 0))
        except (TypeError, ValueError):
            return False
    return total == 1


def fractions(counts: dict[str, int], total: int) -> dict[str, float]:
    return {key: (value / total if total else 0.0) for key, value in counts.items()}


def count_low_observable_reasons(assignments: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for assignment in assignments:
        reasons = str(assignment.get("low_observable_reason", "") or "")
        if not reasons:
            continue
        for reason in reasons.split("|"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return dict(counter)


def count_adequacy(assignments: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(1 for assignment in assignments if str(assignment.get(field, "0")) in {"1", "1.0", "True", "true"})
        for field in ADEQUACY_FIELDS
        if field != "low_observable_reason"
    }


def main() -> int:
    args = parse_args()
    apply_splits = None
    if args.apply_splits:
        apply_splits = {item.strip() for item in args.apply_splits.split(",") if item.strip()}

    rows, input_columns = load_rows(Path(args.feature_manifest), args.limit, apply_splits)
    audit = audit_calibration_v2_columns(input_columns)
    if not audit["passed"]:
        print(f"Forbidden calibration-v2 columns: {audit['forbidden_columns']}", file=sys.stderr)
        return 2

    calibration = fit_calibration_v2(rows, fit_split=args.fit_split)
    calibration_source = str(Path(args.feature_manifest))

    axes_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    for row in rows:
        axes = compute_calibrated_axes_v2(row, calibration)
        adequacy = compute_adequacy(row, axes)
        assignment = assign_evidence_state_v2(row, axes, adequacy, calibration)
        assignment["leakage_audit_passed"] = True
        canonical_id = canonical_pair_id(row)

        axis_record: dict[str, Any] = {column: row.get(column, "") for column in IDENTITY_COLUMNS if column != "canonical_pair_id"}
        axis_record["canonical_pair_id"] = canonical_id
        for column in FEATURE_COLUMNS:
            axis_record[column] = row.get(column, "")
        for field in ADEQUACY_FIELDS:
            axis_record[field] = assignment.get(field, "")
        for axis in AXIS_NAMES:
            axis_record[axis] = assignment.get(axis, axes.get(axis))
        axis_record["calibration_version"] = CALIBRATION_V2_VERSION
        axis_record["calibration_fit_scope"] = calibration.get("fit_scope", "")
        axis_record["calibration_source"] = calibration_source
        axes_rows.append(axis_record)

        manifest_record = {column: row.get(column, "") for column in IDENTITY_COLUMNS if column != "canonical_pair_id"}
        manifest_record["canonical_pair_id"] = canonical_id
        manifest_record.update(assignment)
        manifest_rows.append(manifest_record)
        assignments.append(assignment)

    raw_feature_columns = [column for column in FEATURE_COLUMNS if column not in IDENTITY_COLUMNS]
    axes_fieldnames = (
        IDENTITY_COLUMNS
        + raw_feature_columns
        + ADEQUACY_FIELDS
        + AXIS_NAMES
        + ["calibration_version", "calibration_fit_scope", "calibration_source"]
    )
    manifest_fieldnames = (
        IDENTITY_COLUMNS
        + ADEQUACY_FIELDS
        + AXIS_NAMES
        + ["base_regime"]
        + [f"base_{regime}" for regime in BASE_REGIMES]
        + ["risk_tags", "calibration_version", "leakage_audit_passed"]
    )

    write_csv(Path(args.out_axes), axes_fieldnames, axes_rows)
    write_csv(Path(args.out_manifest), manifest_fieldnames, manifest_rows)

    summary = summarize_calibration_v2(rows, assignments)
    base_counts = summary["base_regime_counts"]
    row_count = len(rows)
    base_fracs = fractions(base_counts, row_count)
    canonical_nonempty = sum(1 for assignment in assignments if assignment.get("canonical_pair_id"))
    exactly_one = all(exactly_one_base_regime(row) for row in manifest_rows)
    metrics = {
        "schema_version": "phase27_a_feature_calibration_v2_build_v1",
        "row_count": row_count,
        "axes_path": args.out_axes,
        "manifest_path": args.out_manifest,
        "v1_manifest": args.v1_manifest,
        "v2_manifest": args.v2_manifest,
        "leakage_audit": audit,
        "leakage_audit_passed": True,
        "calibration": {
            "version": CALIBRATION_V2_VERSION,
            "fit_scope": calibration.get("fit_scope", ""),
            "fit_row_count": calibration.get("fit_row_count", 0),
            "source": calibration_source,
            "axis_medians": calibration.get("axis_medians", {}),
            "absolute_thresholds": calibration.get("absolute_thresholds", {}),
            "control_centrality_threshold": calibration.get("control_centrality_threshold"),
            "feature_columns_used": calibration.get("feature_columns_used", []),
            "target_columns_used": calibration.get("target_columns_used", []),
        },
        "adequacy_counts": count_adequacy(assignments),
        "low_observable_reason_counts": count_low_observable_reasons(assignments),
        "base_regime_counts": base_counts,
        "base_regime_fractions": base_fracs,
        "ordinary_control_anchor_fraction": base_fracs.get("ordinary_control_anchor", 0.0),
        "high_evidence_anchor_fraction": base_fracs.get("high_evidence_anchor", 0.0),
        "max_base_regime_fraction": max(base_fracs.values()) if base_fracs else 0.0,
        "canonical_pair_id_nonempty_fraction": canonical_nonempty / row_count if row_count else 0.0,
        "exactly_one_base_regime_passed": exactly_one,
        "training_started": False,
        "submission_created": False,
    }
    ensure_parent(Path(args.metrics_out))
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not exactly_one:
        print("Exactly-one-base-regime invariant failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
