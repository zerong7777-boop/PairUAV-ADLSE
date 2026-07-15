#!/usr/bin/env python3
"""Build calibrated Phase27 A v3 evidence-state axes and manifest CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_v3_calibration import (
    AXIS_NAMES,
    BASE_REGIMES,
    CALIBRATION_VERSION,
    FEATURE_COLUMNS,
    assign_calibrated_evidence_state,
    audit_calibration_columns,
    compute_calibrated_axes,
    fit_calibration,
    summarize_calibration,
)


IDENTITY_COLUMNS = [
    "source_split",
    "json_path",
    "group_id",
    "pair_id",
    "pair_key",
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
    parser.add_argument("--raw-v3-manifest", default=None)
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
    return "" if value is None else value


def build_fieldnames(base: list[str], rows: list[dict[str, Any]]) -> list[str]:
    present = set().union(*(row.keys() for row in rows)) if rows else set()
    return [column for column in base if column in present] + [column for column in base if column not in present]


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


def main() -> int:
    args = parse_args()
    feature_manifest = Path(args.feature_manifest)
    apply_splits = None
    if args.apply_splits:
        apply_splits = {item.strip() for item in args.apply_splits.split(",") if item.strip()}

    rows, input_columns = load_rows(feature_manifest, args.limit, apply_splits)
    audit = audit_calibration_columns(input_columns)
    if not audit["passed"]:
        print(f"Forbidden calibration columns: {audit['forbidden_columns']}", file=sys.stderr)
        return 2

    fit_rows = [row for row in rows if str(row.get("source_split", "")).strip() == args.fit_split]
    calibration_rows = fit_rows if fit_rows else rows
    calibration = fit_calibration(calibration_rows)
    calibration["fit_scope"] = f"source_split_{args.fit_split}" if fit_rows else calibration.get("fit_scope", "all_rows_no_split")
    calibration_source = str(feature_manifest)

    axes_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for row in rows:
        axes = compute_calibrated_axes(row, calibration)
        axis_record: dict[str, Any] = {column: row.get(column, "") for column in IDENTITY_COLUMNS}
        for column in FEATURE_COLUMNS:
            axis_record[column] = row.get(column, "")
        for axis in AXIS_NAMES:
            axis_record[axis] = axes.get(axis)
            axis_record[f"{axis}_band"] = axes.get(f"{axis}_band", "missing")
        axis_record["calibration_version"] = CALIBRATION_VERSION
        axis_record["calibration_fit_scope"] = calibration.get("fit_scope", "")
        axis_record["calibration_source"] = calibration_source
        axes_rows.append(axis_record)

        assignment = assign_calibrated_evidence_state(row, axes, calibration)
        assignment["leakage_audit_passed"] = True
        manifest_record = {column: row.get(column, "") for column in IDENTITY_COLUMNS}
        manifest_record.update(assignment)
        manifest_rows.append(manifest_record)
        assignments.append(assignment)

    axes_fieldnames = (
        build_fieldnames(IDENTITY_COLUMNS, rows)
        + [column for column in FEATURE_COLUMNS if column not in IDENTITY_COLUMNS]
        + AXIS_NAMES
        + [f"{axis}_band" for axis in AXIS_NAMES]
        + ["calibration_version", "calibration_fit_scope", "calibration_source"]
    )
    manifest_fieldnames = (
        build_fieldnames(IDENTITY_COLUMNS, rows)
        + ["base_regime"]
        + [f"base_{regime}" for regime in BASE_REGIMES]
        + ["risk_tags"]
        + AXIS_NAMES
        + [f"{axis}_band" for axis in AXIS_NAMES]
        + ["calibration_version", "leakage_audit_passed"]
    )
    write_csv(Path(args.out_axes), axes_fieldnames, axes_rows)
    write_csv(Path(args.out_manifest), manifest_fieldnames, manifest_rows)

    summary = summarize_calibration(rows, assignments)
    base_counts = summary["base_regime_counts"]
    row_count = len(rows)
    exactly_one_passed = all(exactly_one_base_regime(row) for row in manifest_rows)
    metrics = {
        "row_count": row_count,
        "leakage_audit": audit,
        "leakage_audit_passed": True,
        "calibration": {
            "version": CALIBRATION_VERSION,
            "fit_scope": calibration.get("fit_scope", ""),
            "fit_split": args.fit_split,
            "fit_row_count": calibration.get("fit_row_count", 0),
            "source": calibration_source,
            "raw_v3_manifest": args.raw_v3_manifest,
            "v2_manifest": args.v2_manifest,
            "axis_thresholds": calibration.get("axis_thresholds", {}),
            "feature_columns_used": calibration.get("feature_columns_used", []),
        },
        "base_regime_counts": base_counts,
        "base_regime_fractions": {
            key: (value / row_count if row_count else 0.0) for key, value in base_counts.items()
        },
        "axis_band_counts": summary["axis_band_counts"],
        "exactly_one_base_regime_passed": exactly_one_passed,
        "training_started": False,
        "submission_created": False,
    }
    metrics["base_regime_total"] = sum(Counter(row["base_regime"] for row in manifest_rows).values())
    ensure_parent(Path(args.metrics_out))
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not exactly_one_passed:
        print("Exactly-one-base-regime invariant failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
