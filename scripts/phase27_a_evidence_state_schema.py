#!/usr/bin/env python3
"""Shared schema and deterministic state rules for Phase27 A manifest."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "phase27_a_manifest_v1"

REQUIRED_ID_COLUMNS = ["pair_id", "query_name", "reference_name"]

STATE_COLUMNS = [
    "ordinary_control",
    "high_evidence_anchor",
    "low_observable",
    "heading_risk",
    "range_risk",
    "ambiguous_scale",
    "semantic_geometry_conflict_candidate",
    "target_regime_shift_candidate",
]

ALLOWED_CONSTRUCTION_PATTERNS = [
    "match",
    "valid",
    "confidence",
    "entropy",
    "occupied",
    "spread",
    "scale",
    "anchor",
    "target",
    "group",
]

FORBIDDEN_CONSTRUCTION_PATTERNS = [
    "angle_err",
    "range_err",
    "combined_error",
    "residual",
    "official",
    "leaderboard",
    "phase11",
    "phase13",
    "phase14",
    "slice_label",
]

DEFAULT_THRESHOLDS = {
    "low_match_count": 10.0,
    "low_valid_ratio": 0.20,
    "low_occupied_cells": 4.0,
    "high_match_count": 50.0,
    "high_valid_ratio": 0.50,
    "high_confidence_sum": 5.0,
    "high_occupied_cells": 8.0,
    "low_anchor_spread": 0.15,
    "high_anchor_spread": 0.60,
    "low_spatial_entropy": 0.25,
    "low_scale_balance": 0.35,
    "high_scale_balance": 0.75,
    "semantic_geometry_gap": 0.35,
    "target_shift_proxy": 0.70,
}


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = _lower(column)
    return any(pattern in lowered for pattern in patterns)


def audit_construction_columns(columns: Iterable[str]) -> dict[str, Any]:
    """Classify columns for non-leaky construction usage."""

    unique_columns = list(dict.fromkeys(str(column) for column in columns))
    forbidden_columns = [
        column
        for column in unique_columns
        if _matches_any(column, FORBIDDEN_CONSTRUCTION_PATTERNS)
    ]
    allowed_columns = [
        column
        for column in unique_columns
        if column not in forbidden_columns and _matches_any(column, ALLOWED_CONSTRUCTION_PATTERNS)
    ]
    unknown_columns = [
        column
        for column in unique_columns
        if column not in forbidden_columns and column not in allowed_columns
    ]
    return {
        "passed": not forbidden_columns,
        "forbidden_columns": forbidden_columns,
        "allowed_columns": allowed_columns,
        "unknown_columns": unknown_columns,
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _get(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row:
            value = _to_float(row[name])
            if value is not None:
                return value
    return None


def _get_bool(row: dict[str, Any], *names: str) -> bool:
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n", ""}:
            return False
    return False


def assign_evidence_states(
    row: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, int]:
    """Assign binary evidence states from observable construction fields."""

    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update({key: float(value) for key, value in thresholds.items()})

    match_count = _get(row, "match_count", "raw_global_stats.log1p_match_count")
    if "raw_global_stats.log1p_match_count" in row and "match_count" not in row:
        log_match_count = _get(row, "raw_global_stats.log1p_match_count")
        match_count = math.expm1(log_match_count) if log_match_count is not None else None
    valid_ratio = _get(row, "valid_ratio")
    confidence_sum = _get(row, "confidence_sum")
    occupied_cells = _get(row, "occupied_cells")
    spatial_entropy = _get(row, "spatial_entropy", "raw_global_stats.spatial_entropy")
    anchor_spread = _get(row, "anchor_spread")
    scale_balance = _get(row, "scale_balance")
    semantic_proxy = _get(row, "semantic_proxy")
    geometry_proxy = _get(row, "geometry_proxy")
    target_shift_proxy = _get(row, "target_shift_proxy")
    fallback_used = _get_bool(row, "fallback_used", "raw_global_stats.fallback_used")

    observability_values = [match_count, occupied_cells]
    missing_observability = any(value is None for value in observability_values)

    low_observable = fallback_used or missing_observability
    if match_count is not None and match_count < t["low_match_count"]:
        low_observable = True
    if valid_ratio is not None and valid_ratio < t["low_valid_ratio"]:
        low_observable = True
    if occupied_cells is not None and occupied_cells < t["low_occupied_cells"]:
        low_observable = True

    high_evidence_anchor = False
    if not missing_observability and confidence_sum is not None:
        high_evidence_anchor = (
            match_count >= t["high_match_count"]
            and occupied_cells >= t["high_occupied_cells"]
            and confidence_sum >= t["high_confidence_sum"]
            and (valid_ratio is None or valid_ratio >= t["high_valid_ratio"])
        )

    heading_risk = False
    if anchor_spread is not None and anchor_spread < t["low_anchor_spread"]:
        heading_risk = True
    if spatial_entropy is not None and spatial_entropy < t["low_spatial_entropy"]:
        heading_risk = True

    range_risk = False
    ambiguous_scale = False
    if scale_balance is not None:
        range_risk = scale_balance < t["low_scale_balance"]
        ambiguous_scale = t["low_scale_balance"] <= scale_balance <= t["high_scale_balance"]
    if anchor_spread is not None and anchor_spread > t["high_anchor_spread"] and scale_balance is None:
        ambiguous_scale = True

    semantic_geometry_conflict = False
    if semantic_proxy is not None and geometry_proxy is not None:
        semantic_geometry_conflict = abs(semantic_proxy - geometry_proxy) >= t["semantic_geometry_gap"]

    target_regime_shift = False
    if target_shift_proxy is not None:
        target_regime_shift = target_shift_proxy >= t["target_shift_proxy"]

    ordinary_control = not any(
        [
            low_observable,
            heading_risk,
            range_risk,
            ambiguous_scale,
            semantic_geometry_conflict,
            target_regime_shift,
        ]
    )

    states = {
        "ordinary_control": int(ordinary_control and not low_observable),
        "high_evidence_anchor": int(high_evidence_anchor),
        "low_observable": int(low_observable),
        "heading_risk": int(heading_risk),
        "range_risk": int(range_risk),
        "ambiguous_scale": int(ambiguous_scale),
        "semantic_geometry_conflict_candidate": int(semantic_geometry_conflict),
        "target_regime_shift_candidate": int(target_regime_shift),
    }
    return {column: int(bool(states[column])) for column in STATE_COLUMNS}


def write_manifest_schema(path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "required_id_columns": REQUIRED_ID_COLUMNS,
        "state_columns": STATE_COLUMNS,
        "allowed_construction_patterns": ALLOWED_CONSTRUCTION_PATTERNS,
        "forbidden_construction_patterns": FORBIDDEN_CONSTRUCTION_PATTERNS,
        "default_thresholds": DEFAULT_THRESHOLDS,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_manifest_schema(args.out)
