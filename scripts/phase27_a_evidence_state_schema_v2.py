#!/usr/bin/env python3
"""Phase27 A evidence-state manifest v2 schema.

v2 separates mutually exclusive base regimes from orthogonal risk tags. This
keeps ordinary/control anchoring explicit instead of treating it as the absence
of every risk flag.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "phase27_a_manifest_v2"

BASE_REGIMES = [
    "ordinary_control_anchor",
    "high_evidence_anchor",
    "hard_trainable",
    "low_observable",
    "ambiguous_unreliable",
    "unknown_insufficient_features",
]

RISK_TAGS = [
    "heading_risk",
    "range_risk",
    "ambiguous_scale_tag",
    "semantic_geometry_conflict_tag",
    "target_regime_shift_tag",
    "matcher_fallback_tag",
    "weak_spatial_support_tag",
]

BASE_FLAG_COLUMNS = [f"base_{name}" for name in BASE_REGIMES]

FORBIDDEN_CONSTRUCTION_PATTERNS = [
    "angle_err",
    "range_err",
    "combined_error",
    "final_score",
    "residual",
    "gt_angle",
    "gt_distance",
    "official",
    "leaderboard",
    "phase11",
    "phase13",
    "phase14",
    "slice_label",
]

ALLOWED_CONSTRUCTION_PATTERNS = [
    "pair",
    "query",
    "reference",
    "group",
    "match",
    "valid",
    "confidence",
    "entropy",
    "occupied",
    "anchor",
    "spread",
    "scale",
    "fallback",
    "semantic_proxy",
    "geometry_proxy",
    "target_shift_proxy",
]

DEFAULT_THRESHOLDS = {
    "low_match_count": 10.0,
    "high_match_count": 50.0,
    "low_valid_ratio": 0.20,
    "low_occupied_cells": 4.0,
    "high_occupied_cells": 10.0,
    "low_confidence_sum": 2.0,
    "high_confidence_sum": 25.0,
    "low_mean_confidence": 0.20,
    "high_mean_confidence": 0.65,
    "low_anchor_spread": 0.03,
    "high_anchor_spread": 0.30,
    "low_scale_balance": 0.30,
    "high_scale_balance": 0.80,
    "low_spatial_entropy": 0.25,
    "semantic_geometry_gap": 0.35,
    "target_shift_proxy": 0.70,
}


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = _lower(column)
    return any(pattern in lowered for pattern in patterns)


def audit_construction_columns_v2(columns: Iterable[str]) -> dict[str, Any]:
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
        if text in {"1", "1.0", "true", "yes", "y"}:
            return True
        if text in {"0", "0.0", "false", "no", "n", ""}:
            return False
    return False


def _between(value: float | None, low: float, high: float) -> bool:
    return value is not None and low <= value <= high


def _risk_tags(row: dict[str, Any], t: dict[str, float]) -> dict[str, int]:
    match_count = _get(row, "match_count")
    valid_ratio = _get(row, "valid_ratio")
    occupied_cells = _get(row, "occupied_cells")
    spatial_entropy = _get(row, "spatial_entropy")
    anchor_spread = _get(row, "anchor_spread")
    scale_balance = _get(row, "scale_balance")
    semantic_proxy = _get(row, "semantic_proxy")
    geometry_proxy = _get(row, "geometry_proxy")
    target_shift_proxy = _get(row, "target_shift_proxy")
    fallback_used = _get_bool(row, "fallback_used")

    weak_spatial = False
    if occupied_cells is not None and occupied_cells < t["low_occupied_cells"]:
        weak_spatial = True
    if spatial_entropy is not None and spatial_entropy < t["low_spatial_entropy"]:
        weak_spatial = True

    heading_risk = False
    if anchor_spread is not None and anchor_spread < t["low_anchor_spread"]:
        heading_risk = True
    if weak_spatial:
        heading_risk = True

    range_risk = scale_balance is not None and scale_balance < t["low_scale_balance"]
    ambiguous_scale = scale_balance is not None and t["low_scale_balance"] <= scale_balance <= t["high_scale_balance"]

    semantic_conflict = False
    if semantic_proxy is not None and geometry_proxy is not None:
        semantic_conflict = abs(semantic_proxy - geometry_proxy) >= t["semantic_geometry_gap"]

    target_shift = target_shift_proxy is not None and target_shift_proxy >= t["target_shift_proxy"]

    # Very low match/valid evidence is an observability risk, but not the only
    # route to base low_observable.
    if match_count is not None and match_count < t["low_match_count"]:
        weak_spatial = True
    if valid_ratio is not None and valid_ratio < t["low_valid_ratio"]:
        weak_spatial = True

    return {
        "heading_risk": int(heading_risk),
        "range_risk": int(range_risk),
        "ambiguous_scale_tag": int(ambiguous_scale),
        "semantic_geometry_conflict_tag": int(semantic_conflict),
        "target_regime_shift_tag": int(target_shift),
        "matcher_fallback_tag": int(fallback_used),
        "weak_spatial_support_tag": int(weak_spatial),
    }


def assign_evidence_state_v2(
    row: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update({key: float(value) for key, value in thresholds.items()})

    match_count = _get(row, "match_count")
    valid_ratio = _get(row, "valid_ratio")
    confidence_sum = _get(row, "confidence_sum")
    mean_confidence = _get(row, "mean_confidence")
    occupied_cells = _get(row, "occupied_cells")
    anchor_spread = _get(row, "anchor_spread")
    scale_balance = _get(row, "scale_balance")
    fallback_used = _get_bool(row, "fallback_used")

    tags = _risk_tags(row, t)
    required_missing = match_count is None or occupied_cells is None or confidence_sum is None

    if required_missing:
        base_regime = "unknown_insufficient_features"
    elif (
        fallback_used
        or match_count < t["low_match_count"]
        or occupied_cells < t["low_occupied_cells"]
        or (valid_ratio is not None and valid_ratio < t["low_valid_ratio"])
    ):
        base_regime = "low_observable"
    elif (
        match_count >= t["high_match_count"]
        and occupied_cells >= t["high_occupied_cells"]
        and confidence_sum >= t["high_confidence_sum"]
        and (valid_ratio is None or valid_ratio >= t["low_valid_ratio"])
    ):
        base_regime = "high_evidence_anchor"
    elif (
        not fallback_used
        and _between(match_count, t["low_match_count"], t["high_match_count"])
        and occupied_cells >= t["low_occupied_cells"]
        and _between(confidence_sum, t["low_confidence_sum"], t["high_confidence_sum"])
        and (mean_confidence is None or _between(mean_confidence, t["low_mean_confidence"], t["high_mean_confidence"]))
        and (anchor_spread is None or _between(anchor_spread, t["low_anchor_spread"], t["high_anchor_spread"]))
        and (scale_balance is None or _between(scale_balance, t["low_scale_balance"], t["high_scale_balance"]))
    ):
        base_regime = "ordinary_control_anchor"
    elif (
        tags["semantic_geometry_conflict_tag"]
        or tags["heading_risk"]
        or tags["range_risk"]
        or tags["target_regime_shift_tag"]
    ):
        base_regime = "hard_trainable"
    else:
        base_regime = "ambiguous_unreliable"

    out: dict[str, Any] = {"base_regime": base_regime}
    for regime in BASE_REGIMES:
        out[f"base_{regime}"] = int(regime == base_regime)
    out.update(tags)
    out["ordinary_with_risk_tag"] = int(
        base_regime == "ordinary_control_anchor" and any(tags[tag] for tag in RISK_TAGS)
    )
    return out


def write_manifest_schema_v2(
    path: str | Path,
    thresholds: dict[str, float] | None = None,
    audit: dict[str, Any] | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "base_regimes": BASE_REGIMES,
        "base_flag_columns": BASE_FLAG_COLUMNS,
        "risk_tags": RISK_TAGS,
        "forbidden_construction_patterns": FORBIDDEN_CONSTRUCTION_PATTERNS,
        "allowed_construction_patterns": ALLOWED_CONSTRUCTION_PATTERNS,
        "thresholds": dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds),
        "construction_column_audit": audit,
        "ordinary_control_rule": "positive moderate-evidence anchor, not absence of risk tags",
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_manifest_schema_v2(args.out)
