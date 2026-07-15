#!/usr/bin/env python3
"""Distribution-aware Phase27 A v3 evidence-state calibration.

The calibration core only consumes feature columns derived before target
construction. It fits empirical axis thresholds on train rows when available
and applies the same thresholds to all later splits.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


CALIBRATION_VERSION = "phase27_a_feature_calibration_v1"
AXIS_NAMES = [
    "observability_axis",
    "pair_similarity_axis",
    "scale_compatibility_axis",
    "layout_risk_axis",
]
BAND_VALUES = ["low", "mid", "high", "missing", "conflict"]
BASE_REGIMES = [
    "ordinary_control_anchor",
    "high_evidence_anchor",
    "hard_trainable",
    "low_observable",
    "ambiguous_unreliable",
    "unknown_insufficient_features",
]
FORBIDDEN_CONSTRUCTION_PATTERNS = [
    "heading_num",
    "range_num",
    "gt_angle",
    "gt_distance",
    "final_score",
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

OBSERVABILITY_COLUMNS = [
    "image_a_exists",
    "image_b_exists",
    "has_cheap_image_features",
    "brightness_gap_abs",
    "contrast_gap_abs",
    "grayscale_hist_similarity",
    "aspect_ratio_gap_abs",
]
PAIR_SIMILARITY_COLUMNS = [
    "grayscale_hist_similarity",
    "brightness_gap_abs",
    "contrast_gap_abs",
    "image_index_gap_abs",
    "cached_match_count",
]
SCALE_COMPATIBILITY_COLUMNS = [
    "aspect_ratio_gap_abs",
    "image_a_width",
    "image_a_height",
    "image_b_width",
    "image_b_height",
    "cached_scale_balance",
]
LAYOUT_RISK_COLUMNS = [
    "image_index_gap_abs",
    "query_reference_order",
    "image_a_name",
    "image_b_name",
    "cached_spatial_entropy",
    "cached_anchor_spread",
]
FEATURE_COLUMNS = list(
    dict.fromkeys(
        OBSERVABILITY_COLUMNS
        + PAIR_SIMILARITY_COLUMNS
        + SCALE_COMPATIBILITY_COLUMNS
        + LAYOUT_RISK_COLUMNS
    )
)


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = _lower(column)
    return any(pattern in lowered for pattern in patterns)


def audit_calibration_columns(columns: Iterable[str]) -> dict[str, Any]:
    unique_columns = list(dict.fromkeys(str(column) for column in columns))
    forbidden_columns = [
        column
        for column in unique_columns
        if _matches_any(column, FORBIDDEN_CONSTRUCTION_PATTERNS)
    ]
    feature_columns = [
        column for column in unique_columns if column in FEATURE_COLUMNS and column not in forbidden_columns
    ]
    return {
        "passed": not forbidden_columns,
        "forbidden_columns": forbidden_columns,
        "feature_columns": feature_columns,
        "column_count": len(unique_columns),
    }


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        parsed = float(value)
        return default if math.isnan(parsed) or math.isinf(parsed) else parsed
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def rank01(values: Iterable[Any]) -> list[float | None]:
    parsed = [safe_float(value) for value in values]
    indexed = [(index, value) for index, value in enumerate(parsed) if value is not None]
    if not indexed:
        return [None for _ in parsed]
    indexed.sort(key=lambda item: item[1])
    denominator = max(1, len(indexed) - 1)
    ranks: list[float | None] = [None for _ in parsed]
    for rank, (index, _value) in enumerate(indexed):
        ranks[index] = rank / denominator
    return ranks


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _invert_gap(value: Any, scale: float) -> float | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return _clip01(1.0 - parsed / scale)


def _bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _lower(value) in {"1", "1.0", "true", "yes", "y"}


def _size_compatibility(row: dict[str, Any]) -> float | None:
    aw = safe_float(row.get("image_a_width"))
    ah = safe_float(row.get("image_a_height"))
    bw = safe_float(row.get("image_b_width"))
    bh = safe_float(row.get("image_b_height"))
    if None in {aw, ah, bw, bh}:
        return None
    assert aw is not None and ah is not None and bw is not None and bh is not None
    if min(aw, ah, bw, bh) <= 0:
        return None
    width_ratio = min(aw, bw) / max(aw, bw)
    height_ratio = min(ah, bh) / max(ah, bh)
    return _clip01((width_ratio + height_ratio) / 2.0)


def _same_image_risk(row: dict[str, Any]) -> float | None:
    order = _lower(row.get("query_reference_order", ""))
    if order == "same":
        return 1.0
    name_a = row.get("image_a_name")
    name_b = row.get("image_b_name")
    if name_a is not None and name_b is not None:
        return 1.0 if str(name_a) == str(name_b) else 0.0
    return None


def _observability_score(row: dict[str, Any]) -> float | None:
    image_a_exists = safe_float(row.get("image_a_exists"))
    image_b_exists = safe_float(row.get("image_b_exists"))
    has_cheap = safe_float(row.get("has_cheap_image_features"))
    if image_a_exists is None or image_b_exists is None or has_cheap is None:
        return None
    existence = (image_a_exists + image_b_exists + has_cheap) / 3.0
    quality = _mean(
        [
            safe_float(row.get("grayscale_hist_similarity")),
            _invert_gap(row.get("brightness_gap_abs"), 50.0),
            _invert_gap(row.get("contrast_gap_abs"), 30.0),
            _invert_gap(row.get("aspect_ratio_gap_abs"), 0.5),
        ]
    )
    if quality is None:
        return None
    return _clip01(0.55 * existence + 0.45 * quality)


def _pair_similarity_score(row: dict[str, Any]) -> float | None:
    score = _mean(
        [
            safe_float(row.get("grayscale_hist_similarity")),
            _invert_gap(row.get("brightness_gap_abs"), 50.0),
            _invert_gap(row.get("contrast_gap_abs"), 30.0),
            _invert_gap(row.get("image_index_gap_abs"), 20.0),
        ]
    )
    cached_match_count = safe_float(row.get("cached_match_count"))
    if cached_match_count is not None:
        cached_score = _clip01(cached_match_count / 100.0)
        score = cached_score if score is None else 0.8 * score + 0.2 * cached_score
    return None if score is None else _clip01(score)


def _scale_risk_score(row: dict[str, Any]) -> float | None:
    aspect_risk = safe_float(row.get("aspect_ratio_gap_abs"))
    if aspect_risk is not None:
        aspect_risk = _clip01(aspect_risk / 0.5)
    size_compat = _size_compatibility(row)
    size_risk = None if size_compat is None else 1.0 - size_compat
    cached_scale_balance = safe_float(row.get("cached_scale_balance"))
    cached_risk = None if cached_scale_balance is None else 1.0 - _clip01(cached_scale_balance)
    return _mean([aspect_risk, size_risk, cached_risk])


def _layout_risk_score(row: dict[str, Any]) -> float | None:
    index_risk = safe_float(row.get("image_index_gap_abs"))
    if index_risk is not None:
        index_risk = _clip01(index_risk / 20.0)
    same_risk = _same_image_risk(row)
    entropy = safe_float(row.get("cached_spatial_entropy"))
    entropy_risk = None if entropy is None else 1.0 - _clip01(entropy)
    spread = safe_float(row.get("cached_anchor_spread"))
    spread_risk = None if spread is None else 1.0 - _clip01(spread)
    return _mean([index_risk, same_risk, entropy_risk, spread_risk])


def _axis_scores(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "observability_axis": _observability_score(row),
        "pair_similarity_axis": _pair_similarity_score(row),
        "scale_compatibility_axis": _scale_risk_score(row),
        "layout_risk_axis": _layout_risk_score(row),
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _thresholds(values: list[float]) -> dict[str, float | None]:
    return {"low_mid": _quantile(values, 1.0 / 3.0), "mid_high": _quantile(values, 2.0 / 3.0)}


def fit_calibration(rows: Iterable[dict[str, Any]], split_column: str = "source_split") -> dict[str, Any]:
    row_list = list(rows)
    columns = list(dict.fromkeys(column for item in row_list for column in item.keys()))
    audit = audit_calibration_columns(columns)
    if not audit["passed"]:
        raise ValueError(f"Forbidden calibration columns: {audit['forbidden_columns']}")

    has_split = any(split_column in item for item in row_list)
    if has_split:
        fit_rows = [item for item in row_list if _lower(item.get(split_column)) == "train"]
        fit_scope = f"{split_column}_train"
        if not fit_rows:
            fit_rows = row_list
            fit_scope = "all_rows_no_split"
    else:
        fit_rows = row_list
        fit_scope = "all_rows_no_split"

    values_by_axis: dict[str, list[float]] = {axis: [] for axis in AXIS_NAMES}
    for item in fit_rows:
        for axis, value in _axis_scores(item).items():
            if value is not None:
                values_by_axis[axis].append(value)

    feature_columns_used = [
        column
        for column in FEATURE_COLUMNS
        if any(column in item and safe_float(item.get(column)) is not None for item in fit_rows)
        or any(column in item for item in fit_rows if column in {"query_reference_order", "image_a_name", "image_b_name"})
    ]

    return {
        "calibration_version": CALIBRATION_VERSION,
        "fit_scope": fit_scope,
        "fit_row_count": len(fit_rows),
        "axis_names": list(AXIS_NAMES),
        "band_values": list(BAND_VALUES),
        "base_regimes": list(BASE_REGIMES),
        "axis_thresholds": {
            axis: _thresholds(values)
            for axis, values in values_by_axis.items()
        },
        "feature_columns_used": feature_columns_used,
        "target_columns_used": [],
        "construction_column_audit": audit,
    }


def _band(axis: str, value: float | None, calibration: dict[str, Any]) -> str:
    if value is None:
        return "missing"
    thresholds = calibration.get("axis_thresholds", {}).get(axis, {})
    low_mid = thresholds.get("low_mid")
    mid_high = thresholds.get("mid_high")
    if low_mid is None or mid_high is None:
        return "missing"
    if low_mid > mid_high:
        return "conflict"
    if value < low_mid:
        return "low"
    if value > mid_high:
        return "high"
    return "mid"


def compute_calibrated_axes(row: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    scores = _axis_scores(row)
    out: dict[str, Any] = {}
    for axis in AXIS_NAMES:
        value = scores[axis]
        out[axis] = value
        out[f"{axis}_band"] = _band(axis, value, calibration)
    return out


def _has_required_cheap_features(row: dict[str, Any]) -> bool:
    if not _bool_flag(row.get("has_cheap_image_features")):
        return False
    required = [
        "image_a_exists",
        "image_b_exists",
        "brightness_gap_abs",
        "contrast_gap_abs",
        "grayscale_hist_similarity",
        "aspect_ratio_gap_abs",
    ]
    return all(safe_float(row.get(column)) is not None for column in required)


def assign_calibrated_evidence_state(
    row: dict[str, Any], axes: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    bands = {axis: axes.get(f"{axis}_band", "missing") for axis in AXIS_NAMES}
    missing_axes = [axis for axis, band in bands.items() if band in {"missing", "conflict"}]
    image_missing = not _bool_flag(row.get("image_a_exists")) or not _bool_flag(row.get("image_b_exists"))
    cheap_missing = not _has_required_cheap_features(row)
    risk_tags: list[str] = []

    high_scale_or_layout = (
        bands["scale_compatibility_axis"] == "high"
        or bands["layout_risk_axis"] == "high"
    )
    low_mid_scale_layout = (
        bands["scale_compatibility_axis"] in {"low", "mid"}
        and bands["layout_risk_axis"] in {"low", "mid"}
    )
    high_similarity_with_high_risk = bands["pair_similarity_axis"] == "high" and high_scale_or_layout
    if high_similarity_with_high_risk:
        risk_tags.append("high_similarity_high_risk_conflict")
    if bands["scale_compatibility_axis"] == "high":
        risk_tags.append("high_scale_risk")
    if bands["layout_risk_axis"] == "high":
        risk_tags.append("high_layout_risk")

    if missing_axes or cheap_missing:
        base_regime = "unknown_insufficient_features"
    elif image_missing or bands["observability_axis"] == "low":
        base_regime = "low_observable"
    elif high_similarity_with_high_risk:
        base_regime = "ambiguous_unreliable"
    elif (
        bands["observability_axis"] == "high"
        and bands["pair_similarity_axis"] == "high"
        and low_mid_scale_layout
    ):
        base_regime = "high_evidence_anchor"
    elif (
        bands["observability_axis"] == "mid"
        and bands["pair_similarity_axis"] == "mid"
        and low_mid_scale_layout
    ):
        base_regime = "ordinary_control_anchor"
    elif bands["observability_axis"] in {"mid", "high"} and high_scale_or_layout:
        base_regime = "hard_trainable"
    else:
        base_regime = "ambiguous_unreliable"

    out: dict[str, Any] = {"base_regime": base_regime}
    for regime in BASE_REGIMES:
        out[f"base_{regime}"] = int(regime == base_regime)
    out["risk_tags"] = "|".join(risk_tags)
    out["calibration_source"] = calibration.get("fit_scope", "")
    out["calibration_version"] = calibration.get("calibration_version", CALIBRATION_VERSION)
    for axis in AXIS_NAMES:
        out[axis] = axes.get(axis)
        out[f"{axis}_band"] = axes.get(f"{axis}_band", "missing")
    return out


def summarize_calibration(rows: Iterable[dict[str, Any]], assignments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    assignment_list = list(assignments)
    regime_counts = {regime: 0 for regime in BASE_REGIMES}
    band_counts = {axis: {band: 0 for band in BAND_VALUES} for axis in AXIS_NAMES}
    for assignment in assignment_list:
        regime = assignment.get("base_regime")
        if regime in regime_counts:
            regime_counts[regime] += 1
        for axis in AXIS_NAMES:
            band = assignment.get(f"{axis}_band")
            if band in band_counts[axis]:
                band_counts[axis][band] += 1
    return {
        "calibration_version": CALIBRATION_VERSION,
        "row_count": len(row_list),
        "assignment_count": len(assignment_list),
        "base_regime_counts": regime_counts,
        "axis_band_counts": band_counts,
    }
