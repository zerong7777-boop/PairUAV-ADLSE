#!/usr/bin/env python3
"""Phase27 A evidence-state v2 calibration core.

This module intentionally consumes only pre-target cheap image features. It
uses absolute adequacy and risk gates plus train-split medians for control
centrality; no assignment is produced by fixed quotas.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


CALIBRATION_V2_VERSION = "phase27_a_feature_calibration_v2"
BASE_REGIMES = [
    "ordinary_control_anchor",
    "high_evidence_anchor",
    "hard_trainable",
    "low_observable",
    "ambiguous_unreliable",
    "unknown_insufficient_features",
]
AXIS_NAMES = [
    "observability_axis",
    "pair_similarity_axis",
    "scale_risk_axis",
    "layout_risk_axis",
    "conflict_risk_axis",
    "control_centrality_score",
]
ADEQUACY_FIELDS = [
    "feature_complete",
    "observable_adequate",
    "image_quality_adequate",
    "pair_identity_valid",
    "adequacy_passed",
    "low_observable_reason",
]
FORBIDDEN_CONSTRUCTION_PATTERNS = [
    "final_score",
    "angle_err",
    "range_err",
    "combined_error",
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

REQUIRED_NUMERIC_FIELDS = [
    "brightness_gap_abs",
    "contrast_gap_abs",
    "grayscale_hist_similarity",
    "aspect_ratio_gap_abs",
    "image_index_gap_abs",
    "image_a_width",
    "image_a_height",
    "image_b_width",
    "image_b_height",
]
FEATURE_COLUMNS = [
    "group_id",
    "pair_id",
    "image_a_name",
    "image_b_name",
    "image_a_exists",
    "image_b_exists",
    "has_cheap_image_features",
] + REQUIRED_NUMERIC_FIELDS + [
    "cached_match_count",
    "cached_scale_balance",
    "cached_spatial_entropy",
    "cached_anchor_spread",
    "query_reference_order",
    "source_split",
]

ABSOLUTE_THRESHOLDS = {
    "observability_floor": 0.45,
    "brightness_gap_fail": 60.0,
    "contrast_gap_fail": 45.0,
    "aspect_ratio_gap_fail": 0.70,
    "high_observability": 0.88,
    "high_similarity": 0.88,
    "low_scale_risk": 0.30,
    "low_layout_risk": 0.35,
    "low_conflict_risk": 0.25,
    "high_scale_risk": 0.55,
    "high_layout_risk": 0.55,
    "high_conflict_risk": 0.55,
    "control_centrality": 0.82,
}


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = _lower(column)
    return any(pattern in lowered for pattern in patterns)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _lower(value) in {"1", "1.0", "true", "yes", "y"}


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


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


def canonicalize_image_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    token = re.sub(r"\.(jpeg|jpg|png)$", "", token, flags=re.IGNORECASE)
    token = re.sub(r"^image-", "", token, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+", token):
        return f"{int(token):02d}"
    return token


def canonical_pair_id(row: dict[str, Any]) -> str:
    group_id = str(row.get("group_id", "") or "").strip()
    for key in ("pair_id", "json_id", "pair_key"):
        pair_id = str(row.get(key, "") or "").strip()
        if not pair_id:
            continue
        group = group_id
        rest = pair_id
        if "/" in pair_id:
            group, rest = pair_id.split("/", 1)
            group = group.strip() or group_id
        parts = re.split(r"[_-]", rest, maxsplit=1)
        if len(parts) == 2:
            query = canonicalize_image_token(parts[0])
            reference = canonicalize_image_token(parts[1])
            return f"{group}/{query}_{reference}" if group and query and reference else ""
    image_a = canonicalize_image_token(row.get("image_a_name"))
    image_b = canonicalize_image_token(row.get("image_b_name"))
    return f"{group_id}/{image_a}_{image_b}" if group_id and image_a and image_b else ""


def audit_calibration_v2_columns(columns: Iterable[str]) -> dict[str, Any]:
    unique_columns = list(dict.fromkeys(str(column) for column in columns))
    forbidden_columns = [
        column for column in unique_columns if _matches_any(column, FORBIDDEN_CONSTRUCTION_PATTERNS)
    ]
    return {
        "passed": not forbidden_columns,
        "forbidden_columns": forbidden_columns,
        "feature_columns": [
            column for column in unique_columns if column in FEATURE_COLUMNS and column not in forbidden_columns
        ],
        "column_count": len(unique_columns),
    }


def _invert_gap(value: Any, scale: float) -> float | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return _clip01(1.0 - parsed / scale)


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
    return _clip01((min(aw, bw) / max(aw, bw) + min(ah, bh) / max(ah, bh)) / 2.0)


def compute_observability_axis(row: dict[str, Any]) -> float | None:
    image_a_exists = safe_float(row.get("image_a_exists"))
    image_b_exists = safe_float(row.get("image_b_exists"))
    has_cheap = safe_float(row.get("has_cheap_image_features"))
    if image_a_exists is None or image_b_exists is None or has_cheap is None:
        return None
    existence = (image_a_exists + image_b_exists + has_cheap) / 3.0
    quality = _mean(
        [
            safe_float(row.get("grayscale_hist_similarity")),
            _invert_gap(row.get("brightness_gap_abs"), 60.0),
            _invert_gap(row.get("contrast_gap_abs"), 45.0),
            _invert_gap(row.get("aspect_ratio_gap_abs"), 0.70),
        ]
    )
    if quality is None:
        return None
    return _clip01(0.50 * existence + 0.50 * quality)


def compute_pair_similarity_axis(row: dict[str, Any]) -> float | None:
    score = _mean(
        [
            safe_float(row.get("grayscale_hist_similarity")),
            _invert_gap(row.get("brightness_gap_abs"), 60.0),
            _invert_gap(row.get("contrast_gap_abs"), 45.0),
            _invert_gap(row.get("image_index_gap_abs"), 20.0),
        ]
    )
    cached_match_count = safe_float(row.get("cached_match_count"))
    if cached_match_count is not None:
        cached_score = _clip01(cached_match_count / 100.0)
        score = cached_score if score is None else 0.80 * score + 0.20 * cached_score
    return None if score is None else _clip01(score)


def compute_scale_risk_axis(row: dict[str, Any]) -> float | None:
    aspect_gap = safe_float(row.get("aspect_ratio_gap_abs"))
    aspect_risk = None if aspect_gap is None else _clip01(aspect_gap / 0.70)
    size_compat = _size_compatibility(row)
    size_risk = None if size_compat is None else 1.0 - size_compat
    cached_balance = safe_float(row.get("cached_scale_balance"))
    cached_risk = None if cached_balance is None else 1.0 - _clip01(cached_balance)
    return _mean([aspect_risk, size_risk, cached_risk])


def compute_layout_risk_axis(row: dict[str, Any]) -> float | None:
    index_gap = safe_float(row.get("image_index_gap_abs"))
    index_risk = None if index_gap is None else _clip01(index_gap / 20.0)
    same_order = 1.0 if _lower(row.get("query_reference_order", "")) == "same" else None
    if row.get("image_a_name") is not None and row.get("image_b_name") is not None:
        same_order = 1.0 if str(row.get("image_a_name")) == str(row.get("image_b_name")) else 0.0
    entropy = safe_float(row.get("cached_spatial_entropy"))
    entropy_risk = None if entropy is None else 1.0 - _clip01(entropy)
    spread = safe_float(row.get("cached_anchor_spread"))
    spread_risk = None if spread is None else 1.0 - _clip01(spread)
    return _mean([index_risk, same_order, entropy_risk, spread_risk])


def compute_conflict_risk_axis(row: dict[str, Any], axes: dict[str, Any]) -> float | None:
    similarity = axes.get("pair_similarity_axis")
    scale_risk = axes.get("scale_risk_axis")
    layout_risk = axes.get("layout_risk_axis")
    if similarity is None or (scale_risk is None and layout_risk is None):
        return None
    risk_peak = max(value for value in [scale_risk, layout_risk] if value is not None)
    high_similarity_high_risk = _clip01((similarity - 0.70) / 0.30) * _clip01((risk_peak - 0.35) / 0.65)
    same_pair = 0.0
    if canonicalize_image_token(row.get("image_a_name")) and (
        canonicalize_image_token(row.get("image_a_name")) == canonicalize_image_token(row.get("image_b_name"))
    ):
        same_pair = 1.0
    return _clip01(max(high_similarity_high_risk, same_pair))


def compute_adequacy(row: dict[str, Any], axes: dict[str, Any]) -> dict[str, Any]:
    pair_id = canonical_pair_id(row)
    image_a_exists_value = row.get("image_a_exists")
    image_b_exists_value = row.get("image_b_exists")
    images_exist_known = not _is_missing_value(image_a_exists_value) and not _is_missing_value(image_b_exists_value)
    images_exist = images_exist_known and _bool_flag(image_a_exists_value) and _bool_flag(image_b_exists_value)
    has_cheap_value = row.get("has_cheap_image_features")
    has_cheap_known = not _is_missing_value(has_cheap_value)
    has_cheap = has_cheap_known and _bool_flag(has_cheap_value)
    required_numeric_present = all(safe_float(row.get(field)) is not None for field in REQUIRED_NUMERIC_FIELDS)
    feature_complete = int(images_exist and has_cheap and required_numeric_present)
    observability = axes.get("observability_axis")
    observable_adequate = int(observability is not None and observability >= ABSOLUTE_THRESHOLDS["observability_floor"])

    brightness = safe_float(row.get("brightness_gap_abs"))
    contrast = safe_float(row.get("contrast_gap_abs"))
    aspect = safe_float(row.get("aspect_ratio_gap_abs"))
    image_quality_adequate = int(
        brightness is not None
        and contrast is not None
        and aspect is not None
        and brightness <= ABSOLUTE_THRESHOLDS["brightness_gap_fail"]
        and contrast <= ABSOLUTE_THRESHOLDS["contrast_gap_fail"]
        and aspect <= ABSOLUTE_THRESHOLDS["aspect_ratio_gap_fail"]
    )
    pair_identity_valid = int(bool(pair_id))

    reasons: list[str] = []
    if images_exist_known and not images_exist:
        reasons.append("image_missing")
    if has_cheap_known and not has_cheap:
        reasons.append("cheap_features_missing")
    if has_cheap and required_numeric_present and not observable_adequate:
        reasons.append("observability_below_floor")
    if has_cheap and required_numeric_present and not image_quality_adequate:
        reasons.append("image_quality_failure")

    adequacy_passed = int(
        feature_complete
        and observable_adequate
        and image_quality_adequate
        and pair_identity_valid
    )
    return {
        "feature_complete": feature_complete,
        "observable_adequate": observable_adequate,
        "image_quality_adequate": image_quality_adequate,
        "pair_identity_valid": pair_identity_valid,
        "adequacy_passed": adequacy_passed,
        "low_observable_reason": "|".join(reasons),
    }


def _raw_axes(row: dict[str, Any]) -> dict[str, float | None]:
    axes = {
        "observability_axis": compute_observability_axis(row),
        "pair_similarity_axis": compute_pair_similarity_axis(row),
        "scale_risk_axis": compute_scale_risk_axis(row),
        "layout_risk_axis": compute_layout_risk_axis(row),
    }
    axes["conflict_risk_axis"] = compute_conflict_risk_axis(row, axes)
    return axes


def fit_calibration_v2(rows: Iterable[dict[str, Any]], fit_split: str = "train") -> dict[str, Any]:
    row_list = list(rows)
    columns = list(dict.fromkeys(column for item in row_list for column in item.keys()))
    audit = audit_calibration_v2_columns(columns)
    if not audit["passed"]:
        raise ValueError(f"Forbidden calibration columns: {audit['forbidden_columns']}")

    has_split = any("source_split" in item for item in row_list)
    fit_rows = [item for item in row_list if _lower(item.get("source_split")) == _lower(fit_split)] if has_split else row_list
    fit_scope = f"source_split_{fit_split}" if has_split and fit_rows else "all_rows_no_split"
    if not fit_rows:
        fit_rows = row_list

    values_by_axis: dict[str, list[float]] = {axis: [] for axis in AXIS_NAMES if axis != "control_centrality_score"}
    for item in fit_rows:
        axes = _raw_axes(item)
        for axis, value in axes.items():
            if value is not None:
                values_by_axis[axis].append(value)

    axis_medians = {axis: _median(values) for axis, values in values_by_axis.items()}
    feature_columns_used = [
        column
        for column in FEATURE_COLUMNS
        if column != "source_split" and any(column in item for item in fit_rows)
    ]
    return {
        "calibration_version": CALIBRATION_V2_VERSION,
        "fit_scope": fit_scope,
        "fit_row_count": len(fit_rows),
        "axis_names": list(AXIS_NAMES),
        "base_regimes": list(BASE_REGIMES),
        "axis_medians": axis_medians,
        "absolute_thresholds": dict(ABSOLUTE_THRESHOLDS),
        "control_centrality_threshold": ABSOLUTE_THRESHOLDS["control_centrality"],
        "feature_columns_used": feature_columns_used,
        "target_columns_used": [],
        "construction_column_audit": audit,
    }


def compute_control_centrality(
    row: dict[str, Any],
    axes: dict[str, Any],
    adequacy: dict[str, Any],
    calibration: dict[str, Any],
) -> float | None:
    if not adequacy.get("adequacy_passed"):
        return None
    medians = calibration.get("axis_medians", {})
    obs = axes.get("observability_axis")
    sim = axes.get("pair_similarity_axis")
    scale = axes.get("scale_risk_axis")
    layout = axes.get("layout_risk_axis")
    conflict = axes.get("conflict_risk_axis")
    if None in {obs, sim, scale, layout, conflict}:
        return None
    obs_med = medians.get("observability_axis")
    sim_med = medians.get("pair_similarity_axis")
    if obs_med is None or sim_med is None:
        return None
    central_obs = 1.0 - min(1.0, abs(obs - obs_med) / 0.50)
    central_sim = 1.0 - min(1.0, abs(sim - sim_med) / 0.50)
    low_risk = (1.0 - scale + 1.0 - layout + 1.0 - conflict) / 3.0
    return _clip01(0.15 + 0.40 * ((central_obs + central_sim) / 2.0) + 0.45 * low_risk)


def compute_calibrated_axes_v2(row: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, Any] = _raw_axes(row)
    adequacy = compute_adequacy(row, axes)
    axes["control_centrality_score"] = compute_control_centrality(row, axes, adequacy, calibration)
    return axes


def _risk_tags(axes: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if axes.get("scale_risk_axis") is not None and axes["scale_risk_axis"] >= ABSOLUTE_THRESHOLDS["high_scale_risk"]:
        tags.append("high_scale_risk")
    if axes.get("layout_risk_axis") is not None and axes["layout_risk_axis"] >= ABSOLUTE_THRESHOLDS["high_layout_risk"]:
        tags.append("high_layout_risk")
    if axes.get("conflict_risk_axis") is not None and axes["conflict_risk_axis"] >= ABSOLUTE_THRESHOLDS["high_conflict_risk"]:
        tags.append("high_conflict_risk")
    if (
        axes.get("pair_similarity_axis") is not None
        and axes.get("pair_similarity_axis") >= 0.80
        and (("high_scale_risk" in tags) or ("high_layout_risk" in tags))
    ):
        tags.append("high_similarity_high_risk_conflict")
    return tags


def assign_evidence_state_v2(
    row: dict[str, Any],
    axes: dict[str, Any],
    adequacy: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    thresholds = calibration.get("absolute_thresholds", ABSOLUTE_THRESHOLDS)
    pair_id = canonical_pair_id(row)
    missing_numeric = any(safe_float(row.get(field)) is None for field in REQUIRED_NUMERIC_FIELDS)
    missing_image_existence = _is_missing_value(row.get("image_a_exists")) or _is_missing_value(row.get("image_b_exists"))
    missing_cheap_flag = _is_missing_value(row.get("has_cheap_image_features"))
    missing_required = (
        not adequacy.get("pair_identity_valid")
        or missing_cheap_flag
        or missing_image_existence
        or missing_numeric
    )
    low_observable = bool(adequacy.get("low_observable_reason"))
    scale = axes.get("scale_risk_axis")
    layout = axes.get("layout_risk_axis")
    conflict = axes.get("conflict_risk_axis")
    obs = axes.get("observability_axis")
    sim = axes.get("pair_similarity_axis")
    centrality = axes.get("control_centrality_score")
    tags = _risk_tags(axes)

    low_abs_risk = (
        scale is not None
        and layout is not None
        and conflict is not None
        and scale <= thresholds["low_scale_risk"]
        and layout <= thresholds["low_layout_risk"]
        and conflict <= thresholds["low_conflict_risk"]
    )
    high_evidence = (
        adequacy.get("adequacy_passed")
        and obs is not None
        and sim is not None
        and obs >= thresholds["high_observability"]
        and sim >= thresholds["high_similarity"]
        and low_abs_risk
    )
    high_conflict = conflict is not None and conflict >= thresholds["high_conflict_risk"]
    high_trainable_risk = (
        (scale is not None and scale >= thresholds["high_scale_risk"])
        or (layout is not None and layout >= thresholds["high_layout_risk"])
    )

    if missing_required:
        base_regime = "unknown_insufficient_features"
    elif low_observable:
        base_regime = "low_observable"
    elif high_conflict:
        base_regime = "ambiguous_unreliable"
    elif high_evidence:
        base_regime = "high_evidence_anchor"
    elif (
        adequacy.get("adequacy_passed")
        and low_abs_risk
        and centrality is not None
        and centrality >= calibration.get("control_centrality_threshold", thresholds["control_centrality"])
    ):
        base_regime = "ordinary_control_anchor"
    elif adequacy.get("adequacy_passed") and high_trainable_risk:
        base_regime = "hard_trainable"
    elif adequacy.get("adequacy_passed"):
        base_regime = "ambiguous_unreliable"
    else:
        base_regime = "unknown_insufficient_features"

    out: dict[str, Any] = {"canonical_pair_id": pair_id}
    out.update({field: adequacy.get(field) for field in ADEQUACY_FIELDS})
    out.update({axis: axes.get(axis) for axis in AXIS_NAMES})
    out["base_regime"] = base_regime
    for regime in BASE_REGIMES:
        out[f"base_{regime}"] = int(regime == base_regime)
    out["risk_tags"] = "|".join(tags)
    out["calibration_version"] = calibration.get("calibration_version", CALIBRATION_V2_VERSION)
    out["calibration_source"] = calibration.get("fit_scope", "")
    return out


def summarize_calibration_v2(
    rows: Iterable[dict[str, Any]],
    assignments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    row_list = list(rows)
    assignment_list = list(assignments)
    regime_counts = {regime: 0 for regime in BASE_REGIMES}
    for assignment in assignment_list:
        regime = assignment.get("base_regime")
        if regime in regime_counts:
            regime_counts[regime] += 1
    return {
        "calibration_version": CALIBRATION_V2_VERSION,
        "row_count": len(row_list),
        "assignment_count": len(assignment_list),
        "base_regime_counts": regime_counts,
    }
