"""Derived-state rules for Phase27 A taxonomy redesign-v2."""

from __future__ import annotations

import math


REQUIRED_AXIS_FIELDS = (
    "evidence_sufficiency_score",
    "heading_observability_score",
    "range_observability_score",
    "semantic_geometric_conflict_score",
    "match_sufficiency_score",
    "layout_scale_risk_score",
    "augmentation_consistency_score",
    "baseline_error_score",
    "stress_sensitivity_score",
    "ambiguity_tail_risk_score",
    "control_stability_score",
)


def safe_float(value, default):
    """Convert numeric-like input to float, falling back for missing/bad values."""

    if value is None or value == "":
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def default_thresholds():
    return {
        "evidence_sufficiency_high": 0.67,
        "evidence_sufficiency_low": 0.33,
        "observability_low": 0.33,
        "conflict_high": 0.67,
        "baseline_difficulty_high": 0.67,
        "stress_sensitivity_high": 0.67,
        "tail_risk_high": 0.67,
        "control_stability_high": 0.67,
        "control_stability_low": 0.33,
    }


def assign_derived_state(row, thresholds=None):
    """Assign one conservative derived state for a canonical pair row."""

    thresholds = _merged_thresholds(thresholds)
    if not _has_value(row.get("canonical_pair_id")):
        return "unknown_unvalidated"
    if any(not _has_value(row.get(field)) for field in REQUIRED_AXIS_FIELDS):
        return "unknown_unvalidated"

    evidence = safe_float(row.get("evidence_sufficiency_score"), 0.0)
    match_sufficiency = safe_float(row.get("match_sufficiency_score"), 0.0)
    heading_observability = safe_float(row.get("heading_observability_score"), 0.0)
    range_observability = safe_float(row.get("range_observability_score"), 0.0)
    observability = min(heading_observability, range_observability)
    conflict = safe_float(row.get("semantic_geometric_conflict_score"), 0.0)
    baseline_error = safe_float(row.get("baseline_error_score"), 0.0)
    stress_sensitivity = safe_float(row.get("stress_sensitivity_score"), 0.0)
    tail_risk = safe_float(row.get("ambiguity_tail_risk_score"), 0.0)
    control_stability = safe_float(row.get("control_stability_score"), 0.0)

    if observability <= thresholds["observability_low"]:
        return "low_observable"
    if evidence <= thresholds["evidence_sufficiency_low"] or tail_risk >= thresholds["tail_risk_high"]:
        return "ambiguous_unreliable"
    if stress_sensitivity >= thresholds["stress_sensitivity_high"]:
        return "stress_sensitive_control"
    if conflict >= thresholds["conflict_high"]:
        return "conflict_candidate"
    if (
        baseline_error >= thresholds["baseline_difficulty_high"]
        and evidence >= thresholds["evidence_sufficiency_high"]
        and match_sufficiency >= thresholds["evidence_sufficiency_high"]
    ):
        return "evidence_sufficient_hard"
    if (
        baseline_error >= thresholds["baseline_difficulty_high"]
        or control_stability <= thresholds["control_stability_low"]
    ):
        return "ambiguous_unreliable"
    if (
        evidence >= thresholds["evidence_sufficiency_high"]
        and match_sufficiency >= thresholds["evidence_sufficiency_high"]
        and control_stability >= thresholds["control_stability_high"]
    ):
        return "stable_control_anchor"
    return "unknown_unvalidated"


def assign_training_readiness_verdict(derived_state):
    mapping = {
        "stable_control_anchor": "READY_FOR_ANCHOR_CANDIDATE",
        "evidence_sufficient_hard": "READY_FOR_HARD_TRAINING_CANDIDATE",
        "conflict_candidate": "READY_FOR_CORRESPONDENCE_ROUTING_CANDIDATE",
        "low_observable": "QUARANTINE_OR_WEAK_SUPERVISION_CANDIDATE",
        "ambiguous_unreliable": "QUARANTINE_OR_WEAK_SUPERVISION_CANDIDATE",
        "stress_sensitive_control": "ANALYSIS_ONLY",
        "unknown_unvalidated": "NOT_READY",
    }
    return mapping.get(derived_state, "NOT_READY")


def assign_rows(rows_or_dataframe, thresholds=None):
    """Return copied records with derived state and readiness verdict assigned."""

    if hasattr(rows_or_dataframe, "to_dict"):
        rows = rows_or_dataframe.to_dict("records")
    else:
        rows = rows_or_dataframe

    assigned = []
    for row in rows:
        copied = dict(row)
        state = assign_derived_state(copied, thresholds=thresholds)
        copied["derived_state"] = state
        copied["training_readiness_verdict"] = assign_training_readiness_verdict(state)
        assigned.append(copied)
    return assigned


def _merged_thresholds(thresholds):
    merged = default_thresholds()
    if thresholds:
        merged.update(thresholds)
    return merged


def _has_value(value):
    return value is not None and value != ""
