"""Phase27 A taxonomy redesign-v2 schema and source-category contract.

This module is intentionally stdlib-only and side-effect free so it can be
staged locally, copied into the remote repository, and imported by tests or
registry-generation jobs without requiring the UAVM runtime environment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE_CATEGORIES = (
    "INPUT_SIDE_NON_LEAKING",
    "MATCHER_SIDE_NON_LEAKING",
    "BASELINE_OUTCOME_VALIDATION_ONLY",
    "OUT_OF_FOLD_TRAINING_ELIGIBLE",
    "ANALYSIS_ONLY",
)

IDENTITY_FIELDS = (
    "canonical_pair_id",
    "source_image_key",
    "target_image_key",
    "target_key",
    "scene_key",
    "split_key",
    "key_schema_version",
)

NON_LEAKING_AXIS_FIELDS = (
    "evidence_sufficiency_score",
    "heading_observability_score",
    "range_observability_score",
    "semantic_geometric_conflict_score",
    "match_sufficiency_score",
    "layout_scale_risk_score",
    "augmentation_consistency_score",
)

ANALYSIS_ONLY_AXIS_FIELDS = (
    "baseline_error_score",
    "heading_error_score",
    "range_error_score",
    "stress_sensitivity_score",
    "checkpoint_disagreement_score",
    "tail_outlier_flag",
)

DERIVED_FIELDS = (
    "ambiguity_tail_risk_score",
    "low_observable_flag",
    "control_stability_score",
    "derived_state",
    "training_readiness_verdict",
    "validation_status",
)

FIELD_SOURCE_CATEGORY = {
    "canonical_pair_id": "INPUT_SIDE_NON_LEAKING",
    "source_image_key": "INPUT_SIDE_NON_LEAKING",
    "target_image_key": "MATCHER_SIDE_NON_LEAKING",
    "target_key": "MATCHER_SIDE_NON_LEAKING",
    "scene_key": "INPUT_SIDE_NON_LEAKING",
    "split_key": "INPUT_SIDE_NON_LEAKING",
    "key_schema_version": "INPUT_SIDE_NON_LEAKING",
    "evidence_sufficiency_score": "MATCHER_SIDE_NON_LEAKING",
    "heading_observability_score": "INPUT_SIDE_NON_LEAKING",
    "range_observability_score": "INPUT_SIDE_NON_LEAKING",
    "semantic_geometric_conflict_score": "MATCHER_SIDE_NON_LEAKING",
    "match_sufficiency_score": "MATCHER_SIDE_NON_LEAKING",
    "layout_scale_risk_score": "MATCHER_SIDE_NON_LEAKING",
    "augmentation_consistency_score": "MATCHER_SIDE_NON_LEAKING",
    "baseline_error_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "heading_error_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "range_error_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "stress_sensitivity_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "checkpoint_disagreement_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "tail_outlier_flag": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "ambiguity_tail_risk_score": "BASELINE_OUTCOME_VALIDATION_ONLY",
    "low_observable_flag": "OUT_OF_FOLD_TRAINING_ELIGIBLE",
    "control_stability_score": "OUT_OF_FOLD_TRAINING_ELIGIBLE",
    "derived_state": "OUT_OF_FOLD_TRAINING_ELIGIBLE",
    "training_readiness_verdict": "OUT_OF_FOLD_TRAINING_ELIGIBLE",
    "validation_status": "ANALYSIS_ONLY",
}

FORBIDDEN_DEPLOYABLE_PATTERNS = (
    "baseline",
    "final_score",
    "final-score",
    "finalscore",
    "residual",
    "error",
)

_DEPLOYABLE_SOURCE_CATEGORIES = {
    "INPUT_SIDE_NON_LEAKING",
    "MATCHER_SIDE_NON_LEAKING",
    "OUT_OF_FOLD_TRAINING_ELIGIBLE",
}


def _all_contract_fields():
    return (
        IDENTITY_FIELDS
        + NON_LEAKING_AXIS_FIELDS
        + ANALYSIS_ONLY_AXIS_FIELDS
        + DERIVED_FIELDS
    )


def validate_schema_columns(columns):
    """Validate a concrete column sequence against the staged schema contract.

    Returns a dictionary instead of raising so upload/controller code can report
    all schema defects at once.
    """

    seen = set()
    duplicate = []
    for column in columns:
        if column in seen and column not in duplicate:
            duplicate.append(column)
        seen.add(column)

    expected = list(_all_contract_fields())
    expected_set = set(expected)
    provided_set = set(columns)
    return {
        "missing": [field for field in expected if field not in provided_set],
        "unknown": [field for field in columns if field not in expected_set],
        "duplicate": duplicate,
    }


def validate_source_categories():
    """Validate that every schema field has exactly one known source category."""

    expected_fields = set(_all_contract_fields())
    registered_fields = set(FIELD_SOURCE_CATEGORY)
    allowed_categories = set(SOURCE_CATEGORIES)
    unknown_categories = {
        field: category
        for field, category in FIELD_SOURCE_CATEGORY.items()
        if category not in allowed_categories
    }
    return {
        "unknown_categories": unknown_categories,
        "missing_fields": [field for field in _all_contract_fields() if field not in registered_fields],
        "extra_fields": sorted(registered_fields - expected_fields),
    }


def assert_no_forbidden_deployable_fields():
    """Assert deployable fields do not contain leakage-prone outcome names."""

    violations = []
    for field, category in FIELD_SOURCE_CATEGORY.items():
        if category not in _DEPLOYABLE_SOURCE_CATEGORIES:
            continue
        lowered = field.lower()
        if any(pattern in lowered for pattern in FORBIDDEN_DEPLOYABLE_PATTERNS):
            violations.append(field)

    if violations:
        raise AssertionError(
            "Forbidden deployable field pattern(s) found: " + ", ".join(sorted(violations))
        )
    return []


def write_source_category_registry(path):
    """Write the field-to-source-category registry as CSV or JSON.

    The output is ordered by the canonical schema field order. A ``.json`` suffix
    writes structured JSON; all other suffixes write CSV.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"field": field, "source_category": FIELD_SOURCE_CATEGORY[field]}
        for field in _all_contract_fields()
    ]

    if path.suffix.lower() == ".json":
        payload = {
            "source_categories": list(SOURCE_CATEGORIES),
            "fields": rows,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("field", "source_category"))
        writer.writeheader()
        writer.writerows(rows)
    return path
