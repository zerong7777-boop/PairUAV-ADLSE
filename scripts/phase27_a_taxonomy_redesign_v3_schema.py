"""Schema and deployability contract for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "phase27_a_taxonomy_redesign_v3_schema_v1"
DEPLOYABILITY_LABELS = {"deployable", "validation_only", "analysis_only", "forbidden"}
SOURCE_CATEGORIES = {
    "input_side_non_leaking",
    "matcher_side_non_leaking",
    "baseline_outcome_validation_only",
    "stress_outcome_validation_only",
    "derived_readiness_analysis_only",
    "forbidden_leakage",
}

IDENTITY_FIELDS = [
    "canonical_pair_id", "source_split", "json_path", "group_id", "pair_id", "pair_key",
    "source_image_key", "target_image_key", "target_key", "scene_key", "split_key",
]
LAYER1_CANDIDATE_FIELDS = [
    "evidence_sufficient_candidate", "heading_observable_candidate", "range_observable_candidate",
    "low_observable_candidate", "semantic_geometric_conflict_candidate",
    "local_alignment_needed_candidate", "ambiguity_candidate", "ordinary_candidate", "control_candidate",
]
LAYER2_OUTCOME_FIELDS = [
    "baseline_heading_hard", "baseline_range_hard", "baseline_joint_hard",
    "stress_heading_sensitive", "stress_range_sensitive", "stress_joint_sensitive",
    "tail_error_high", "checkpoint_disagreement_high", "augmentation_instability_high",
]
LAYER3_READINESS_FIELDS = [
    "READY_CONTROL_PRESERVATION", "READY_HEADING_HARD_TRAINING", "READY_RANGE_HARD_TRAINING",
    "READY_CORRESPONDENCE_DIAGNOSTIC", "READY_WEAK_SUPERVISION", "READY_MULTI_HYPOTHESIS",
    "QUARANTINE_LOW_OBSERVABLE", "ANALYSIS_ONLY", "NOT_READY",
]
AMBIGUITY_SUBTYPE_FIELDS = [
    "low_evidence_ambiguous", "multi_modal_ambiguous", "semantic_geometric_conflict",
    "stress_sensitive_ambiguous", "tail_error_unreliable",
]
HEADING_RANGE_HARD_FIELDS = [
    "evidence_sufficient_heading_hard", "evidence_sufficient_range_hard",
    "evidence_sufficient_joint_hard", "heading_stress_sensitive", "range_stress_sensitive",
]
REASON_FIELDS = [
    "candidate_reason_codes", "ambiguity_reason_codes", "missing_candidate_source_fields",
    "outcome_reason_codes", "verdict_reason_codes",
]
STATUS_FIELDS = ["full_dev_join_status", "stress_join_status", "validation_status"]
REQUIRED_FIELDS = (
    IDENTITY_FIELDS + LAYER1_CANDIDATE_FIELDS + LAYER2_OUTCOME_FIELDS + LAYER3_READINESS_FIELDS
    + AMBIGUITY_SUBTYPE_FIELDS + HEADING_RANGE_HARD_FIELDS + REASON_FIELDS + STATUS_FIELDS
)
FORBIDDEN_DEPLOYABLE_PATTERNS = [
    r"(^|_)gt($|_)",
    r"baseline_.*(error|score|pred|residual)",
    r"stress_.*(error|score|delta|residual)",
    r"official_.*score",
    r"final_state",
    r"derived_state",
    r"hard_trainable",
]

SOURCE_CATEGORY_BY_FIELD = {}
DEPLOYABILITY_BY_FIELD = {}
for field in IDENTITY_FIELDS:
    SOURCE_CATEGORY_BY_FIELD[field] = "input_side_non_leaking"
    DEPLOYABILITY_BY_FIELD[field] = "deployable"
for field in LAYER1_CANDIDATE_FIELDS + AMBIGUITY_SUBTYPE_FIELDS:
    SOURCE_CATEGORY_BY_FIELD[field] = "input_side_non_leaking"
    DEPLOYABILITY_BY_FIELD[field] = "deployable"
for field in LAYER2_OUTCOME_FIELDS + HEADING_RANGE_HARD_FIELDS:
    SOURCE_CATEGORY_BY_FIELD[field] = (
        "stress_outcome_validation_only" if "stress" in field else "baseline_outcome_validation_only"
    )
    DEPLOYABILITY_BY_FIELD[field] = "validation_only"
for field in LAYER3_READINESS_FIELDS + REASON_FIELDS + STATUS_FIELDS:
    SOURCE_CATEGORY_BY_FIELD[field] = "derived_readiness_analysis_only"
    DEPLOYABILITY_BY_FIELD[field] = "analysis_only"
for field in ["hard_trainable", "base_hard_trainable", "final_state", "derived_state", "training_readiness_verdict"]:
    SOURCE_CATEGORY_BY_FIELD[field] = "forbidden_leakage"
    DEPLOYABILITY_BY_FIELD[field] = "forbidden"


def all_registered_fields() -> list[str]:
    return list(SOURCE_CATEGORY_BY_FIELD.keys())


def validate_required_fields(columns: Iterable[str]) -> bool:
    present = set(columns)
    missing = [field for field in REQUIRED_FIELDS if field not in present]
    if missing:
        raise ValueError("missing required v3 fields: " + ", ".join(missing))
    return True


def validate_field_registry() -> bool:
    errors = []
    for field in REQUIRED_FIELDS:
        category = SOURCE_CATEGORY_BY_FIELD.get(field)
        deployability = DEPLOYABILITY_BY_FIELD.get(field)
        if category not in SOURCE_CATEGORIES:
            errors.append(f"invalid source category for {field}: {category}")
        if deployability not in DEPLOYABILITY_LABELS:
            errors.append(f"invalid deployability for {field}: {deployability}")
    if errors:
        raise ValueError("; ".join(errors))
    assert_no_forbidden_deployable_fields()
    return True


def assert_no_forbidden_deployable_fields() -> bool:
    patterns = [re.compile(pattern) for pattern in FORBIDDEN_DEPLOYABLE_PATTERNS]
    violations = [
        field for field, deployability in DEPLOYABILITY_BY_FIELD.items()
        if deployability == "deployable" and any(pattern.search(field) for pattern in patterns)
    ]
    if violations:
        raise ValueError("forbidden deployable fields: " + ", ".join(sorted(violations)))
    return True


def write_schema_registry(path: str | Path) -> None:
    validate_field_registry()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "deployability_labels": sorted(DEPLOYABILITY_LABELS),
        "source_categories": sorted(SOURCE_CATEGORIES),
        "required_fields": REQUIRED_FIELDS,
        "source_category_by_field": SOURCE_CATEGORY_BY_FIELD,
        "deployability_by_field": DEPLOYABILITY_BY_FIELD,
        "forbidden_deployable_patterns": FORBIDDEN_DEPLOYABLE_PATTERNS,
    }, indent=2, sort_keys=True), encoding="utf-8")
