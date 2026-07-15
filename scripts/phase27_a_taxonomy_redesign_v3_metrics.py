"""Metrics and audits for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from scripts import phase27_a_taxonomy_redesign_v3_schema as schema


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def count_boolean_fields(rows: Iterable[dict], fields: list[str]) -> dict[str, int]:
    counts = {field: 0 for field in fields}
    for row in rows:
        for field in fields:
            if _is_true(row.get(field)):
                counts[field] += 1
    return counts


def compute_multilabel_overlap(rows: list[dict], fields: list[str]) -> dict:
    pairwise = {}
    hard_ambiguous = 0
    for i, field_a in enumerate(fields):
        for field_b in fields[i + 1:]:
            overlap = sum(1 for row in rows if _is_true(row.get(field_a)) and _is_true(row.get(field_b)))
            pairwise[f"{field_a}__AND__{field_b}"] = overlap
    for row in rows:
        hard = any(_is_true(row.get(field)) for field in schema.HEADING_RANGE_HARD_FIELDS)
        ambiguous = any(_is_true(row.get(field)) for field in schema.AMBIGUITY_SUBTYPE_FIELDS)
        if hard and ambiguous:
            hard_ambiguous += 1
    return {"pairwise_overlap": pairwise, "hard_ambiguity_overlap_count": hard_ambiguous}


def compute_ambiguity_subtype_breakdown(rows: list[dict]) -> dict:
    return {
        "total_rows": len(rows),
        "subtype_counts": count_boolean_fields(rows, schema.AMBIGUITY_SUBTYPE_FIELDS),
    }


def compute_heading_range_hard_split(rows: list[dict]) -> dict:
    fields = [
        "evidence_sufficient_heading_hard",
        "evidence_sufficient_range_hard",
        "evidence_sufficient_joint_hard",
        "heading_stress_sensitive",
        "range_stress_sensitive",
    ]
    counts = count_boolean_fields(rows, fields)
    counts["heading_and_range_hard_overlap"] = sum(
        1 for row in rows
        if _is_true(row.get("evidence_sufficient_heading_hard")) and _is_true(row.get("evidence_sufficient_range_hard"))
    )
    counts["heading_only_hard"] = sum(
        1 for row in rows
        if _is_true(row.get("evidence_sufficient_heading_hard")) and not _is_true(row.get("evidence_sufficient_range_hard"))
    )
    counts["range_only_hard"] = sum(
        1 for row in rows
        if _is_true(row.get("evidence_sufficient_range_hard")) and not _is_true(row.get("evidence_sufficient_heading_hard"))
    )
    return counts


def compute_baseline_stress_consistency_audit(rows: list[dict]) -> dict:
    baseline = [row for row in rows if _is_true(row.get("baseline_joint_hard"))]
    stress = [row for row in rows if _is_true(row.get("stress_joint_sensitive"))]
    overlap = [
        row for row in rows
        if _is_true(row.get("baseline_joint_hard")) and _is_true(row.get("stress_joint_sensitive"))
    ]
    verdict = "expected-axis-difference"
    if baseline and stress and not overlap:
        verdict = "zero-overlap-axis-difference-or-normalization-mismatch"
    if not baseline or not stress:
        verdict = "insufficient-positive-signal"
    return {
        "baseline_joint_hard_count": len(baseline),
        "stress_joint_sensitive_count": len(stress),
        "overlap_count": len(overlap),
        "overlap_ratio_vs_baseline": len(overlap) / len(baseline) if baseline else 0.0,
        "overlap_ratio_vs_stress": len(overlap) / len(stress) if stress else 0.0,
        "verdict": verdict,
        "metric_interpretation": "baseline is absolute error; stress is robustness delta",
    }


def compute_join_coverage_bias_report(rows: list[dict]) -> dict:
    def bucket(status_field: str) -> dict[str, int]:
        counts = {}
        for row in rows:
            status = str(row.get(status_field) or "missing")
            counts[status] = counts.get(status, 0) + 1
        return counts

    return {
        "total_rows": len(rows),
        "full_dev_join_status": bucket("full_dev_join_status"),
        "stress_join_status": bucket("stress_join_status"),
        "validation_status": bucket("validation_status"),
    }


def compute_leakage_deployability_audit() -> dict:
    schema.validate_field_registry()
    rows = []
    violations = []
    for field in sorted(schema.SOURCE_CATEGORY_BY_FIELD):
        deployability = schema.DEPLOYABILITY_BY_FIELD[field]
        uses_validation = schema.SOURCE_CATEGORY_BY_FIELD[field] in {
            "baseline_outcome_validation_only",
            "stress_outcome_validation_only",
            "forbidden_leakage",
        }
        violation = deployability == "deployable" and uses_validation
        if violation:
            violations.append(field)
        rows.append({
            "field": field,
            "source_category": schema.SOURCE_CATEGORY_BY_FIELD[field],
            "deployability": deployability,
            "uses_validation_or_forbidden_source": uses_validation,
            "violation": violation,
        })
    return {"violations": violations, "passed": not violations, "fields": rows}


def compute_no_go_training_policy_boundary(rows: list[dict]) -> dict:
    return {
        "training_policy_allowed": False,
        "reason": "A-v3 bounded eval and knowledge-review are required before any training policy.",
        "row_count": len(rows),
        "forbidden_actions": [
            "training",
            "finetuning",
            "sample_weighting",
            "curriculum",
            "oversampling",
            "submission_packaging",
            "B/C gate training from A states",
        ],
    }


def write_metrics_bundle(rows: list[dict], output_dir: str | Path) -> dict:
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "ambiguity_subtype_breakdown": compute_ambiguity_subtype_breakdown(rows),
        "heading_range_hard_split": compute_heading_range_hard_split(rows),
        "baseline_stress_consistency_audit": compute_baseline_stress_consistency_audit(rows),
        "multilabel_overlap_report": compute_multilabel_overlap(
            rows,
            schema.AMBIGUITY_SUBTYPE_FIELDS + schema.HEADING_RANGE_HARD_FIELDS + schema.LAYER3_READINESS_FIELDS,
        ),
        "join_coverage_bias_report": compute_join_coverage_bias_report(rows),
        "leakage_deployability_audit": compute_leakage_deployability_audit(),
        "no_go_training_policy_boundary": compute_no_go_training_policy_boundary(rows),
    }
    for name, data in bundle.items():
        (metrics_dir / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return bundle
