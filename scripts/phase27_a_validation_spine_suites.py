"""Phase 27A validation-spine suite checks.

This module is intentionally standalone: it uses only the Python standard
library and accepts plain dictionaries/lists so later workers can feed combined
metrics without depending on task-specific modules.
"""

from __future__ import annotations

from collections.abc import Mapping


ALLOWED_VERDICTS = {
    "A-validation-spine-ready-for-training-policy-spec",
    "A-validation-spine-reference-unresolved-but-shadow-valid",
    "A-validation-spine-needs-redesign",
    "A-route-rejected-for-now",
}


def _as_mapping(value):
    return value if isinstance(value, Mapping) else {}


def _to_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        is_percent = cleaned.endswith("%")
        if is_percent:
            cleaned = cleaned[:-1].strip()
        try:
            parsed = float(cleaned)
        except ValueError:
            return default
        return parsed if not is_percent else parsed
    return default


def _first_numeric(mapping, keys, default=None):
    data = _as_mapping(mapping)
    for key in keys:
        parsed = _to_float(data.get(key), None)
        if parsed is not None:
            return parsed
    return default


def _first_value(mapping, keys, default=None):
    data = _as_mapping(mapping)
    for key in keys:
        if key in data:
            return data[key]
    return default


def _passed(result):
    return bool(_as_mapping(result).get("passed"))


def _flatten_forbidden_fields(value, forbidden, prefix=""):
    hits = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = key_text if not prefix else prefix + "." + key_text
            if key_text in forbidden:
                hits.append(path)
            hits.extend(_flatten_forbidden_fields(nested, forbidden, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            path = "%s[%d]" % (prefix, index) if prefix else "[%d]" % index
            hits.extend(_flatten_forbidden_fields(nested, forbidden, path))
    return hits


def identity_suite(evidence_to_baseline=None, evidence_to_reference=None):
    """Validate identity evidence against baseline and optional reference."""

    baseline = _as_mapping(evidence_to_baseline)
    reference = _as_mapping(evidence_to_reference)

    baseline_overlap = _first_numeric(
        baseline,
        ("canonical_overlap", "overlap", "canonical_overlap_pct"),
        0.0,
    )
    duplicates = _first_numeric(
        baseline,
        ("duplicates", "duplicate_count", "canonical_duplicates"),
        0.0,
    )
    collisions = _first_numeric(
        baseline,
        ("collisions", "collision_count", "canonical_collisions"),
        0.0,
    )

    reference_overlap = _first_numeric(
        reference,
        ("canonical_overlap", "overlap", "canonical_overlap_pct"),
        0.0,
    )
    failure_classification = _first_value(
        reference,
        ("failure_classification", "failure", "failure_class"),
        "",
    )

    baseline_passed = (
        baseline_overlap >= 100.0 and duplicates == 0.0 and collisions == 0.0
    )
    if failure_classification == "reference_too_small":
        reference_status = "reference_unresolved"
        reference_passed = True
    elif reference_overlap > 0.0:
        reference_status = "reference_resolved"
        reference_passed = True
    else:
        reference_status = "reference_failed"
        reference_passed = False

    return {
        "passed": baseline_passed and reference_passed,
        "baseline_passed": baseline_passed,
        "baseline_canonical_overlap": baseline_overlap,
        "duplicates": duplicates,
        "collisions": collisions,
        "reference_passed": reference_passed,
        "reference_status": reference_status,
        "reference_canonical_overlap": reference_overlap,
        "failure_classification": failure_classification,
    }


def lineage_suite(lineage_evidence=None, required_fields=None):
    """Validate lineage diagnostics without assuming a concrete schema."""

    evidence = _as_mapping(lineage_evidence)
    required = tuple(required_fields or ())
    missing_required = [field for field in required if not evidence.get(field)]
    reported_missing = list(evidence.get("missing_fields") or ())
    broken_links = list(evidence.get("broken_links") or ())
    missing = sorted(set(missing_required + reported_missing))
    passed = not missing and not broken_links and not bool(evidence.get("lineage_error"))

    return {
        "passed": passed,
        "missing_lineage_fields": missing,
        "broken_lineage_links": broken_links,
        "lineage_error": evidence.get("lineage_error"),
    }


def leakage_suite(deployable_evidence=None, forbidden_fields=None):
    """Reject deployable evidence that contains label or outcome leakage."""

    forbidden = set(
        forbidden_fields
        or (
            "final_score",
            "label",
            "target",
            "ground_truth",
            "oracle_score",
            "evaluation_score",
        )
    )
    fields = _flatten_forbidden_fields(deployable_evidence, forbidden)

    return {
        "passed": len(fields) == 0,
        "forbidden_deployable_fields": fields,
        "forbidden_field_count": len(fields),
    }


def state_distribution_suite(distribution=None, max_unknown_fraction=0.50):
    """Check that validation states are not dominated by unknown examples."""

    data = _as_mapping(distribution)
    ordinary = _first_numeric(data, ("ordinary", "ordinary_fraction"), 0.0)
    hard = _first_numeric(data, ("hard", "hard_fraction"), 0.0)
    ambiguous = _first_numeric(data, ("ambiguous", "ambiguous_fraction"), 0.0)
    unknown = _first_numeric(data, ("unknown", "unknown_fraction"), 0.0)
    known_fraction = ordinary + hard + ambiguous
    total_fraction = known_fraction + unknown

    passed = (
        unknown <= max_unknown_fraction
        and ordinary > 0.0
        and hard > 0.0
        and total_fraction > 0.0
    )
    return {
        "passed": passed,
        "ordinary_fraction": ordinary,
        "hard_fraction": hard,
        "ambiguous_fraction": ambiguous,
        "unknown_fraction": unknown,
        "known_fraction": known_fraction,
        "total_fraction": total_fraction,
        "max_unknown_fraction": max_unknown_fraction,
    }


def state_error_association_suite(scores=None, min_hard_minus_control_delta=0.0):
    """Check whether harder states have higher observed error than controls."""

    data = _as_mapping(scores)
    ordinary = _first_numeric(data, ("ordinary", "ordinary_score", "control"), None)
    hard = _first_numeric(data, ("hard", "hard_score"), None)
    ambiguous = _first_numeric(data, ("ambiguous", "ambiguous_score"), None)

    if ordinary is None or hard is None:
        delta = None
        passed = False
    else:
        delta = hard - ordinary
        passed = delta > min_hard_minus_control_delta
        if ambiguous is not None:
            passed = passed and ambiguous >= hard

    return {
        "passed": passed,
        "ordinary_score": ordinary,
        "hard_score": hard,
        "ambiguous_score": ambiguous,
        "hard_minus_control_delta": delta,
        "min_hard_minus_control_delta": min_hard_minus_control_delta,
    }


def matcher_sufficiency_suite(matcher_evidence=None, min_coverage=0.95):
    """Validate matcher coverage and unresolved-pair diagnostics."""

    data = _as_mapping(matcher_evidence)
    coverage = _first_numeric(data, ("coverage", "matcher_coverage"), None)
    unresolved = _first_numeric(data, ("unresolved", "unresolved_pairs"), 0.0)
    collisions = _first_numeric(data, ("collisions", "collision_count"), 0.0)
    if coverage is None:
        matched = _first_numeric(data, ("matched", "matched_pairs"), None)
        total = _first_numeric(data, ("total", "total_pairs"), None)
        coverage = (matched / total) if matched is not None and total else 0.0

    passed = coverage >= min_coverage and unresolved == 0.0 and collisions == 0.0
    return {
        "passed": passed,
        "matcher_coverage": coverage,
        "min_coverage": min_coverage,
        "unresolved_pairs": unresolved,
        "collision_count": collisions,
    }


def control_stability_suite(scores=None, min_hard_minus_ordinary_delta=0.0):
    """Check that hard controls remain separated from ordinary controls."""

    data = _as_mapping(scores)
    ordinary = _first_numeric(data, ("ordinary", "ordinary_score", "control"), None)
    hard = _first_numeric(data, ("hard", "hard_score"), None)
    if ordinary is None or hard is None:
        delta = None
        passed = False
    else:
        delta = hard - ordinary
        passed = delta > min_hard_minus_ordinary_delta

    return {
        "passed": passed,
        "ordinary_score": ordinary,
        "hard_score": hard,
        "hard_minus_ordinary_delta": delta,
        "min_hard_minus_ordinary_delta": min_hard_minus_ordinary_delta,
    }


def training_readiness_suite(
    identity=None,
    lineage=None,
    leakage=None,
    distribution=None,
    state_error=None,
    control=None,
    matcher=None,
):
    """Combine suite outputs into an allowed Phase 27A readiness verdict."""

    required = {
        "identity": identity,
        "lineage": lineage,
        "leakage": leakage,
        "distribution": distribution,
        "state_error": state_error,
        "control": control,
    }
    if matcher is not None:
        required["matcher"] = matcher

    failed_suites = [name for name, result in required.items() if not _passed(result)]
    identity_status = _as_mapping(identity).get("reference_status")

    if failed_suites:
        verdict = "A-validation-spine-needs-redesign"
        passed = False
    elif identity_status == "reference_unresolved":
        verdict = "A-validation-spine-reference-unresolved-but-shadow-valid"
        passed = True
    elif identity_status in ("reference_resolved", "reference_verified"):
        verdict = "A-validation-spine-ready-for-training-policy-spec"
        passed = True
    else:
        verdict = "A-route-rejected-for-now"
        passed = False

    return {
        "passed": passed,
        "verdict": verdict,
        "allowed_verdicts": sorted(ALLOWED_VERDICTS),
        "failed_suites": failed_suites,
        "reference_status": identity_status,
    }
