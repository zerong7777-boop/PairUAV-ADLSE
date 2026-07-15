"""Manifest loading and joining helpers for Phase27 A validation spine."""

from __future__ import annotations

import csv
from pathlib import Path

ANALYSIS_ONLY_COLUMNS = {
    "baseline_heading_error",
    "baseline_distance_error",
    "baseline_final_score",
    "prediction_variance",
    "final_score",
    "angle_err",
    "range_err",
    "distance_rel_error",
    "angle_rel_error",
}

DEPLOYABLE_EVIDENCE_COLUMNS = {
    "observability_score",
    "heading_observability_score",
    "range_observability_score",
    "semantic_geometry_conflict_score",
    "ambiguity_score",
    "control_centrality_score",
    "matcher_sufficiency_score",
    "evidence_state",
    "state_confidence",
    "state_rule_version",
}


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fields = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def classify_columns(columns):
    deployable = []
    analysis = []
    identity = []
    other = []
    for column in columns:
        if column in DEPLOYABLE_EVIDENCE_COLUMNS:
            deployable.append(column)
        elif column in ANALYSIS_ONLY_COLUMNS:
            analysis.append(column)
        elif column in {"canonical_pair_id", "source_image_key", "target_image_key", "canonical_group_id"}:
            identity.append(column)
        else:
            other.append(column)
    return {
        "deployable_evidence_columns": deployable,
        "analysis_only_columns": analysis,
        "identity_columns": identity,
        "other_columns": other,
        "leakage_passed": not analysis,
    }


def join_rows_by_key(left_rows, right_rows, key="canonical_pair_id", right_prefix="right"):
    right_by_key = {}
    for row in right_rows:
        value = row.get(key, "")
        if value and value not in right_by_key:
            right_by_key[value] = row

    joined = []
    matched_right = set()
    unmatched_left = 0
    for left in left_rows:
        value = left.get(key, "")
        right = right_by_key.get(value)
        if not value or right is None:
            unmatched_left += 1
            continue
        merged = dict(left)
        for right_key, right_value in right.items():
            if right_key == key:
                continue
            if right_key in merged:
                merged[f"{right_prefix}_{right_key}"] = right_value
            else:
                merged[right_key] = right_value
        joined.append(merged)
        matched_right.add(value)

    unmatched_right = sum(1 for row in right_rows if row.get(key, "") and row.get(key, "") not in matched_right)
    return {
        "rows": joined,
        "joined_rows": joined,
        "unmatched_left_count": unmatched_left,
        "unmatched_right_count": unmatched_right,
    }


def make_shadow_rows(evidence_rows, baseline_rows, matcher_rows=None, reference_rows=None):
    baseline_join = join_rows_by_key(evidence_rows, baseline_rows, "canonical_pair_id", "baseline")
    rows = baseline_join["rows"]
    unmatched = {
        "unmatched_evidence_count": baseline_join["unmatched_left_count"],
        "unmatched_baseline_count": baseline_join["unmatched_right_count"],
    }

    def attach_optional(left_rows, right_rows, prefix):
        right_by_key = {}
        for row in right_rows:
            value = row.get("canonical_pair_id", "")
            if value and value not in right_by_key:
                right_by_key[value] = row
        attached = []
        unmatched_left = 0
        matched_right = set()
        for left in left_rows:
            value = left.get("canonical_pair_id", "")
            right = right_by_key.get(value)
            merged = dict(left)
            if right is None:
                unmatched_left += 1
            else:
                matched_right.add(value)
                for right_key, right_value in right.items():
                    if right_key == "canonical_pair_id":
                        continue
                    out_key = right_key if right_key not in merged else f"{prefix}_{right_key}"
                    merged[out_key] = right_value
            attached.append(merged)
        unmatched_right = sum(1 for row in right_rows if row.get("canonical_pair_id", "") and row.get("canonical_pair_id", "") not in matched_right)
        return attached, unmatched_left, unmatched_right

    if matcher_rows:
        rows, unmatched_left, unmatched_right = attach_optional(rows, matcher_rows, "matcher")
        unmatched["unmatched_shadow_to_matcher_count"] = unmatched_left
        unmatched["unmatched_matcher_count"] = unmatched_right
    if reference_rows:
        rows, unmatched_left, unmatched_right = attach_optional(rows, reference_rows, "reference")
        unmatched["unmatched_shadow_to_reference_count"] = unmatched_left
        unmatched["unmatched_reference_count"] = unmatched_right
    return {"rows": rows, **unmatched}
