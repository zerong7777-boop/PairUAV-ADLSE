#!/usr/bin/env python3
"""Build Phase27 calibration-v3 overlap validation artifacts.

This script is evaluation-only: it reads existing manifests/metrics, audits
pair identity overlap, writes matched control-preservation surfaces, and emits
reports. It does not train, infer, create submissions, or modify calibration-v2
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_v3_overlap_identity import (
    attach_canonical_keys,
    audit_forbidden_columns,
    classify_order,
    count_key_duplicates,
    key_coverage,
)


CONTROL_REGIME = "ordinary_control_anchor"

MATCHED_FIELDS = [
    "canonical_pair_id",
    "reference_source_name",
    "reference_source_row_index",
    "v3_source_row_index",
    "join_strategy",
    "order_status",
    "reference_group_id_raw",
    "reference_query_raw",
    "reference_reference_raw",
    "v3_group_id_raw",
    "v3_query_raw",
    "v3_reference_raw",
    "reference_base_regime",
    "reference_control_tag",
    "v3_base_regime",
    "v3_risk_tags",
    "v3_feature_complete",
    "v3_observable_adequate",
    "v3_image_quality_adequate",
    "v3_pair_identity_valid",
    "v3_control_centrality_score",
    "v3_scale_risk_axis",
    "v3_layout_risk_axis",
    "v3_conflict_risk_axis",
    "preservation_status",
]


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for idx, row in enumerate(reader):
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_path(value: str) -> Path:
    return Path(value).expanduser()


def normalize_rows(rows: list[dict[str, Any]], source_name: str, source_path: Path) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows):
        out = attach_canonical_keys(row)
        out["source_name"] = source_name
        out["source_path"] = str(source_path)
        out["source_row_index"] = idx
        normalized.append(out)
    return normalized


def row_value(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def group_id(row: dict[str, Any]) -> str:
    return row_value(row, "canonical_group_id", "group_id", "group", "target")


def query_id(row: dict[str, Any]) -> str:
    return row_value(row, "canonical_query_id", "query_name", "query_image", "image_a_name", "image_a")


def reference_id(row: dict[str, Any]) -> str:
    return row_value(row, "canonical_reference_id", "reference_name", "reference_image", "image_b_name", "image_b")


def orderless_key(row: dict[str, Any]) -> str:
    return row_value(row, "canonical_pair_id_orderless")


def pair_key(row: dict[str, Any]) -> str:
    return row_value(row, "canonical_pair_id")


def raw_pair_key(row: dict[str, Any]) -> str:
    return row_value(row, "pair_id", "pair_key")


def image_token_pair(row: dict[str, Any]) -> str:
    left = query_id(row)
    right = reference_id(row)
    if not left or not right:
        return ""
    return "_".join(sorted([left, right]))


def set_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]], fn) -> int:
    left_keys = {fn(row) for row in left if fn(row)}
    right_keys = {fn(row) for row in right if fn(row)}
    return len(left_keys & right_keys)


def first_unmatched(rows: list[dict[str, Any]], matched_ids: set[int], limit: int = 10) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if id(row) in matched_ids:
            continue
        out.append({
            "source_row_index": row.get("source_row_index", ""),
            "pair_id": raw_pair_key(row),
            "canonical_pair_id": pair_key(row),
            "canonical_pair_id_orderless": orderless_key(row),
            "group_id": group_id(row),
            "query_id": query_id(row),
            "reference_id": reference_id(row),
        })
        if len(out) >= limit:
            break
    return out


def index_by(rows: list[dict[str, Any]], fn) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = fn(row)
        if key:
            indexed[key].append(row)
    return indexed


def is_control_row(row: dict[str, Any]) -> bool:
    if row_value(row, "base_regime") == CONTROL_REGIME:
        return True
    if row_value(row, "base_ordinary_control_anchor") in ("1", "true", "True", "yes"):
        return True
    haystack = " ".join(
        row_value(row, "reference_control_tag", "risk_tags", "ordinary_with_risk_tag").lower().split()
    )
    return "ordinary_control_anchor" in haystack or "control" in haystack


def reference_control_semantics(rows: list[dict[str, Any]]) -> str:
    fields = set()
    for row in rows:
        fields.update(row.keys())
    if "base_regime" in fields or "base_ordinary_control_anchor" in fields:
        return "base_regime"
    if {"reference_control_tag", "risk_tags", "ordinary_with_risk_tag"} & fields:
        return "tag"
    return "missing"


def make_match(ref: dict[str, Any], v3: dict[str, Any], strategy: str) -> dict[str, Any]:
    ref_control = is_control_row(ref)
    v3_control = is_control_row(v3)
    status = "preserved" if (not ref_control or v3_control) else "regressed"
    return {
        "canonical_pair_id": pair_key(ref) or pair_key(v3),
        "reference_source_name": ref.get("source_name", ""),
        "reference_source_row_index": ref.get("source_row_index", ""),
        "v3_source_row_index": v3.get("source_row_index", ""),
        "join_strategy": strategy,
        "order_status": classify_order(pair_key(ref), pair_key(v3)),
        "reference_group_id_raw": row_value(ref, "group_id", "group"),
        "reference_query_raw": row_value(ref, "query_name", "query_image", "image_a_name", "image_a"),
        "reference_reference_raw": row_value(ref, "reference_name", "reference_image", "image_b_name", "image_b"),
        "v3_group_id_raw": row_value(v3, "group_id", "group"),
        "v3_query_raw": row_value(v3, "image_a_name", "image_a", "query_image"),
        "v3_reference_raw": row_value(v3, "image_b_name", "image_b", "reference_image"),
        "reference_base_regime": row_value(ref, "base_regime"),
        "reference_control_tag": row_value(ref, "reference_control_tag", "ordinary_with_risk_tag", "risk_tags"),
        "v3_base_regime": row_value(v3, "base_regime"),
        "v3_risk_tags": row_value(v3, "risk_tags"),
        "v3_feature_complete": row_value(v3, "feature_complete"),
        "v3_observable_adequate": row_value(v3, "observable_adequate"),
        "v3_image_quality_adequate": row_value(v3, "image_quality_adequate"),
        "v3_pair_identity_valid": row_value(v3, "pair_identity_valid"),
        "v3_control_centrality_score": row_value(v3, "control_centrality_score"),
        "v3_scale_risk_axis": row_value(v3, "scale_risk_axis"),
        "v3_layout_risk_axis": row_value(v3, "layout_risk_axis"),
        "v3_conflict_risk_axis": row_value(v3, "conflict_risk_axis"),
        "preservation_status": status,
    }


def build_matches(reference_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    same_index = index_by(v3_rows, pair_key)
    flipped_index = index_by(v3_rows, lambda row: row_value(row, "canonical_pair_id_flipped"))
    orderless_index = index_by(v3_rows, orderless_key)
    used_ref: set[int] = set()
    used_v3: set[int] = set()
    matches: list[dict[str, Any]] = []

    for strategy, key_fn, index in [
        ("same_order", pair_key, same_index),
        ("flipped_order", pair_key, flipped_index),
        ("orderless", orderless_key, orderless_index),
    ]:
        for ref in reference_rows:
            if id(ref) in used_ref:
                continue
            key = key_fn(ref)
            candidates = [row for row in index.get(key, []) if id(row) not in used_v3]
            if not candidates:
                continue
            v3 = candidates[0]
            matches.append(make_match(ref, v3, strategy))
            used_ref.add(id(ref))
            used_v3.add(id(v3))
    return matches, used_ref, used_v3


def source_audit(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {
        "source_name": name,
        "row_count": len(rows),
        "canonical_pair_id_coverage": key_coverage(rows, "canonical_pair_id"),
        "canonical_pair_id_orderless_coverage": key_coverage(rows, "canonical_pair_id_orderless"),
        "canonical_pair_id_duplicates": count_key_duplicates(rows, "canonical_pair_id"),
        "canonical_pair_id_orderless_duplicates": count_key_duplicates(rows, "canonical_pair_id_orderless"),
    }


def classify_failure(reference_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]], matches: list[dict[str, Any]]) -> str:
    ref_cov = key_coverage(reference_rows, "canonical_pair_id")["nonempty_fraction"]
    v3_cov = key_coverage(v3_rows, "canonical_pair_id")["nonempty_fraction"]
    if ref_cov < 0.50 or v3_cov < 0.50:
        return "missing_identity_fields"
    if len(reference_rows) < 20 and not matches:
        return "reference_too_small"
    if matches:
        if len(reference_rows) < 20:
            return "reference_too_small"
        return "overlap_available"
    if set_overlap(reference_rows, v3_rows, group_id) > 0 or set_overlap(reference_rows, v3_rows, image_token_pair) > 0:
        return "namespace_mismatch"
    if set_overlap(reference_rows, v3_rows, pair_key) == 0 and set_overlap(reference_rows, v3_rows, orderless_key) == 0:
        return "true_non_overlap"
    return "mixed_or_unresolved"


def gate_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "passed", "pass")
    return default


def build_full_regression(v2_metrics: dict[str, Any]) -> dict[str, Any]:
    row_count = int(v2_metrics.get("row_count") or 0)
    ordinary = float(v2_metrics.get("ordinary_control_anchor_fraction") or 0.0)
    high = float(v2_metrics.get("high_evidence_anchor_fraction") or 0.0)
    max_bucket = float(v2_metrics.get("max_base_regime_fraction") or 1.0)
    quota = bool(v2_metrics.get("quota_only_suspected", True))
    leakage = v2_metrics.get("leakage_audit", {})
    leakage_passed = gate_bool(leakage.get("passed") if isinstance(leakage, dict) else leakage, True)
    exactly_one = gate_bool(v2_metrics.get("exactly_one_base_regime_passed"), False)
    gates = {
        "coverage_gate_passed": row_count == 60000,
        "leakage_gate_passed": leakage_passed,
        "exactly_one_base_regime_gate_passed": exactly_one,
        "ordinary_control_gate_passed": ordinary >= 0.15,
        "high_evidence_gate_passed": high < 0.70,
        "max_bucket_gate_passed": max_bucket < 0.85,
        "quota_gate_passed": not quota,
    }
    gates["full_surface_regression_passed"] = all(gates.values())
    return {
        "row_count": row_count,
        "leakage_audit": leakage,
        "exactly_one_base_regime_passed": exactly_one,
        "base_regime_counts": v2_metrics.get("base_regime_counts", {}),
        "base_regime_fractions": v2_metrics.get("base_regime_fractions", {}),
        "ordinary_control_anchor_fraction": ordinary,
        "high_evidence_anchor_fraction": high,
        "max_base_regime_fraction": max_bucket,
        "quota_only_suspected": quota,
        "low_observable_reason_counts": v2_metrics.get("low_observable_reason_counts", {}),
        "adequacy_counts": v2_metrics.get("adequacy_counts", {}),
        "gates": gates,
    }


def build_control_metrics(matches: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref_control_count = sum(1 for row in reference_rows if is_control_row(row))
    matched_ref_control_count = sum(1 for match in matches if match["reference_base_regime"] == CONTROL_REGIME)
    v3_control_on_matched = sum(1 for match in matches if match["v3_base_regime"] == CONTROL_REGIME)
    preserved = sum(1 for match in matches if match["preservation_status"] == "preserved")
    regressed = sum(1 for match in matches if match["preservation_status"] == "regressed")
    semantics = reference_control_semantics(reference_rows)
    preservation_rate = preserved / matched_ref_control_count if matched_ref_control_count else 0.0
    return {
        "matched_row_count": len(matches),
        "same_order_count": sum(1 for match in matches if match["join_strategy"] == "same_order"),
        "flipped_order_count": sum(1 for match in matches if match["join_strategy"] == "flipped_order"),
        "orderless_count": sum(1 for match in matches if match["join_strategy"] == "orderless"),
        "reference_control_semantics": semantics,
        "reference_ordinary_control_count": ref_control_count,
        "matched_reference_ordinary_control_count": matched_ref_control_count,
        "v3_ordinary_control_count_on_matched": v3_control_on_matched,
        "preserved_ordinary_control_count": preserved,
        "regressed_ordinary_control_count": regressed,
        "preservation_rate": preservation_rate,
        "promotion_eligible_overlap": len(matches) >= 20 and preservation_rate >= 0.90 and semantics != "missing",
        "disagreement_examples": [match for match in matches if match["preservation_status"] == "regressed"][:10],
    }


def decide_verdict(identity_audit: dict[str, Any], control: dict[str, Any], full: dict[str, Any]) -> str:
    full_passed = bool(full["gates"]["full_surface_regression_passed"])
    if not full_passed:
        return "A-route-rejected-for-now"
    if control["promotion_eligible_overlap"]:
        return "calibration-v3-ready-for-A-training-policy-plan"
    if control["matched_row_count"] < 20 or control["reference_control_semantics"] == "missing":
        return "calibration-v3-blocked-by-reference-surface"
    if identity_audit["failure_classification"] in {"missing_identity_fields", "implementation_bug", "mixed_or_unresolved"}:
        return "calibration-v3-needs-redesign"
    return "A-route-rejected-for-now"


def write_report(path: Path, verdict: str, identity: dict[str, Any], control: dict[str, Any], full: dict[str, Any]) -> None:
    lines = [
        "# Phase27 A Evidence-State Calibration-v3 Overlap Validation",
        "",
        f"- Verdict: `{verdict}`",
        f"- Failure classification: `{identity['failure_classification']}`",
        f"- Reference rows: `{identity['reference']['row_count']}`",
        f"- V3 rows: `{identity['v3']['row_count']}`",
        f"- Matched rows: `{control['matched_row_count']}`",
        f"- Matched reference ordinary/control rows: `{control['matched_reference_ordinary_control_count']}`",
        f"- Preservation rate: `{control['preservation_rate']:.6f}`",
        f"- Full-surface regression passed: `{full['gates']['full_surface_regression_passed']}`",
        "",
        "## Overlap Counts",
        "",
    ]
    for key, value in identity["overlap"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Full-Surface Gates", ""])
    for key, value in full["gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Unmatched Reference Examples", "", "```json"])
    lines.append(json.dumps(identity["first_unmatched_reference_examples"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Unmatched V3 Examples", "", "```json"])
    lines.append(json.dumps(identity["first_unmatched_v3_examples"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--v3-calibrated-manifest", required=True)
    parser.add_argument("--v3-calibrated-axes", required=True)
    parser.add_argument("--v2-reference-manifest", required=True)
    parser.add_argument("--v2-reference-metrics", required=True)
    parser.add_argument("--v2-calibration-metrics", required=True)
    parser.add_argument("--extra-reference-manifest", action="append", default=[])
    parser.add_argument("--identity-audit-json", required=True)
    parser.add_argument("--matched-surface-out", required=True)
    parser.add_argument("--control-metrics-json", required=True)
    parser.add_argument("--full-regression-json", required=True)
    parser.add_argument("--combined-metrics-json", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=["smoke", "bounded"], required=True)
    args = parser.parse_args()

    v3_manifest_path = as_path(args.v3_calibrated_manifest)
    v3_axes_path = as_path(args.v3_calibrated_axes)
    v2_reference_path = as_path(args.v2_reference_manifest)
    v2_metrics_path = as_path(args.v2_calibration_metrics)

    v3_rows = normalize_rows(read_csv(v3_manifest_path, args.limit), "v3_calibrated_manifest", v3_manifest_path)
    axes_rows = normalize_rows(read_csv(v3_axes_path, args.limit), "v3_calibrated_axes", v3_axes_path)
    reference_rows = normalize_rows(read_csv(v2_reference_path, args.limit), "v2_reference_manifest", v2_reference_path)
    for extra in args.extra_reference_manifest:
        extra_path = as_path(extra)
        reference_rows.extend(normalize_rows(read_csv(extra_path, args.limit), "extra_reference_manifest", extra_path))

    v3_audit = audit_forbidden_columns(v3_rows[0].keys() if v3_rows else [])
    ref_audit = audit_forbidden_columns(reference_rows[0].keys() if reference_rows else [])
    matches, used_ref, used_v3 = build_matches(reference_rows, v3_rows)
    identity_audit = {
        "mode": args.mode,
        "training_started": False,
        "submission_created": False,
        "v3_calibrated_axes_row_count": len(axes_rows),
        "v3_forbidden_columns_present_but_ignored": v3_audit["forbidden_columns"],
        "reference_forbidden_columns_present_but_ignored": ref_audit["forbidden_columns"],
        "reference": source_audit(reference_rows, "reference"),
        "v3": source_audit(v3_rows, "v3_calibrated_manifest"),
        "overlap": {
            "raw_pair_id_overlap_count": set_overlap(reference_rows, v3_rows, raw_pair_key),
            "canonical_same_order_overlap_count": sum(1 for match in matches if match["join_strategy"] == "same_order"),
            "canonical_flipped_overlap_count": sum(1 for match in matches if match["join_strategy"] == "flipped_order"),
            "canonical_orderless_overlap_count": set_overlap(reference_rows, v3_rows, orderless_key),
            "group_only_overlap_count": set_overlap(reference_rows, v3_rows, group_id),
            "image_name_only_overlap_count": set_overlap(reference_rows, v3_rows, image_token_pair),
        },
        "first_unmatched_reference_examples": first_unmatched(reference_rows, used_ref),
        "first_unmatched_v3_examples": first_unmatched(v3_rows, used_v3),
    }
    identity_audit["failure_classification"] = classify_failure(reference_rows, v3_rows, matches)

    control_metrics = build_control_metrics(matches, reference_rows)
    full_regression = build_full_regression(read_json(v2_metrics_path))
    verdict = decide_verdict(identity_audit, control_metrics, full_regression)
    combined = {
        "verdict": verdict,
        "mode": args.mode,
        "training_started": False,
        "submission_created": False,
        "identity_audit_metrics": identity_audit,
        "control_preservation_metrics": control_metrics,
        "full_surface_regression_metrics": full_regression,
    }

    write_json(as_path(args.identity_audit_json), identity_audit)
    write_csv(as_path(args.matched_surface_out), matches, MATCHED_FIELDS)
    write_json(as_path(args.control_metrics_json), control_metrics)
    write_json(as_path(args.full_regression_json), full_regression)
    write_json(as_path(args.combined_metrics_json), combined)
    write_report(as_path(args.report_out), verdict, identity_audit, control_metrics, full_regression)
    print(f"verdict={verdict}")
    print(f"matched_row_count={control_metrics['matched_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
