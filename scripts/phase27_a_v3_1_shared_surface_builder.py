"""Build a shared baseline/stress canonical-pair outcome surface."""
from __future__ import annotations

from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import (
    ensure_output_dirs,
    read_csv_dicts,
    stable_pair_key,
    to_float,
    truthy,
    write_csv_dicts,
)


CANDIDATE_FIELDS = [
    "canonical_pair_id",
    "source_split",
    "json_path",
    "group_id",
    "pair_id",
    "pair_key",
    "source_image_key",
    "target_image_key",
    "target_key",
    "scene_key",
    "split_key",
    "evidence_sufficient_candidate",
    "heading_observable_candidate",
    "range_observable_candidate",
    "low_observable_candidate",
    "semantic_geometric_conflict_candidate",
    "local_alignment_needed_candidate",
    "ambiguity_candidate",
    "ordinary_candidate",
    "control_candidate",
    "candidate_reason_codes",
    "multi_modal_ambiguous",
    "semantic_geometric_conflict",
    "stress_sensitive_ambiguous",
    "tail_error_unreliable",
    "READY_CONTROL_PRESERVATION",
]
BASELINE_FIELDS = [
    "baseline_joined",
    "baseline_join_status",
    "baseline_pred_angle",
    "baseline_pred_distance",
    "baseline_angle_abs_error",
    "baseline_angle_rel_error",
    "baseline_distance_abs_error",
    "baseline_distance_rel_error",
    "baseline_joint_error_score",
    "baseline_heading_hard",
    "baseline_range_hard",
    "baseline_joint_hard",
]


def _first(row, names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _status_joined(row, status_field, bool_field=None):
    if row.get(status_field):
        return str(row.get(status_field)).strip().lower() == "joined"
    if bool_field and row.get(bool_field) not in (None, ""):
        return truthy(row.get(bool_field))
    return True


def normalize_baseline_row(row):
    return {
        "baseline_pred_angle": _first(row, ["baseline_pred_angle", "pred_angle"]),
        "baseline_pred_distance": _first(row, ["baseline_pred_distance", "pred_distance"]),
        "baseline_angle_abs_error": _first(row, ["baseline_angle_abs_error", "heading_error_score"]),
        "baseline_angle_rel_error": _first(row, ["baseline_angle_rel_error"]),
        "baseline_distance_abs_error": _first(row, ["baseline_distance_abs_error"]),
        "baseline_distance_rel_error": _first(row, ["baseline_distance_rel_error", "range_error_score"]),
        "baseline_joint_error_score": _first(row, ["baseline_joint_error_score", "baseline_error_score"]),
        "baseline_heading_hard": _first(row, ["baseline_heading_hard", "evidence_sufficient_heading_hard"]),
        "baseline_range_hard": _first(row, ["baseline_range_hard", "evidence_sufficient_range_hard"]),
        "baseline_joint_hard": _first(row, ["baseline_joint_hard", "evidence_sufficient_joint_hard"]),
    }


def normalize_stress_row(row, variant_name):
    prefix = f"stress_{variant_name}_"
    return {
        f"{prefix}pred_angle": _first(row, [f"{prefix}pred_angle", "stress_baseline_pred_angle", "baseline_pred_angle", "pred_angle"]),
        f"{prefix}pred_distance": _first(row, [f"{prefix}pred_distance", "stress_baseline_pred_distance", "baseline_pred_distance", "pred_distance"]),
        f"{prefix}angle_abs_error": _first(row, [f"{prefix}angle_abs_error", "stress_baseline_angle_abs_error", "baseline_angle_abs_error"]),
        f"{prefix}angle_rel_error": _first(row, [f"{prefix}angle_rel_error", "stress_baseline_angle_rel_error", "baseline_angle_rel_error"]),
        f"{prefix}distance_abs_error": _first(row, [f"{prefix}distance_abs_error", "stress_baseline_distance_abs_error", "baseline_distance_abs_error"]),
        f"{prefix}distance_rel_error": _first(row, [f"{prefix}distance_rel_error", "stress_baseline_distance_rel_error", "baseline_distance_rel_error"]),
        f"{prefix}joint_error_score": _first(row, [f"{prefix}joint_error_score", "stress_baseline_final_score", "baseline_final_score"]),
        f"{prefix}heading_sensitive": _first(row, [f"{prefix}heading_sensitive", "stress_heading_sensitive", "heading_stress_sensitive"]),
        f"{prefix}range_sensitive": _first(row, [f"{prefix}range_sensitive", "stress_range_sensitive", "range_stress_sensitive"]),
        f"{prefix}joint_sensitive": _first(row, [f"{prefix}joint_sensitive", "stress_joint_sensitive"]),
    }


def index_surface_rows(rows, surface_name):
    rows_by_key = {}
    duplicate_keys = {}
    key_sources = {}
    for row in rows:
        key_source, key = stable_pair_key(row)
        if not key:
            continue
        key_sources[key] = key_source
        if key in rows_by_key:
            duplicate_keys.setdefault(key, [rows_by_key[key]]).append(row)
        else:
            rows_by_key[key] = row
    return {"surface_name": surface_name, "rows_by_key": rows_by_key, "duplicate_keys": duplicate_keys, "key_sources": key_sources}


def _delta(stress_value, baseline_value):
    if stress_value in (None, "") or baseline_value in (None, ""):
        return ""
    return to_float(stress_value) - to_float(baseline_value)


def build_shared_row(candidate_row, baseline_index, stress_indexes):
    key_source, key = stable_pair_key(candidate_row)
    row = {field: candidate_row.get(field, "") for field in CANDIDATE_FIELDS}
    row["canonical_pair_id"] = row.get("canonical_pair_id") or key
    row["join_key_source"] = key_source
    row["join_key_confidence"] = "high" if key_source in {"canonical_pair_id", "source_target_pair_composite"} else "fallback"
    reasons = []

    baseline_duplicate = key in baseline_index["duplicate_keys"]
    baseline_src = baseline_index["rows_by_key"].get(key)
    if baseline_duplicate:
        row["baseline_joined"] = "false"
        row["baseline_join_status"] = "duplicate_blocked"
        reasons.append("duplicate_baseline")
    elif baseline_src and _status_joined(baseline_src, "full_dev_join_status", "full_dev_joined"):
        row["baseline_joined"] = "true"
        row["baseline_join_status"] = "joined"
        row.update(normalize_baseline_row(baseline_src))
    else:
        row["baseline_joined"] = "false"
        row["baseline_join_status"] = "missing"
        reasons.append("missing_baseline")

    any_stress_joined = False
    duplicate_stress = False
    for variant, index in stress_indexes.items():
        prefix = f"stress_{variant}_"
        stress_duplicate = key in index["duplicate_keys"]
        stress_src = index["rows_by_key"].get(key)
        if stress_duplicate:
            duplicate_stress = True
            row[f"{prefix}joined"] = "false"
            row[f"{prefix}join_status"] = "duplicate_blocked"
            reasons.append(f"duplicate_stress_{variant}")
            continue
        if stress_src and _status_joined(stress_src, "stress_join_status", "stress_joined"):
            any_stress_joined = True
            row[f"{prefix}joined"] = "true"
            row[f"{prefix}join_status"] = "joined"
            row.update(normalize_stress_row(stress_src, variant))
            row[f"{prefix}heading_delta"] = _delta(row.get(f"{prefix}angle_abs_error"), row.get("baseline_angle_abs_error"))
            row[f"{prefix}range_delta"] = _delta(row.get(f"{prefix}distance_rel_error"), row.get("baseline_distance_rel_error"))
            row[f"{prefix}joint_delta"] = _delta(row.get(f"{prefix}joint_error_score"), row.get("baseline_joint_error_score"))
            if row[f"{prefix}heading_delta"] == "" or row[f"{prefix}range_delta"] == "" or row[f"{prefix}joint_delta"] == "":
                reasons.append("missing_error_for_delta")
        else:
            row[f"{prefix}joined"] = "false"
            row[f"{prefix}join_status"] = "missing"
            reasons.append(f"missing_stress_{variant}")

    baseline_joined = row.get("baseline_joined") == "true"
    if baseline_duplicate or duplicate_stress:
        shared_status = "duplicate_blocked"
    elif baseline_joined and any_stress_joined:
        shared_status = "joined"
    elif not baseline_joined and not any_stress_joined:
        shared_status = "missing_both"
    elif not baseline_joined:
        shared_status = "missing_baseline"
    else:
        shared_status = "missing_stress"
    row["shared_baseline_stress_joined"] = "true" if shared_status == "joined" else "false"
    row["shared_join_status"] = shared_status
    row["missing_surface_reason_codes"] = "|".join(sorted(set(reasons)))
    return row


def build_shared_surface(candidate_rows, baseline_rows, stress_surface_map):
    baseline_index = index_surface_rows(baseline_rows, "baseline")
    stress_indexes = {name: index_surface_rows(rows, name) for name, rows in stress_surface_map.items()}
    return [build_shared_row(row, baseline_index, stress_indexes) for row in candidate_rows]


def shared_surface_fieldnames(stress_variants):
    fields = list(CANDIDATE_FIELDS) + BASELINE_FIELDS
    for variant in stress_variants:
        prefix = f"stress_{variant}_"
        fields.extend(
            [
                f"{prefix}joined",
                f"{prefix}join_status",
                f"{prefix}pred_angle",
                f"{prefix}pred_distance",
                f"{prefix}angle_abs_error",
                f"{prefix}angle_rel_error",
                f"{prefix}distance_abs_error",
                f"{prefix}distance_rel_error",
                f"{prefix}joint_error_score",
                f"{prefix}heading_delta",
                f"{prefix}range_delta",
                f"{prefix}joint_delta",
                f"{prefix}heading_sensitive",
                f"{prefix}range_sensitive",
                f"{prefix}joint_sensitive",
            ]
        )
    fields.extend(["shared_baseline_stress_joined", "shared_join_status", "missing_surface_reason_codes", "join_key_source", "join_key_confidence"])
    return fields


def write_shared_surface(candidate_path, baseline_path, stress_paths, output_dir):
    out = ensure_output_dirs(output_dir)
    candidate_rows = read_csv_dicts(candidate_path)
    baseline_rows = read_csv_dicts(baseline_path)
    stress_surface_map = {name: read_csv_dicts(path) for name, path in stress_paths.items()}
    rows = build_shared_surface(candidate_rows, baseline_rows, stress_surface_map)
    fieldnames = shared_surface_fieldnames(list(stress_surface_map))
    target = out / "manifests" / "a_v3_1_shared_outcome_surface_manifest.csv"
    write_csv_dicts(target, rows, fieldnames)
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stress", action="append", default=[], help="variant=path")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    stress_paths = {}
    for item in args.stress:
        variant, path = item.split("=", 1)
        stress_paths[variant] = path
    write_shared_surface(args.candidate, args.baseline, stress_paths, args.output_dir)


if __name__ == "__main__":
    main()
