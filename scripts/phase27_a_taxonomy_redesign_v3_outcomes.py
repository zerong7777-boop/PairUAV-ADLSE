"""Layer 2 validation-only outcome attribution for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

from pathlib import Path

from scripts.phase27_a_taxonomy_redesign_v3_io import left_join_by_pair_id, read_csv_dicts, to_bool, to_float, write_csv_dicts
from scripts.phase27_a_taxonomy_redesign_v3_schema import HEADING_RANGE_HARD_FIELDS, LAYER2_OUTCOME_FIELDS


def derive_heading_range_hard(row: dict) -> dict:
    heading_error = max(to_float(row.get("heading_error_score")) or 0.0, to_float(row.get("full_dev_baseline_angle_rel_error")) or 0.0)
    range_error = max(to_float(row.get("range_error_score")) or 0.0, to_float(row.get("full_dev_baseline_distance_rel_error")) or 0.0)
    evidence = to_bool(row.get("evidence_sufficient_candidate"))
    heading_hard = evidence and heading_error >= 0.75
    range_hard = evidence and range_error >= 0.75
    return {
        "evidence_sufficient_heading_hard": heading_hard,
        "evidence_sufficient_range_hard": range_hard,
        "evidence_sufficient_joint_hard": heading_hard or range_hard,
    }


def derive_stress_sensitivity(row: dict) -> dict:
    stress = to_float(row.get("stress_sensitivity_score"))
    heading_error = max(to_float(row.get("heading_error_score")) or 0.0, to_float(row.get("full_dev_baseline_angle_rel_error")) or 0.0)
    range_error = max(to_float(row.get("range_error_score")) or 0.0, to_float(row.get("full_dev_baseline_distance_rel_error")) or 0.0)
    if stress is None:
        base = max(to_float(row.get("full_dev_baseline_angle_rel_error")) or 0.0, to_float(row.get("full_dev_baseline_distance_rel_error")) or 0.0)
        stressed = max(to_float(row.get("stress_baseline_angle_rel_error")) or 0.0, to_float(row.get("stress_baseline_distance_rel_error")) or 0.0)
        stress = max(0.0, stressed - base)
    return {
        "heading_stress_sensitive": stress >= 0.65 and heading_error >= range_error,
        "range_stress_sensitive": stress >= 0.65 and range_error > heading_error,
    }


def derive_layer2_outcomes(row: dict) -> dict:
    split = derive_heading_range_hard(row)
    stress_split = derive_stress_sensitivity(row)
    baseline_error = to_float(row.get("baseline_error_score")) or 0.0
    stress = to_float(row.get("stress_sensitivity_score"))
    if stress is None:
        base = max(to_float(row.get("full_dev_baseline_angle_rel_error")) or 0.0, to_float(row.get("full_dev_baseline_distance_rel_error")) or 0.0)
        stressed = max(to_float(row.get("stress_baseline_angle_rel_error")) or 0.0, to_float(row.get("stress_baseline_distance_rel_error")) or 0.0)
        stress = max(0.0, stressed - base)
    checkpoint = to_float(row.get("checkpoint_disagreement_score")) or 0.0
    tail = to_bool(row.get("tail_outlier_flag"))
    full_joined = to_bool(row.get("full_dev_joined")) or row.get("full_dev_join_status") == "joined"
    stress_joined = to_bool(row.get("stress_joined")) or row.get("stress_join_status") == "joined"
    if full_joined and stress_joined:
        validation_status = "validated_ready"
    elif not full_joined and not stress_joined:
        validation_status = "unknown_due_to_missing_join"
    else:
        validation_status = "candidate_only_unvalidated"
    out = {
        "baseline_heading_hard": split["evidence_sufficient_heading_hard"],
        "baseline_range_hard": split["evidence_sufficient_range_hard"],
        "baseline_joint_hard": split["evidence_sufficient_joint_hard"] or baseline_error >= 0.80,
        "stress_heading_sensitive": stress_split["heading_stress_sensitive"],
        "stress_range_sensitive": stress_split["range_stress_sensitive"],
        "stress_joint_sensitive": stress >= 0.65,
        "tail_error_high": tail,
        "checkpoint_disagreement_high": checkpoint >= 0.65,
        "augmentation_instability_high": stress >= 0.80,
        **split,
        **stress_split,
        "full_dev_join_status": "joined" if full_joined else row.get("full_dev_join_status", "unjoined"),
        "stress_join_status": "joined" if stress_joined else row.get("stress_join_status", "unjoined"),
        "validation_status": validation_status,
    }
    reasons = [name for name in LAYER2_OUTCOME_FIELDS + HEADING_RANGE_HARD_FIELDS if out.get(name)]
    out["outcome_reason_codes"] = "|".join(reasons)
    return out


def build_outcome_manifest(candidate_rows: list[dict], full_dev_rows: list[dict], stress_rows: list[dict]) -> list[dict]:
    rows = left_join_by_pair_id(candidate_rows, full_dev_rows, "full_dev")
    if stress_rows:
        rows = left_join_by_pair_id(rows, stress_rows, "stress")
    output = []
    for row in rows:
        merged = dict(row)
        merged.update(derive_layer2_outcomes(merged))
        output.append(merged)
    return output


def write_outcome_manifest(input_paths: dict, output_dir: str | Path) -> list[dict]:
    candidate_rows = read_csv_dicts(Path(output_dir) / "manifests" / "non_leaking_candidate_manifest.csv")
    full_rows = read_csv_dicts(input_paths["full_dev_surface"]) if input_paths.get("full_dev_surface") else []
    stress_rows = []
    for path in input_paths.get("stress_surfaces", [])[:1]:
        stress_rows.extend(read_csv_dicts(path))
    manifest = build_outcome_manifest(candidate_rows, full_rows, stress_rows)
    fieldnames = list(dict.fromkeys(list(manifest[0].keys()) if manifest else []))
    write_csv_dicts(Path(output_dir) / "manifests" / "validation_only_outcome_attribution_manifest.csv", manifest, fieldnames)
    return manifest
