"""Layer 3 multi-label readiness verdicts for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

from pathlib import Path

from scripts.phase27_a_taxonomy_redesign_v3_io import read_csv_dicts, to_bool, write_csv_dicts
from scripts.phase27_a_taxonomy_redesign_v3_schema import LAYER3_READINESS_FIELDS


def derive_readiness_verdicts(row: dict) -> dict:
    low = to_bool(row.get("low_observable_candidate"))
    evidence = to_bool(row.get("evidence_sufficient_candidate"))
    ordinary = to_bool(row.get("ordinary_candidate"))
    control = to_bool(row.get("control_candidate"))
    conflict = to_bool(row.get("semantic_geometric_conflict_candidate")) or to_bool(row.get("semantic_geometric_conflict"))
    heading_hard = to_bool(row.get("baseline_heading_hard"))
    range_hard = to_bool(row.get("baseline_range_hard"))
    stress = to_bool(row.get("stress_joint_sensitive"))
    multi_modal = to_bool(row.get("multi_modal_ambiguous"))
    validation_status = row.get("validation_status", "")
    missing_validation = validation_status == "unknown_due_to_missing_join"

    out = {field: False for field in LAYER3_READINESS_FIELDS}
    reasons = []
    if low:
        out["QUARANTINE_LOW_OBSERVABLE"] = True
        out["NOT_READY"] = True
        reasons.append("low_observable_quarantine")
    if ordinary and control and not heading_hard and not range_hard and not stress and not low:
        out["READY_CONTROL_PRESERVATION"] = True
        reasons.append("ordinary_validated_control")
    if conflict and heading_hard and not low:
        out["READY_CORRESPONDENCE_DIAGNOSTIC"] = True
        out["ANALYSIS_ONLY"] = True
        reasons.append("conflict_heading_hard_diagnostic")
    if evidence and heading_hard and not low and not missing_validation:
        out["READY_HEADING_HARD_TRAINING"] = True
        reasons.append("heading_hard_validation_readiness")
    if evidence and range_hard and not low and not missing_validation:
        out["READY_RANGE_HARD_TRAINING"] = True
        reasons.append("range_hard_validation_readiness")
    if multi_modal:
        out["READY_MULTI_HYPOTHESIS"] = True
        out["ANALYSIS_ONLY"] = True
        reasons.append("multi_modal_not_single_point")
    if missing_validation:
        out["ANALYSIS_ONLY"] = True
        if not any(out[field] for field in LAYER3_READINESS_FIELDS if field != "ANALYSIS_ONLY"):
            out["NOT_READY"] = True
        reasons.append("missing_validation")
    if not any(out.values()):
        out["NOT_READY"] = True
        reasons.append("no_readiness_condition")
    out["verdict_reason_codes"] = "|".join(reasons)
    return out


def build_readiness_manifest(candidate_rows: list[dict], outcome_rows: list[dict]) -> list[dict]:
    outcome_by_id = {row.get("canonical_pair_id", ""): row for row in outcome_rows}
    output = []
    for row in candidate_rows:
        merged = dict(row)
        merged.update(outcome_by_id.get(row.get("canonical_pair_id", ""), {}))
        merged.update(derive_readiness_verdicts(merged))
        output.append(merged)
    return output


def write_readiness_manifest(input_paths: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    candidate_rows = read_csv_dicts(output_dir / "manifests" / "non_leaking_candidate_manifest.csv")
    outcome_rows = read_csv_dicts(output_dir / "manifests" / "validation_only_outcome_attribution_manifest.csv")
    manifest = build_readiness_manifest(candidate_rows, outcome_rows)
    fieldnames = list(dict.fromkeys(list(manifest[0].keys()) if manifest else []))
    write_csv_dicts(output_dir / "manifests" / "training_readiness_verdict_manifest.csv", manifest, fieldnames)
    return manifest
