"""Layer 1 non-leaking candidate discovery for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

from pathlib import Path

from scripts.phase27_a_taxonomy_redesign_v3_io import read_csv_dicts, to_bool, to_float, write_csv_dicts
from scripts.phase27_a_taxonomy_redesign_v3_schema import IDENTITY_FIELDS, LAYER1_CANDIDATE_FIELDS


def _score(row: dict, field: str, missing: list[str]) -> float | None:
    if field not in row:
        missing.append(f"missing_source_field:{field}")
        return None
    return to_float(row.get(field))


def derive_layer1_candidates(row: dict) -> dict:
    missing = []
    evidence = _score(row, "evidence_sufficiency_score", missing)
    heading = _score(row, "heading_observability_score", missing)
    range_score = _score(row, "range_observability_score", missing)
    conflict = _score(row, "semantic_geometric_conflict_score", missing)
    match = _score(row, "match_sufficiency_score", missing)
    ambiguity_tail = _score(row, "ambiguity_tail_risk_score", missing)
    control = _score(row, "control_stability_score", missing)
    layout_risk = _score(row, "layout_scale_risk_score", missing)
    low_observable_flag = to_bool(row.get("low_observable_flag"))

    evidence_sufficient = (evidence or 0.0) >= 0.60 or (match or 0.0) >= 0.60
    heading_observable = (heading or 0.0) >= 0.50
    range_observable = (range_score or 0.0) >= 0.50
    low_observable = low_observable_flag or (evidence is not None and evidence < 0.20)
    conflict_candidate = (conflict or 0.0) >= 0.55
    local_alignment = conflict_candidate and (evidence or 0.0) >= 0.50
    ambiguity = (ambiguity_tail or 0.0) >= 0.60 or conflict_candidate
    ordinary = (evidence or 0.0) >= 0.45 and not conflict_candidate and not low_observable
    control_candidate = ordinary and (control or 0.0) >= 0.70 and (layout_risk or 0.0) < 0.70

    reason = []
    if evidence_sufficient:
        reason.append("evidence_or_match_sufficient")
    if conflict_candidate:
        reason.append("semantic_geometric_conflict")
    if local_alignment:
        reason.append("local_alignment_needed")
    if low_observable:
        reason.append("low_observable")
    if control_candidate:
        reason.append("stable_control_like")

    out = {
        "evidence_sufficient_candidate": evidence_sufficient,
        "heading_observable_candidate": heading_observable,
        "range_observable_candidate": range_observable,
        "low_observable_candidate": low_observable,
        "semantic_geometric_conflict_candidate": conflict_candidate,
        "local_alignment_needed_candidate": local_alignment,
        "ambiguity_candidate": ambiguity,
        "ordinary_candidate": ordinary,
        "control_candidate": control_candidate,
        "candidate_reason_codes": "|".join(reason),
        "missing_candidate_source_fields": "|".join(sorted(set(missing))),
    }
    out.update(derive_ambiguity_subtypes_from_candidates({**row, **out}))
    return out


def derive_ambiguity_subtypes_from_candidates(row: dict) -> dict:
    conflict = to_bool(row.get("semantic_geometric_conflict_candidate"))
    low = to_bool(row.get("low_observable_candidate"))
    ambiguity_tail = to_float(row.get("ambiguity_tail_risk_score")) or 0.0
    stress = to_float(row.get("stress_sensitivity_score")) or 0.0
    tail_outlier = to_bool(row.get("tail_outlier_flag"))
    multi_modal = ambiguity_tail >= 0.75 and not low
    out = {
        "low_evidence_ambiguous": low,
        "multi_modal_ambiguous": multi_modal,
        "semantic_geometric_conflict": conflict,
        "stress_sensitive_ambiguous": stress >= 0.65,
        "tail_error_unreliable": tail_outlier or ambiguity_tail >= 0.90,
    }
    reasons = [name for name, value in out.items() if value]
    out["ambiguity_reason_codes"] = "|".join(reasons)
    return out


def build_candidate_manifest(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        out = {field: row.get(field, "") for field in IDENTITY_FIELDS if field in row}
        out["canonical_pair_id"] = row.get("canonical_pair_id", "")
        out.update(derive_layer1_candidates(row))
        output.append(out)
    return output


def write_candidate_manifest(input_paths: dict, output_dir: str | Path) -> list[dict]:
    rows = read_csv_dicts(input_paths["candidate_manifest"])
    manifest = build_candidate_manifest(rows)
    fieldnames = list(dict.fromkeys(["canonical_pair_id"] + IDENTITY_FIELDS + LAYER1_CANDIDATE_FIELDS + [
        "low_evidence_ambiguous", "multi_modal_ambiguous", "semantic_geometric_conflict",
        "stress_sensitive_ambiguous", "tail_error_unreliable",
        "candidate_reason_codes", "ambiguity_reason_codes", "missing_candidate_source_fields",
    ]))
    write_csv_dicts(Path(output_dir) / "manifests" / "non_leaking_candidate_manifest.csv", manifest, fieldnames)
    return manifest
