"""Write the A-v3.2a canonical pair identity contract."""
from pathlib import Path

from scripts.phase27_a_v3_2a_identity_common import ensure_dirs


def contract_field_table():
    rows = [
        ("canonical_pair_id", "promotion_key", "conditional", "Allowed only when unique within each artifact and semantically consistent across artifacts.", "normalize_token"),
        ("source_image_key", "promotion_key_candidate", "yes", "Candidate-side image identity component.", "normalize_image_key for path-normalized strategy"),
        ("target_image_key", "promotion_key_candidate", "yes", "Candidate-side image identity component.", "normalize_image_key for path-normalized strategy"),
        ("source_image_a", "promotion_key_candidate", "yes", "Surface-side source image identity component.", "normalize_image_key for path-normalized strategy"),
        ("source_image_b", "promotion_key_candidate", "yes", "Surface-side target image identity component.", "normalize_image_key for path-normalized strategy"),
        ("pair_key", "promotion_key_candidate", "yes", "Candidate-side pair key; valid only with source/target image keys.", "normalize_token"),
        ("source_pair_key", "promotion_key_candidate", "yes", "Surface-side pair key; valid only with source/target image keys.", "normalize_token"),
        ("pair_id", "diagnostic_key", "no", "May be local to manifest namespace.", "normalize_token"),
        ("source_row_index", "forbidden_key", "no", "Row index is not stable across artifacts.", "normalize_token"),
        ("source_json_id", "diagnostic_key", "no", "Useful for namespace audit, insufficient alone.", "normalize_token"),
        ("json_path", "diagnostic_key", "no", "Path namespace may differ across artifacts.", "normalize_token"),
        ("source_json_path", "diagnostic_key", "no", "Surface path namespace may differ from candidate manifest.", "normalize_token"),
        ("target_key", "metadata_only", "no", "Bias stratification field, not pair identity.", "normalize_token"),
        ("group_id", "metadata_only", "no", "Bias stratification field, not pair identity.", "normalize_token"),
        ("source_group_id", "metadata_only", "no", "Surface group metadata.", "normalize_token"),
        ("scene_key", "metadata_only", "no", "Bias stratification field.", "normalize_token"),
        ("split_key", "metadata_only", "no", "Split provenance, not pair identity.", "normalize_token"),
        ("source_split", "metadata_only", "no", "Split provenance.", "normalize_token"),
        ("baseline_surface_source", "metadata_only", "no", "Surface provenance.", "normalize_token"),
        ("stress_baseline_surface_source", "metadata_only", "no", "Stress surface provenance.", "normalize_token"),
    ]
    return [
        {"field": f, "role": r, "promotion_allowed": p, "reason": reason, "normalization": norm}
        for f, r, p, reason, norm in rows
    ]


def contract_policy_sections():
    return {
        "ordered_pair": "Default promotion identity is ordered: source and target roles are preserved.",
        "direction_invariant": "Direction-invariant keys are diagnostic only unless manually justified.",
        "duplicates": "Duplicates are blocked and exported; no silent first-match or silent deduplication is allowed.",
        "missing_keys": "Rows missing promotion keys remain analysis-only and must receive reason codes.",
        "fuzzy_matching": "Fuzzy joins are forbidden for promotion and may only be reported as diagnostic candidates with false-match risk.",
        "no_training": "This contract authorizes no training, finetuning, sample weighting, curriculum, sampler, or B/C gate.",
    }


def write_identity_contract(output_dir):
    out = ensure_dirs(output_dir)
    table = contract_field_table()
    policies = contract_policy_sections()
    lines = [
        "# Canonical Pair Identity Contract",
        "",
        "Status: validation-only contract for A-v3.2a.",
        "",
        "## Field Roles",
        "",
        "| field | role | promotion_allowed | normalization | reason |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['field']} | {row['role']} | {row['promotion_allowed']} | {row['normalization']} | {row['reason']} |")
    lines += ["", "## Policies", ""]
    for key, text in policies.items():
        lines.append(f"- {key}: {text}")
    lines += ["", "No training, finetuning, sample weighting, curriculum, threshold tuning, fuzzy-overlap construction, silent deduplication, sampler, or B/C gate is allowed."]
    path = out / "contract" / "canonical_pair_identity_contract.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "fields": table, "policies": policies}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_identity_contract(args.output_dir)


if __name__ == "__main__":
    main()
