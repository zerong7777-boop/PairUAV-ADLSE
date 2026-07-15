from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import ensure_dirs, read_csv_dicts, to_float, truthy, write_csv_dicts


FORBIDDEN_COLUMNS = {"gate_label", "train_label", "sampler_weight", "oversample", "loss_weight"}


def slice_definitions():
    return [
        {
            "name": "semantic_geometric_conflict",
            "required_true": ["semantic_geometric_conflict"],
            "reason": "semantic/geometric conflict diagnostic",
        },
        {
            "name": "heading_hard_semantic_geometric_conflict",
            "required_true": ["evidence_sufficient_heading_hard", "semantic_geometric_conflict"],
            "reason": "heading hard with correspondence conflict",
        },
        {
            "name": "heading_hard_multi_modal_ambiguous",
            "required_true": ["evidence_sufficient_heading_hard", "multi_modal_ambiguous"],
            "reason": "heading hard with multi-modal ambiguity",
        },
        {
            "name": "heading_hard_stress_sensitive_ambiguous",
            "required_true": ["evidence_sufficient_heading_hard", "stress_sensitive_ambiguous"],
            "reason": "heading hard with stress-sensitive ambiguity",
        },
        {
            "name": "hard_ambiguity_overlap",
            "any_true": ["evidence_sufficient_heading_hard", "evidence_sufficient_range_hard", "evidence_sufficient_joint_hard"],
            "any_ambiguity": ["multi_modal_ambiguous", "semantic_geometric_conflict", "stress_sensitive_ambiguous", "tail_error_unreliable"],
            "reason": "any hard evidence overlapping ambiguity/failure subtype",
        },
        {
            "name": "control_candidate_low_stress_low_error",
            "required_true": ["control_candidate"],
            "max_fields": {"stress_sensitivity_score": 0.3, "baseline_error_score": 0.3},
            "reason": "candidate preservation control under low observed error/stress",
        },
    ]


def row_matches_slice(row, definition):
    for field in definition.get("required_true", []):
        if not truthy(row.get(field)):
            return False
    if "any_true" in definition and not any(truthy(row.get(field)) for field in definition["any_true"]):
        return False
    if "any_ambiguity" in definition and not any(truthy(row.get(field)) for field in definition["any_ambiguity"]):
        return False
    for field, maximum in definition.get("max_fields", {}).items():
        if to_float(row.get(field)) > maximum:
            return False
    return True


def build_b_diagnostic_slices(rows):
    output = []
    for row in rows:
        for definition in slice_definitions():
            if not row_matches_slice(row, definition):
                continue
            output.append(
                {
                    "canonical_pair_id": row.get("canonical_pair_id") or row.get("pair_id") or row.get("pair_key") or "",
                    "slice_name": definition["name"],
                    "diagnostic_only": "true",
                    "reason_codes": definition["reason"],
                    "heading_hard": truthy(row.get("evidence_sufficient_heading_hard")),
                    "range_hard": truthy(row.get("evidence_sufficient_range_hard")),
                    "semantic_geometric_conflict": truthy(row.get("semantic_geometric_conflict")),
                    "multi_modal_ambiguous": truthy(row.get("multi_modal_ambiguous")),
                    "stress_sensitive_ambiguous": truthy(row.get("stress_sensitive_ambiguous")),
                    "tail_error_unreliable": truthy(row.get("tail_error_unreliable")),
                    "control_candidate": truthy(row.get("control_candidate")),
                }
            )
    return output


def write_b_diagnostic_slices(rows, output_dir):
    out = ensure_dirs(output_dir)
    slices = build_b_diagnostic_slices(rows)
    fields = [
        "canonical_pair_id",
        "slice_name",
        "diagnostic_only",
        "reason_codes",
        "heading_hard",
        "range_hard",
        "semantic_geometric_conflict",
        "multi_modal_ambiguous",
        "stress_sensitive_ambiguous",
        "tail_error_unreliable",
        "control_candidate",
    ]
    if FORBIDDEN_COLUMNS.intersection(fields):
        raise RuntimeError("forbidden training-policy column requested")
    write_csv_dicts(out / "tables" / "a_v3_b_offline_diagnostic_slices.csv", slices, fields)
    counts = {}
    for row in slices:
        counts[row["slice_name"]] = counts.get(row["slice_name"], 0) + 1
    lines = ["# A-v3 B Offline Diagnostic Slices", ""]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "All slices are diagnostic-only. Forbidden training-policy columns are not emitted."]
    report_path = out / "reports" / "a_v3_b_offline_diagnostic_slices.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"slice_count": len(slices), "slice_counts": counts, "forbidden_columns_present": False}


def write_training_policy_readiness_verdict(output_dir, audit_metrics):
    out = ensure_dirs(output_dir)
    outcome = audit_metrics.get("outcome", {})
    join = audit_metrics.get("join_bias", {})
    predictability = audit_metrics.get("predictability", {})
    control = audit_metrics.get("stable_control", {})

    if not outcome or not join or not predictability or not control:
        verdict = "training-policy-no-go"
        reason = "required_audit_metric_missing"
    elif outcome.get("verdict") == "unresolved-blocker":
        verdict = "training-policy-blocked-by-outcome-surface"
        reason = "baseline_vs_stress_zero_overlap_unresolved"
    elif join.get("verdict") == "join-bias-blocks-training-policy":
        verdict = "training-policy-blocked-by-join-bias"
        reason = "join_bias_blocks_training_policy"
    elif int(predictability.get("useful_pair_count", 0)) <= 0:
        verdict = "training-policy-blocked-by-weak-predictability"
        reason = "no_useful_layer1_to_layer2_predictability_pair"
    elif control.get("verdict") == "control-preservation-no-go":
        verdict = "training-policy-no-go"
        reason = "stable_control_no_go"
    else:
        verdict = "training-policy-spec-allowed-for-shadow-only"
        reason = "audits_allow_shadow_only_spec_not_training"

    text = "\n".join(
        [
            "# A-v3 Training Policy Readiness Verdict",
            "",
            f"verdict: `{verdict}`",
            f"reason: `{reason}`",
            "",
            "This is not a training policy. No training, finetuning, sampler, gate label, or threshold tuning was run.",
        ]
    )
    report_path = out / "reports" / "a_v3_training_policy_readiness_verdict.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text + "\n", encoding="utf-8")
    return {"verdict": verdict, "reason": reason}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_b_diagnostic_slices(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
