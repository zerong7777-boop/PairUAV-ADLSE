from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import (
    ensure_dirs,
    group_count,
    quantiles,
    read_csv_dicts,
    safe_div,
    to_float,
    truthy,
    write_json,
)


ERROR_FIELDS = [
    "baseline_error_score",
    "heading_error_score",
    "range_error_score",
    "stress_sensitivity_score",
]
CANDIDATE_FIELDS = [
    "evidence_sufficient_candidate",
    "heading_observable_candidate",
    "range_observable_candidate",
    "semantic_geometric_conflict_candidate",
    "local_alignment_needed_candidate",
    "ambiguity_candidate",
    "ordinary_candidate",
    "control_candidate",
]
HARD_AMBIGUITY_FIELDS = [
    "evidence_sufficient_heading_hard",
    "evidence_sufficient_range_hard",
    "evidence_sufficient_joint_hard",
    "multi_modal_ambiguous",
    "semantic_geometric_conflict",
    "stress_sensitive_ambiguous",
    "tail_error_unreliable",
]


def select_control_rows(rows):
    return [
        row
        for row in rows
        if truthy(row.get("control_candidate")) or truthy(row.get("READY_CONTROL_PRESERVATION"))
    ]


def error_summary(rows, fields):
    result = {}
    for field in fields:
        values = [to_float(row.get(field)) for row in rows if row.get(field) not in (None, "")]
        if values:
            values = sorted(values)
            result[field] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                **quantiles(values),
            }
        else:
            result[field] = {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return result


def compute_stable_control_stress_audit(rows):
    controls = select_control_rows(rows)
    control_count = len(controls)
    tail = sum(1 for row in controls if truthy(row.get("tail_error_high")) or truthy(row.get("tail_error_unreliable")))
    conflict = sum(1 for row in controls if truthy(row.get("semantic_geometric_conflict")) or truthy(row.get("semantic_geometric_conflict_candidate")))
    ambiguity = sum(1 for row in controls if truthy(row.get("ambiguity_candidate")) or truthy(row.get("multi_modal_ambiguous")) or truthy(row.get("stress_sensitive_ambiguous")))
    summaries = error_summary(controls, ERROR_FIELDS)
    stress_p95 = summaries["stress_sensitivity_score"]["p95"]
    tail_rate = safe_div(tail, control_count)
    conflict_rate = safe_div(conflict, control_count)
    ambiguity_rate = safe_div(ambiguity, control_count)
    reasons = []
    if control_count == 0:
        verdict = "control-preservation-no-go"
        reasons.append("no_control_rows")
    elif stress_p95 > 0.65 or tail_rate > 0.05:
        verdict = "control-preservation-no-go"
        reasons.append("control_tail_or_stress_risk_high")
    elif stress_p95 > 0.3 or conflict_rate > 0.1 or ambiguity_rate > 0.1:
        verdict = "control-preservation-analysis-only"
        reasons.append("control_contamination_or_stress_requires_analysis_only")
    else:
        verdict = "control-preservation-safe"
        reasons.append("controls_low_error_low_stress_in_observed_surface")
    return {
        "control_count": control_count,
        "baseline_error_summary": summaries["baseline_error_score"],
        "heading_error_summary": summaries["heading_error_score"],
        "range_error_summary": summaries["range_error_score"],
        "stress_delta_summary": summaries["stress_sensitivity_score"],
        "tail_error_rate": tail_rate,
        "conflict_contamination_rate": conflict_rate,
        "ambiguity_contamination_rate": ambiguity_rate,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


def compute_join_status_distribution(rows):
    return {
        "full_dev_join_status": group_count(rows, "full_dev_join_status"),
        "stress_join_status": group_count(rows, "stress_join_status"),
        "shared_join_mask_count": sum(
            1
            for row in rows
            if str(row.get("full_dev_join_status", "")).lower() == "joined"
            and str(row.get("stress_join_status", "")).lower() == "joined"
        ),
    }


def _distribution_by_status(rows, status_field, fields):
    statuses = sorted(set(row.get(status_field, "") or "missing" for row in rows))
    result = {}
    for status in statuses:
        subset = [row for row in rows if (row.get(status_field, "") or "missing") == status]
        result[status] = {"count": len(subset)}
        for field in fields:
            if rows and field in rows[0]:
                result[status][field] = safe_div(sum(1 for row in subset if truthy(row.get(field))), len(subset))
    return result


def compute_candidate_distribution_by_join(rows):
    return {
        "full_dev": _distribution_by_status(rows, "full_dev_join_status", CANDIDATE_FIELDS),
        "stress": _distribution_by_status(rows, "stress_join_status", CANDIDATE_FIELDS),
    }


def compute_hard_ambiguity_distribution_by_join(rows):
    return {
        "full_dev": _distribution_by_status(rows, "full_dev_join_status", HARD_AMBIGUITY_FIELDS),
        "stress": _distribution_by_status(rows, "stress_join_status", HARD_AMBIGUITY_FIELDS),
    }


def compute_join_bias_extension_report(rows):
    status = compute_join_status_distribution(rows)
    full_joined = [
        row for row in rows if str(row.get("full_dev_join_status", "")).lower() == "joined"
    ]
    stress_joined = [
        row for row in rows if str(row.get("stress_join_status", "")).lower() == "joined"
    ]
    missing_any = sum(
        1
        for row in rows
        if str(row.get("full_dev_join_status", "")).lower() != "joined"
        or str(row.get("stress_join_status", "")).lower() != "joined"
    )
    missing_both = sum(
        1
        for row in rows
        if str(row.get("full_dev_join_status", "")).lower() != "joined"
        and str(row.get("stress_join_status", "")).lower() != "joined"
    )
    reasons = []
    if status["shared_join_mask_count"] == 0 and (full_joined or stress_joined):
        verdict = "join-bias-blocks-training-policy"
        reasons.append("full_dev_and_stress_validation_surfaces_have_zero_shared_join_mask")
    elif safe_div(missing_any, len(rows)) > 0.5:
        verdict = "join-bias-blocks-training-policy"
        reasons.append("majority_rows_missing_at_least_one_validation_join")
    elif status["shared_join_mask_count"] == 0:
        verdict = "join-bias-unresolved"
        reasons.append("no_shared_join_mask")
    else:
        verdict = "join-bias-acceptable-for-analysis"
        reasons.append("shared_join_mask_available_for_analysis_not_training_policy")
    return {
        "total_rows": len(rows),
        **status,
        "full_dev_joined_count": len(full_joined),
        "stress_joined_count": len(stress_joined),
        "missing_any_validation_join_count": missing_any,
        "missing_both_validation_join_count": missing_both,
        "candidate_distribution_by_full_dev_join": compute_candidate_distribution_by_join(rows)["full_dev"],
        "candidate_distribution_by_stress_join": compute_candidate_distribution_by_join(rows)["stress"],
        "hard_ambiguity_distribution_by_join": compute_hard_ambiguity_distribution_by_join(rows),
        "unknown_due_to_missing_join_count": missing_both,
        "target_distribution_by_full_dev_join": _distribution_by_status(rows, "full_dev_join_status", ["target_key", "group_id"]) if rows else {},
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


def _write_report(metrics, path, title):
    lines = [f"# {title}", ""]
    for key, value in metrics.items():
        if key in {"candidate_distribution_by_full_dev_join", "candidate_distribution_by_stress_join", "hard_ambiguity_distribution_by_join", "target_distribution_by_full_dev_join"}:
            continue
        lines.append(f"- {key}: {value}")
    lines += ["", "Validation-only audit. No training policy is created."]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stable_control_stress_audit(rows, output_dir):
    out = ensure_dirs(output_dir)
    metrics = compute_stable_control_stress_audit(rows)
    write_json(out / "metrics" / "a_v3_stable_control_stress_audit.json", metrics)
    _write_report(metrics, out / "reports" / "a_v3_stable_control_stress_audit.md", "A-v3 Stable Control Stress Audit")
    return metrics


def write_join_bias_extension_report(rows, output_dir):
    out = ensure_dirs(output_dir)
    metrics = compute_join_bias_extension_report(rows)
    write_json(out / "metrics" / "a_v3_join_bias_extension_metrics.json", metrics)
    _write_report(metrics, out / "reports" / "a_v3_join_bias_extension_report.md", "A-v3 Join Bias Extension Report")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_csv_dicts(args.input)
    write_stable_control_stress_audit(rows, args.output_dir)
    write_join_bias_extension_report(rows, args.output_dir)


if __name__ == "__main__":
    main()
