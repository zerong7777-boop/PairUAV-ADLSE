from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import (
    count_true,
    ensure_dirs,
    group_count,
    read_csv_dicts,
    safe_div,
    truthy,
    write_json,
)


def compute_overlap_counts(rows):
    baseline = count_true(rows, "baseline_joint_hard")
    stress = count_true(rows, "stress_joint_sensitive")
    overlap = sum(
        1
        for row in rows
        if truthy(row.get("baseline_joint_hard")) and truthy(row.get("stress_joint_sensitive"))
    )
    return {
        "baseline_joint_hard_count": baseline,
        "stress_joint_sensitive_count": stress,
        "baseline_stress_overlap_count": overlap,
        "overlap_ratio_vs_baseline": safe_div(overlap, baseline),
        "overlap_ratio_vs_stress": safe_div(overlap, stress),
    }


def _joined(row, field):
    value = str(row.get(field, "")).strip().lower()
    return value in {"joined", "true", "1", "yes", "ok"}


def compute_shared_join_mask_counts(rows):
    shared = [
        row
        for row in rows
        if _joined(row, "full_dev_join_status") and _joined(row, "stress_join_status")
    ]
    overlap = sum(
        1
        for row in shared
        if truthy(row.get("baseline_joint_hard")) and truthy(row.get("stress_joint_sensitive"))
    )
    return {
        "shared_join_mask_count": len(shared),
        "shared_join_overlap_count": overlap,
        "shared_join_overlap_ratio": safe_div(overlap, len(shared)),
    }


def compute_heading_range_consistency(rows):
    return {
        "heading_hard_count": count_true(rows, "baseline_heading_hard"),
        "range_hard_count": count_true(rows, "baseline_range_hard"),
        "heading_stress_sensitive_count": count_true(rows, "stress_heading_sensitive"),
        "range_stress_sensitive_count": count_true(rows, "stress_range_sensitive"),
        "evidence_sufficient_heading_hard_count": count_true(rows, "evidence_sufficient_heading_hard"),
        "evidence_sufficient_range_hard_count": count_true(rows, "evidence_sufficient_range_hard"),
        "evidence_sufficient_joint_hard_count": count_true(rows, "evidence_sufficient_joint_hard"),
    }


def compute_target_group_distribution(rows):
    result = {}
    for field in ("target_key", "group_id", "scene_key", "split_key"):
        if rows and field in rows[0]:
            result[field] = group_count(rows, field)
    return result


def audit_outcome_surface_consistency(rows):
    overlap = compute_overlap_counts(rows)
    shared = compute_shared_join_mask_counts(rows)
    split = compute_heading_range_consistency(rows)
    total = len(rows)
    reasons = []
    verdict = "expected-axis-difference"

    if overlap["baseline_joint_hard_count"] and overlap["stress_joint_sensitive_count"]:
        if overlap["baseline_stress_overlap_count"] == 0:
            if shared["shared_join_mask_count"] < max(100, int(0.05 * total)):
                verdict = "join-bias"
                reasons.append("shared_join_mask_too_small_to_compare_surfaces")
            else:
                verdict = "unresolved-blocker"
                reasons.append("baseline_and_stress_outcomes_nonempty_but_zero_overlap")
        else:
            verdict = "expected-axis-difference"
            reasons.append("baseline_and_stress_overlap_exists_but_axes_remain_distinct")
    elif not overlap["baseline_joint_hard_count"] and not overlap["stress_joint_sensitive_count"]:
        verdict = "metric-mismatch"
        reasons.append("both_outcome_surfaces_empty")
    else:
        verdict = "threshold-artifact"
        reasons.append("only_one_outcome_surface_nonempty")

    if shared["shared_join_mask_count"] and shared["shared_join_overlap_count"] == 0:
        reasons.append("zero_overlap_persists_on_shared_join_mask")

    result = {
        "total_rows": total,
        **overlap,
        **shared,
        **split,
        "target_distribution_if_available": compute_target_group_distribution(rows),
        "verdict": verdict,
        "verdict_reasons": reasons,
    }
    return result


def _write_report(metrics, path):
    lines = [
        "# A-v3 Outcome Surface Consistency Audit",
        "",
        f"- total_rows: {metrics['total_rows']}",
        f"- baseline_joint_hard_count: {metrics['baseline_joint_hard_count']}",
        f"- stress_joint_sensitive_count: {metrics['stress_joint_sensitive_count']}",
        f"- baseline_stress_overlap_count: {metrics['baseline_stress_overlap_count']}",
        f"- shared_join_mask_count: {metrics['shared_join_mask_count']}",
        f"- shared_join_overlap_count: {metrics['shared_join_overlap_count']}",
        f"- verdict: {metrics['verdict']}",
        "",
        "No training, finetuning, sampler, gate label, or threshold tuning was run.",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outcome_surface_consistency_audit(rows, output_dir):
    out = ensure_dirs(output_dir)
    metrics = audit_outcome_surface_consistency(rows)
    write_json(out / "metrics" / "a_v3_outcome_surface_consistency_audit.json", metrics)
    _write_report(metrics, out / "reports" / "a_v3_outcome_surface_consistency_audit.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_csv_dicts(args.input)
    write_outcome_surface_consistency_audit(rows, args.output_dir)


if __name__ == "__main__":
    main()
