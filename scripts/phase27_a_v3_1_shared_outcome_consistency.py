from pathlib import Path

from scripts.phase27_a_v3_1_coverage_bias_audit import infer_stress_variants
from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, group_count, read_csv_dicts, safe_div, truthy, write_json


def shared_rows(rows):
    return [row for row in rows if truthy(row.get("shared_baseline_stress_joined"))]


def compute_shared_outcome_consistency(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    srows = shared_rows(rows)
    result = {
        "total_rows": len(rows),
        "shared_rows": len(srows),
        "baseline_heading_hard_count": sum(1 for row in srows if truthy(row.get("baseline_heading_hard"))),
        "baseline_range_hard_count": sum(1 for row in srows if truthy(row.get("baseline_range_hard"))),
        "baseline_joint_hard_count": sum(1 for row in srows if truthy(row.get("baseline_joint_hard"))),
        "target_distribution_on_shared": group_count(srows, "target_key"),
        "group_distribution_on_shared": group_count(srows, "group_id"),
    }
    for variant in stress_variants:
        joint_field = f"stress_{variant}_joint_sensitive"
        heading_field = f"stress_{variant}_heading_sensitive"
        range_field = f"stress_{variant}_range_sensitive"
        joint_count = sum(1 for row in srows if truthy(row.get(joint_field)))
        overlap = sum(1 for row in srows if truthy(row.get("baseline_joint_hard")) and truthy(row.get(joint_field)))
        result[f"stress_{variant}_heading_sensitive_count"] = sum(1 for row in srows if truthy(row.get(heading_field)))
        result[f"stress_{variant}_range_sensitive_count"] = sum(1 for row in srows if truthy(row.get(range_field)))
        result[f"stress_{variant}_joint_sensitive_count"] = joint_count
        result[f"baseline_joint_hard_stress_{variant}_joint_overlap_count"] = overlap
        result[f"overlap_ratio_vs_baseline_for_{variant}"] = safe_div(overlap, result["baseline_joint_hard_count"])
        result[f"overlap_ratio_vs_stress_for_{variant}"] = safe_div(overlap, joint_count)
    return result


def compute_outcome_by_target_group(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    groups = {}
    for row in shared_rows(rows):
        key = (row.get("target_key", "") or "missing", row.get("group_id", "") or "missing")
        entry = groups.setdefault(key, {"target_key": key[0], "group_id": key[1], "shared_rows": 0, "baseline_joint_hard": 0})
        entry["shared_rows"] += 1
        entry["baseline_joint_hard"] += int(truthy(row.get("baseline_joint_hard")))
        for variant in stress_variants:
            field = f"stress_{variant}_joint_sensitive"
            entry[field] = entry.get(field, 0) + int(truthy(row.get(field)))
    return list(groups.values())


def _write_report(metrics, path):
    lines = ["# A-v3.1 Shared Outcome Consistency", ""]
    for key, value in metrics.items():
        if not isinstance(value, dict):
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Validation-only recompute on shared rows.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shared_outcome_consistency(rows, output_dir):
    out = ensure_output_dirs(output_dir)
    metrics = compute_shared_outcome_consistency(rows)
    write_json(out / "metrics" / "a_v3_1_shared_outcome_consistency_metrics.json", metrics)
    _write_report(metrics, out / "reports" / "a_v3_1_shared_outcome_consistency_report.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_shared_outcome_consistency(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
