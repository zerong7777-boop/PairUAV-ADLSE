"""Coverage and join-bias audit for A-v3.1 shared surfaces."""
from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, read_csv_dicts, safe_div, truthy, write_csv_dicts, write_json


def infer_stress_variants(rows):
    if not rows:
        return ["main"]
    variants = []
    for field in rows[0]:
        if field.startswith("stress_") and field.endswith("_joined"):
            variants.append(field[len("stress_") : -len("_joined")])
    return variants or ["main"]


def compute_coverage_metrics(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    total = len(rows)
    baseline_joined = sum(1 for row in rows if truthy(row.get("baseline_joined")))
    shared = sum(1 for row in rows if truthy(row.get("shared_baseline_stress_joined")))
    duplicate = sum(1 for row in rows if row.get("shared_join_status") == "duplicate_blocked")
    missing_baseline_only = sum(1 for row in rows if row.get("shared_join_status") == "missing_baseline")
    missing_stress_only = sum(1 for row in rows if row.get("shared_join_status") == "missing_stress")
    missing_both = sum(1 for row in rows if row.get("shared_join_status") == "missing_both")
    stress_counts = {variant: sum(1 for row in rows if truthy(row.get(f"stress_{variant}_joined"))) for variant in stress_variants}
    metrics = {
        "total_rows": total,
        "baseline_joined_count": baseline_joined,
        "stress_joined_count_by_variant": stress_counts,
        "shared_joined_count": shared,
        "shared_coverage_ratio": safe_div(shared, total),
        "missing_baseline_only_count": missing_baseline_only,
        "missing_stress_only_count": missing_stress_only,
        "missing_both_count": missing_both,
        "duplicate_blocked_count": duplicate,
    }
    metrics["target_bias_max_abs_diff"] = _max_join_bias(rows, "target_key")
    metrics["group_bias_max_abs_diff"] = _max_join_bias(rows, "group_id")
    metrics["verdict"] = coverage_verdict(metrics)
    return metrics


def _max_join_bias(rows, field):
    joined = [row for row in rows if truthy(row.get("shared_baseline_stress_joined"))]
    unjoined = [row for row in rows if not truthy(row.get("shared_baseline_stress_joined"))]
    values = set([row.get(field, "") or "missing" for row in joined + unjoined])
    max_diff = 0.0
    for value in values:
        joined_rate = safe_div(sum(1 for row in joined if (row.get(field, "") or "missing") == value), len(joined))
        unjoined_rate = safe_div(sum(1 for row in unjoined if (row.get(field, "") or "missing") == value), len(unjoined))
        max_diff = max(max_diff, abs(joined_rate - unjoined_rate))
    return max_diff


def coverage_verdict(metrics):
    if metrics["duplicate_blocked_count"] > 0 and metrics["shared_joined_count"] == 0:
        return "shared-surface-blocked-identity-join"
    if metrics["shared_joined_count"] == 0:
        return "shared-surface-blocked-zero-coverage"
    if metrics["shared_coverage_ratio"] < 0.30:
        return "shared-surface-analysis-only"
    if max(metrics.get("target_bias_max_abs_diff", 0.0), metrics.get("group_bias_max_abs_diff", 0.0)) > 0.25:
        return "shared-surface-blocked-bias"
    return "shared-surface-pass"


def compute_join_bias_by_target_group(rows):
    groups = {}
    for row in rows:
        key = (row.get("target_key", "") or "missing", row.get("group_id", "") or "missing")
        entry = groups.setdefault(key, {"target_key": key[0], "group_id": key[1], "total": 0, "shared_joined": 0, "baseline_joined": 0, "stress_any_joined": 0})
        entry["total"] += 1
        entry["shared_joined"] += int(truthy(row.get("shared_baseline_stress_joined")))
        entry["baseline_joined"] += int(truthy(row.get("baseline_joined")))
        entry["stress_any_joined"] += int(any(truthy(v) for k, v in row.items() if k.startswith("stress_") and k.endswith("_joined")))
    for entry in groups.values():
        entry["shared_coverage_ratio"] = safe_div(entry["shared_joined"], entry["total"])
    return list(groups.values())


def _write_report(metrics, path):
    lines = ["# A-v3.1 Shared Surface Coverage And Join Bias", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Validation-only. No training, sampler, gate label, or threshold tuning was run.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_coverage_bias_audit(rows, output_dir):
    out = ensure_output_dirs(output_dir)
    metrics = compute_coverage_metrics(rows)
    write_json(out / "metrics" / "a_v3_1_shared_surface_coverage_metrics.json", metrics)
    write_csv_dicts(out / "tables" / "a_v3_1_join_bias_by_target_group.csv", compute_join_bias_by_target_group(rows), ["target_key", "group_id", "total", "shared_joined", "baseline_joined", "stress_any_joined", "shared_coverage_ratio"])
    _write_report(metrics, out / "reports" / "a_v3_1_shared_surface_coverage_report.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_coverage_bias_audit(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
