from pathlib import Path

from scripts.phase27_a_v3_1_coverage_bias_audit import infer_stress_variants
from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, quantiles, read_csv_dicts, safe_div, to_float, truthy, write_json
from scripts.phase27_a_v3_1_shared_outcome_consistency import shared_rows


def select_shared_control_rows(rows):
    return [row for row in shared_rows(rows) if truthy(row.get("control_candidate")) or truthy(row.get("READY_CONTROL_PRESERVATION"))]


def _summary(rows, field):
    values = [to_float(row.get(field)) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {"count": len(values), "mean": sum(values) / len(values), **quantiles(values)}


def compute_shared_stable_control_metrics(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    controls = select_shared_control_rows(rows)
    stress_delta_fields = [f"stress_{variant}_joint_delta" for variant in stress_variants]
    stress_values = []
    for row in controls:
        for field in stress_delta_fields:
            if row.get(field) not in (None, ""):
                stress_values.append(to_float(row.get(field)))
    stress_summary = {"count": len(stress_values), "mean": safe_div(sum(stress_values), len(stress_values)), **quantiles(stress_values)}
    metrics = {
        "shared_control_count": len(controls),
        "baseline_joint_error_summary": _summary(controls, "baseline_joint_error_score"),
        "baseline_heading_error_summary": _summary(controls, "baseline_angle_abs_error"),
        "baseline_range_error_summary": _summary(controls, "baseline_distance_rel_error"),
        "stress_joint_delta_summary": stress_summary,
        "tail_rate": safe_div(sum(1 for row in controls if truthy(row.get("tail_error_unreliable"))), len(controls)),
        "conflict_contamination_rate": safe_div(sum(1 for row in controls if truthy(row.get("semantic_geometric_conflict"))), len(controls)),
        "ambiguity_contamination_rate": safe_div(sum(1 for row in controls if truthy(row.get("multi_modal_ambiguous")) or truthy(row.get("stress_sensitive_ambiguous"))), len(controls)),
    }
    metrics["verdict"] = stable_control_verdict(metrics)
    return metrics


def stable_control_verdict(metrics):
    if metrics["shared_control_count"] == 0:
        return "control-anchor-not-validated"
    if metrics["tail_rate"] > 0.05 or metrics["stress_joint_delta_summary"]["p95"] > 0.65:
        return "control-anchor-analysis-only"
    if metrics["baseline_joint_error_summary"]["p95"] <= 0.30 and metrics["stress_joint_delta_summary"]["p95"] <= 0.30:
        return "control-anchor-shadow-candidate"
    return "control-anchor-analysis-only"


def _write_report(metrics, path):
    lines = ["# A-v3.1 Shared Stable-Control Audit", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("This does not authorize preservation training.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shared_stable_control_audit(rows, output_dir):
    out = ensure_output_dirs(output_dir)
    metrics = compute_shared_stable_control_metrics(rows)
    write_json(out / "metrics" / "a_v3_1_shared_stable_control_metrics.json", metrics)
    _write_report(metrics, out / "reports" / "a_v3_1_shared_stable_control_report.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_shared_stable_control_audit(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
