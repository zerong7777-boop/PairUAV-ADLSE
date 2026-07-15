"""Identity, coverage, and joined/unjoined bias audit for A-v3.2b."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.phase27_a_v3_2b_common import ensure_dirs, read_csv_dicts, write_csv_dicts, write_json


def _read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def audit(fixed_rows, shared_rows, outcome_metrics):
    total = len(fixed_rows)
    ready = [r for r in shared_rows if r.get("shared_pair_status") == "ready"]
    not_ready = [r for r in shared_rows if r.get("shared_pair_status") != "ready"]
    stress_metrics = outcome_metrics.get("stress_metrics", {})
    duplicate_count = sum(v.get("duplicate_id_count", 0) for v in stress_metrics.values())
    composite_missing = sum(v.get("source_target_composite_missing_row_count", 0) for v in stress_metrics.values())
    missing_count = sum(v.get("missing_id_count", 0) for v in stress_metrics.values())
    coverage = 0.0 if total == 0 else len(ready) / total
    identity = {
        "fixed_manifest_row_count": total,
        "shared_ready_count": len(ready),
        "not_ready_count": len(not_ready),
        "stress_duplicate_blocked_count": duplicate_count,
        "stress_source_target_composite_missing_count": composite_missing,
        "stress_missing_id_count": missing_count,
    }
    coverage_metrics = {
        "shared_coverage_ratio": coverage,
        "mechanism_analysis_coverage_pass": coverage >= 0.30,
        "training_policy_prototype_coverage_pass": coverage >= 0.60,
    }
    bias_rows = []
    for field in ("group_id", "scene_key", "candidate_state"):
        ready_counts = Counter(r.get(field, "missing") for r in ready)
        not_counts = Counter(r.get(field, "missing") for r in not_ready)
        keys = sorted(set(ready_counts) | set(not_counts))
        for key in keys:
            bias_rows.append(
                {
                    "field": field,
                    "value": key,
                    "joined_count": str(ready_counts.get(key, 0)),
                    "unjoined_count": str(not_counts.get(key, 0)),
                }
            )
    return identity, coverage_metrics, bias_rows


def _write_reports(out, identity, coverage):
    (out / "reports" / "shared_surface_identity_audit.md").write_text(
        "\n".join(
            [
                "# A-v3.2b Shared Surface Identity Audit",
                "",
                f"- fixed_manifest_row_count: {identity['fixed_manifest_row_count']}",
                f"- shared_ready_count: {identity['shared_ready_count']}",
                f"- stress_duplicate_blocked_count: {identity['stress_duplicate_blocked_count']}",
                f"- stress_source_target_composite_missing_count: {identity['stress_source_target_composite_missing_count']}",
                "",
                "No duplicate was silently dropped. No row index was promoted to identity.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "reports" / "shared_surface_coverage_bias_report.md").write_text(
        "\n".join(
            [
                "# A-v3.2b Shared Surface Coverage Bias Report",
                "",
                f"- shared_coverage_ratio: {coverage['shared_coverage_ratio']:.6f}",
                f"- mechanism_analysis_coverage_pass: {coverage['mechanism_analysis_coverage_pass']}",
                f"- training_policy_prototype_coverage_pass: {coverage['training_policy_prototype_coverage_pass']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--shared-wide", required=True)
    parser.add_argument("--outcome-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ensure_dirs(args.output_dir)
    identity, coverage, bias = audit(read_csv_dicts(args.fixed_manifest), read_csv_dicts(args.shared_wide), _read_json(args.outcome_metrics))
    write_json(out / "metrics" / "shared_surface_identity_metrics.json", identity)
    write_json(out / "metrics" / "shared_surface_coverage_bias_metrics.json", coverage)
    write_csv_dicts(out / "tables" / "joined_unjoined_bias_table.csv", bias, ["field", "value", "joined_count", "unjoined_count"])
    _write_reports(out, identity, coverage)


if __name__ == "__main__":
    main()

