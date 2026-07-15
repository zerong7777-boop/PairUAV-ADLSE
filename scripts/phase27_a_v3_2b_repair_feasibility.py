"""Repair feasibility and route verdict for A-v3.2b."""
import argparse
import json
from pathlib import Path

from scripts.phase27_a_v3_2b_common import ensure_dirs, read_csv_dicts, write_csv_dicts, write_json


def _read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def decide_verdict(identity, coverage, runner):
    decision = runner.get("decision", "")
    duplicates = int(identity.get("stress_duplicate_blocked_count", 0))
    composite_missing = int(identity.get("stress_source_target_composite_missing_count", 0))
    missing = int(identity.get("stress_missing_id_count", 0))
    ratio = float(coverage.get("shared_coverage_ratio", 0.0))
    if decision == "blocked_runner_interface":
        return {"verdict": "shared-surface-blocked-runner-interface", "reason": "no_fixed_manifest_runner_available_reexport_only_diagnostic"}
    if duplicates > 0:
        return {"verdict": "shared-surface-blocked-identity-duplicates", "reason": "stress_duplicate_identity_groups_block_one_to_one_surface"}
    if composite_missing > 0:
        return {"verdict": "shared-surface-repair-only-inconclusive", "reason": "stress_source_target_composite_identity_missing"}
    if ratio < 0.30 or missing > 0:
        return {"verdict": "shared-surface-blocked-low-coverage", "reason": "shared_coverage_below_mechanism_analysis_threshold_or_missing_rows"}
    return {"verdict": "shared-surface-ready-for-outcome-audit", "reason": "identity_clean_and_bounded_coverage_sufficient"}


def repair_candidates(duplicate_rows, identity):
    rows = []
    if duplicate_rows:
        rows.append(
            {
                "repair_class": "duplicate_identity_attribution_required",
                "affected_count": str(len(duplicate_rows)),
                "promotion_support": "no",
                "next_action": "attribute duplicate groups before any aggregation",
            }
        )
    if int(identity.get("stress_source_target_composite_missing_count", 0)) > 0:
        rows.append(
            {
                "repair_class": "stress_composite_identity_reacquisition_required",
                "affected_count": str(identity.get("stress_source_target_composite_missing_count", 0)),
                "promotion_support": "no",
                "next_action": "rerun or re-export stress surfaces with source/target identity",
            }
        )
    if not rows:
        rows.append({"repair_class": "no_repair_required", "affected_count": "0", "promotion_support": "possible", "next_action": "proceed to outcome-consistency audit"})
    return rows


def _write_reports(out, verdict, repairs, identity):
    (out / "reports" / "a_v3_2b_route_verdict.md").write_text(
        "\n".join(
            [
                "# A-v3.2b Route Verdict",
                "",
                f"verdict: `{verdict['verdict']}`",
                f"reason: `{verdict['reason']}`",
                "",
                "No training, finetuning, threshold tuning, submission packaging, B/C gate creation, fuzzy join, or silent deduplication was run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "reports" / "stress_identity_failure_report.md").write_text(
        "\n".join(
            [
                "# A-v3.2b Stress Identity Failure Report",
                "",
                f"- stress_duplicate_blocked_count: {identity.get('stress_duplicate_blocked_count', 0)}",
                f"- stress_source_target_composite_missing_count: {identity.get('stress_source_target_composite_missing_count', 0)}",
                f"- stress_missing_id_count: {identity.get('stress_missing_id_count', 0)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "reports" / "stress_duplicate_group_summary.md").write_text(
        "# A-v3.2b Stress Duplicate Group Summary\n\n"
        + f"- repair_rows: {len(repairs)}\n"
        + "- duplicate rows are retained for attribution; no keep-first policy was applied.\n",
        encoding="utf-8",
    )
    (out / "reports" / "stress_missing_composite_identity_report.md").write_text(
        "# A-v3.2b Stress Missing Composite Identity Report\n\n"
        + f"- missing_count: {identity.get('stress_source_target_composite_missing_count', 0)}\n",
        encoding="utf-8",
    )
    (out / "reports" / "repair_feasibility_verdict.md").write_text(
        "# A-v3.2b Repair Feasibility Verdict\n\n"
        + "\n".join(f"- {r['repair_class']}: {r['next_action']}" for r in repairs)
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--stress-duplicates", required=True)
    parser.add_argument("--identity-metrics", required=True)
    parser.add_argument("--coverage-metrics", required=True)
    parser.add_argument("--runner-capability", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ensure_dirs(args.output_dir)
    duplicates = read_csv_dicts(args.stress_duplicates)
    identity = _read_json(args.identity_metrics)
    coverage = _read_json(args.coverage_metrics)
    runner = _read_json(args.runner_capability)
    verdict = decide_verdict(identity, coverage, runner)
    repairs = repair_candidates(duplicates, identity)
    write_csv_dicts(out / "tables" / "stress_repair_candidate_table.csv", repairs, ["repair_class", "affected_count", "promotion_support", "next_action"])
    write_json(out / "metrics" / "repair_feasibility_metrics.json", {"repair_candidate_count": len(repairs), "repair_classes": [r["repair_class"] for r in repairs]})
    write_json(out / "metrics" / "a_v3_2b_route_verdict.json", verdict)
    _write_reports(out, verdict, repairs, identity)


if __name__ == "__main__":
    main()
