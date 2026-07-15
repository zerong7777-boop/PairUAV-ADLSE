import json
from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import ensure_dirs


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _slice_counts(path):
    import csv

    counts = {}
    if not Path(path).exists():
        return counts
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["slice_name"]] = counts.get(row["slice_name"], 0) + 1
    return counts


def _read_verdict(path):
    if not Path(path).exists():
        return "missing"
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("verdict:"):
            return line.split("`")[1] if "`" in line else line.split(":", 1)[1].strip()
    return "missing"


def write_summary_report(output_dir, input_manifest=None):
    out = ensure_dirs(output_dir)
    metrics = out / "metrics"
    tables = out / "tables"
    reports = out / "reports"
    outcome = _read_json(metrics / "a_v3_outcome_surface_consistency_audit.json")
    predictability = _read_json(metrics / "a_v3_candidate_to_outcome_predictability_metrics.json")
    stable = _read_json(metrics / "a_v3_stable_control_stress_audit.json")
    join = _read_json(metrics / "a_v3_join_bias_extension_metrics.json")
    readiness = _read_verdict(reports / "a_v3_training_policy_readiness_verdict.md")
    counts = _slice_counts(tables / "a_v3_b_offline_diagnostic_slices.csv")

    lines = [
        "# Phase27 A-v3 Validation Extension Summary",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(input_manifest or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Row Counts",
        "",
        f"- total_rows: {outcome.get('total_rows')}",
        f"- shared_join_mask_count: {outcome.get('shared_join_mask_count')}",
        "",
        "## Outcome Consistency",
        "",
        f"- verdict: {outcome.get('verdict')}",
        f"- baseline_joint_hard_count: {outcome.get('baseline_joint_hard_count')}",
        f"- stress_joint_sensitive_count: {outcome.get('stress_joint_sensitive_count')}",
        f"- baseline_stress_overlap_count: {outcome.get('baseline_stress_overlap_count')}",
        "",
        "## Candidate-To-Outcome Predictability",
        "",
        f"- useful_pair_count: {predictability.get('useful_pair_count')}",
        "- top_pairs:",
    ]
    for row in predictability.get("best_pairs", [])[:5]:
        lines.append(
            f"  - {row['score_field']} -> {row['outcome_field']}: "
            f"p@100={row['precision_at_100']:.4f}, lift={row['top_decile_lift']:.4f}, auc={row['auc']:.4f}"
        )
    lines += [
        "",
        "## Stable Control",
        "",
        f"- verdict: {stable.get('verdict')}",
        f"- control_count: {stable.get('control_count')}",
        f"- tail_error_rate: {stable.get('tail_error_rate')}",
        "",
        "## Join Bias",
        "",
        f"- verdict: {join.get('verdict')}",
        f"- unknown_due_to_missing_join_count: {join.get('unknown_due_to_missing_join_count')}",
        "",
        "## B Diagnostic Slices",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Training-Policy Readiness",
        "",
        f"- verdict: {readiness}",
        "",
        "No training, finetuning, sample weighting, curriculum, oversampling, submission packaging, checkpoint selection, threshold tuning, sampler, or B/C gate training was run.",
    ]
    target = reports / "phase27_a_v3_validation_extension_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    manifest = None
    if args.input_manifest:
        manifest = _read_json(args.input_manifest)
    write_summary_report(args.output_dir, manifest)


if __name__ == "__main__":
    main()
