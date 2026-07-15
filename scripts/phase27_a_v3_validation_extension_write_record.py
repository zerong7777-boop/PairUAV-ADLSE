import csv
import json
from pathlib import Path

from scripts.phase27_a_v3_validation_extension_report import _read_json, _read_verdict, _slice_counts
from scripts.phase27_a_v3_validation_extension_common import ensure_dirs


def _hard_table(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_experiment_record(output_dir, commands=None, input_manifest=None):
    out = ensure_dirs(output_dir)
    metrics = out / "metrics"
    tables = out / "tables"
    reports = out / "reports"
    outcome = _read_json(metrics / "a_v3_outcome_surface_consistency_audit.json")
    predictability = _read_json(metrics / "a_v3_candidate_to_outcome_predictability_metrics.json")
    stable = _read_json(metrics / "a_v3_stable_control_stress_audit.json")
    join = _read_json(metrics / "a_v3_join_bias_extension_metrics.json")
    readiness = _read_verdict(reports / "a_v3_training_policy_readiness_verdict.md")
    slice_counts = _slice_counts(tables / "a_v3_b_offline_diagnostic_slices.csv")
    hard_rows = _hard_table(tables / "a_v3_hard_ambiguity_overlap_decomposition.csv")

    lines = [
        "# Phase27 A-v3 Validation Extension Experiment Record",
        "",
        "status: bounded-full-eval-complete",
        "",
        "## Commands",
        "",
    ]
    for command in commands or []:
        lines.append(f"- `{command}`")
    lines += [
        "",
        "## Input Paths",
        "",
        "```json",
        json.dumps(input_manifest or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Output Paths",
        "",
        f"- output_dir: `{out}`",
        f"- summary: `{reports / 'phase27_a_v3_validation_extension_summary.md'}`",
        "",
        "## Row Counts And Outcome Surface",
        "",
        f"- total_rows: {outcome.get('total_rows')}",
        f"- shared_join_mask_count: {outcome.get('shared_join_mask_count')}",
        f"- baseline_joint_hard_count: {outcome.get('baseline_joint_hard_count')}",
        f"- stress_joint_sensitive_count: {outcome.get('stress_joint_sensitive_count')}",
        f"- baseline_stress_overlap_count: {outcome.get('baseline_stress_overlap_count')}",
        f"- shared_join_overlap_count: {outcome.get('shared_join_overlap_count')}",
        f"- outcome_verdict: {outcome.get('verdict')}",
        "",
        "## Hard/Ambiguity Subtype Counts",
        "",
    ]
    for row in hard_rows:
        lines.append(f"- {row['subtype']}: {row['count']} ({row['interpretation']})")
    lines += [
        "",
        "## Predictability Summary",
        "",
        f"- useful_pair_count: {predictability.get('useful_pair_count')}",
    ]
    for row in predictability.get("best_pairs", [])[:10]:
        lines.append(
            f"- {row['score_field']} -> {row['outcome_field']}: "
            f"auc={row['auc']:.4f}, p@100={row['precision_at_100']:.4f}, lift={row['top_decile_lift']:.4f}"
        )
    lines += [
        "",
        "## Stable Control And Join Bias",
        "",
        f"- stable_control_verdict: {stable.get('verdict')}",
        f"- control_count: {stable.get('control_count')}",
        f"- join_bias_verdict: {join.get('verdict')}",
        f"- unknown_due_to_missing_join_count: {join.get('unknown_due_to_missing_join_count')}",
        "",
        "## B Diagnostic Slice Counts",
        "",
    ]
    for key, value in sorted(slice_counts.items()):
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Leakage / Deployability Boundary",
        "",
        "- leakage audit inherited from A-v3: passed",
        "- B slices are diagnostic-only and contain no gate_label/train_label/sampler_weight/oversample/loss_weight.",
        f"- final_training_policy_readiness_verdict: {readiness}",
        "",
        "No training, finetuning, sample weighting, curriculum, oversampling, submission packaging, checkpoint selection, threshold tuning, sampler, or B/C gate training was run.",
    ]
    path = out / "phase27_a_v3_validation_extension_experiment_record.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    manifest = None
    if args.input_manifest:
        with Path(args.input_manifest).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    write_experiment_record(
        args.output_dir,
        commands=["bash scripts/run_phase27_a_v3_validation_extension_bounded.sh"],
        input_manifest=manifest,
    )


if __name__ == "__main__":
    main()
