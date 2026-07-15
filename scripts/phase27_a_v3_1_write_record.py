import csv
import json
from pathlib import Path

from scripts.phase27_a_v3_1_report import read_json


def _count_table(path):
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_experiment_record(output_dir, input_manifest=None):
    out = Path(output_dir)
    coverage = read_json(out / "metrics" / "a_v3_1_shared_surface_coverage_metrics.json")
    consistency = read_json(out / "metrics" / "a_v3_1_shared_outcome_consistency_metrics.json")
    predictability = read_json(out / "metrics" / "a_v3_1_shared_candidate_predictability_metrics.json")
    stable = read_json(out / "metrics" / "a_v3_1_shared_stable_control_metrics.json")
    verdict = read_json(out / "metrics" / "a_v3_1_route_verdict.json")
    lines = [
        "# Phase27 A-v3.1 Shared Outcome Surface Experiment Record",
        "",
        "status: `bounded-full-eval-complete`",
        "",
        "## Commands",
        "- `bash scripts/run_phase27_a_v3_1_shared_surface_bounded.sh`",
        "",
        "## Inputs",
        "```json",
        json.dumps(input_manifest or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Join Key Policy",
        "- priority: canonical_pair_id -> source/target/pair composite -> fallback pair id",
        "- duplicate joins are blocked, not silently first-matched",
        "",
        "## Coverage",
        f"- total_rows: {coverage.get('total_rows')}",
        f"- baseline_joined_count: {coverage.get('baseline_joined_count')}",
        f"- stress_joined_count_by_variant: {coverage.get('stress_joined_count_by_variant')}",
        f"- shared_joined_count: {coverage.get('shared_joined_count')}",
        f"- shared_coverage_ratio: {coverage.get('shared_coverage_ratio')}",
        f"- duplicate_blocked_count: {coverage.get('duplicate_blocked_count')}",
        f"- coverage_verdict: {coverage.get('verdict')}",
        "",
        "## Shared Outcomes",
        f"- shared_rows: {consistency.get('shared_rows', 0)}",
        f"- baseline_joint_hard_count: {consistency.get('baseline_joint_hard_count', 0)}",
        "",
        "## Predictability And Controls",
        f"- useful_pair_count: {predictability.get('useful_pair_count', 0)}",
        f"- stable_control_verdict: {stable.get('verdict', 'missing')}",
        "",
        "## Tables",
        f"- join_bias_rows: {_count_table(out / 'tables' / 'a_v3_1_join_bias_by_target_group.csv')}",
        f"- predictability_by_group_rows: {_count_table(out / 'tables' / 'a_v3_1_candidate_predictability_by_target_group.csv')}",
        f"- hard_ambiguity_rows: {_count_table(out / 'tables' / 'a_v3_1_hard_ambiguity_shared_decomposition.csv')}",
        "",
        "## Final Verdict",
        f"- route_verdict: {verdict.get('verdict')}",
        f"- reason: {verdict.get('reason')}",
        "",
        "No training, finetuning, sample weighting, curriculum, oversampling, checkpoint selection, submission packaging, threshold tuning, sampler, or B/C gate training was run.",
    ]
    path = out / "phase27_a_v3_1_shared_outcome_surface_experiment_record.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    manifest = read_json(args.input_manifest) if args.input_manifest else None
    write_experiment_record(args.output_dir, manifest)


if __name__ == "__main__":
    main()
