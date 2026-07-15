import json
from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, write_json


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def route_verdict(output_dir):
    out = Path(output_dir)
    coverage = read_json(out / "metrics" / "a_v3_1_shared_surface_coverage_metrics.json")
    if not coverage:
        verdict = "shared-surface-blocked-zero-coverage"
        reason = "coverage_metrics_missing"
    elif coverage.get("verdict") == "shared-surface-blocked-zero-coverage":
        verdict = "shared-surface-blocked-zero-coverage"
        reason = "no_shared_baseline_stress_pairs"
    elif coverage.get("verdict") == "shared-surface-blocked-identity-join":
        verdict = "shared-surface-blocked-identity-join"
        reason = "duplicate_or_unstable_identity_join"
    elif coverage.get("verdict") == "shared-surface-blocked-bias":
        verdict = "shared-surface-blocked-bias"
        reason = "joined_unjoined_bias_too_large"
    elif coverage.get("shared_coverage_ratio", 0) < 0.60:
        verdict = "shared-surface-analysis-only"
        reason = "coverage_below_shadow_policy_gate"
    else:
        predictability = read_json(out / "metrics" / "a_v3_1_shared_candidate_predictability_metrics.json")
        stable = read_json(out / "metrics" / "a_v3_1_shared_stable_control_metrics.json")
        if predictability.get("useful_pair_count", 0) > 0 and stable.get("verdict") == "control-anchor-shadow-candidate":
            verdict = "shadow-training-policy-spec-allowed"
            reason = "shared_surface_predictability_and_control_pass_shadow_gate"
        else:
            verdict = "shared-surface-pass-validation-only"
            reason = "shared_surface_passed_but_shadow_gate_not_met"
    data = {"verdict": verdict, "reason": reason}
    write_json(out / "metrics" / "a_v3_1_route_verdict.json", data)
    lines = ["# A-v3.1 Route Verdict", "", f"verdict: `{verdict}`", f"reason: `{reason}`", "", "No training, finetuning, sample weighting, threshold tuning, sampler, or B/C gate was run."]
    (out / "reports" / "a_v3_1_route_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def write_summary_report(output_dir, input_manifest=None):
    out = ensure_output_dirs(output_dir)
    coverage = read_json(out / "metrics" / "a_v3_1_shared_surface_coverage_metrics.json")
    consistency = read_json(out / "metrics" / "a_v3_1_shared_outcome_consistency_metrics.json")
    predictability = read_json(out / "metrics" / "a_v3_1_shared_candidate_predictability_metrics.json")
    stable = read_json(out / "metrics" / "a_v3_1_shared_stable_control_metrics.json")
    verdict = route_verdict(out)
    lines = [
        "# Phase27 A-v3.1 Shared Outcome Surface Summary",
        "",
        "## Inputs",
        "```json",
        json.dumps(input_manifest or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Coverage",
        f"- total_rows: {coverage.get('total_rows')}",
        f"- baseline_joined_count: {coverage.get('baseline_joined_count')}",
        f"- stress_joined_count_by_variant: {coverage.get('stress_joined_count_by_variant')}",
        f"- shared_joined_count: {coverage.get('shared_joined_count')}",
        f"- shared_coverage_ratio: {coverage.get('shared_coverage_ratio')}",
        f"- coverage_verdict: {coverage.get('verdict')}",
        "",
        "## Shared Outcome Consistency",
        f"- shared_rows: {consistency.get('shared_rows', 0)}",
        f"- baseline_joint_hard_count: {consistency.get('baseline_joint_hard_count', 0)}",
        "",
        "## Candidate Predictability",
        f"- useful_pair_count: {predictability.get('useful_pair_count', 0)}",
        "",
        "## Stable Control",
        f"- verdict: {stable.get('verdict', 'missing')}",
        f"- shared_control_count: {stable.get('shared_control_count', 0)}",
        "",
        "## Route Verdict",
        f"- verdict: {verdict['verdict']}",
        f"- reason: {verdict['reason']}",
        "",
        "No training, finetuning, sample weighting, curriculum, oversampling, checkpoint selection, submission packaging, threshold tuning, sampler, or B/C gate training was run.",
    ]
    path = out / "reports" / "phase27_a_v3_1_shared_outcome_surface_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    manifest = read_json(args.input_manifest) if args.input_manifest else None
    write_summary_report(args.output_dir, manifest)


if __name__ == "__main__":
    main()
