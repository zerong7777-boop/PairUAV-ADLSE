#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


B1_V1 = {
    "aggregate_delta": -0.001114,
    "phase11_semantic_decoupled_delta": -0.124274,
    "phase11_control_low_sim_low_error_delta": 0.176844,
    "verdict": "weak-inconclusive",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def decide(metrics, coverage_summary):
    slices = metrics["slices"]
    sem = slices["phase11_semantic_decoupled"]
    control = slices["phase11_control_low_sim_low_error"]
    aggregate = slices["aggregate_all"]
    phase14 = slices["phase14_alignment_sensitive"]
    coverage = coverage_summary["eval"]["coverage_rate"]

    if metrics["phase11_joined_rows"] <= 0 or metrics["phase14_joined_rows"] <= 0 or coverage < 0.95:
        return "blocked", "Missing required joins or matcher coverage below 95%."
    if sem["sample_count"] <= 0 or control["sample_count"] <= 0:
        return "blocked", "Missing primary Phase11 semantic/control slice."

    sem_delta = sem["delta_final_fusion_minus_baseline"]
    control_delta = control["delta_final_fusion_minus_baseline"]
    aggregate_delta = aggregate["delta_final_fusion_minus_baseline"]
    phase14_delta = phase14["delta_final_fusion_minus_baseline"]

    if sem_delta is None or sem_delta >= 0:
        return "rejected", "Frozen matcher fusion did not improve Phase11 semantic-decoupled slice."
    if control_delta is not None and control_delta > 0.01:
        return "rejected", "Frozen matcher fusion materially degraded Phase11 low-sim/low-error control."
    if aggregate_delta is not None and aggregate_delta > 0.005:
        return "weak-inconclusive", "Semantic slice improved but aggregate degraded beyond the primary tolerance."
    if phase14_delta is not None and phase14_delta > 0.01:
        return "weak-inconclusive", "Semantic/control conditions are acceptable but Phase14 alignment-sensitive slice degraded."
    return "method-promising", "Frozen matcher fusion improved semantic-decoupled slice without material control degradation."


def table(rows):
    lines = [
        "| slice | n | baseline_final | fusion_final | delta | baseline_dist | fusion_dist | baseline_angle | fusion_angle | covered | fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["sample_count"]),
                    fmt(row["baseline_final"]),
                    fmt(row["fusion_final"]),
                    fmt(row["delta_final_fusion_minus_baseline"]),
                    fmt(row["baseline_distance_rel_error"]),
                    fmt(row["fusion_distance_rel_error"]),
                    fmt(row["baseline_angle_rel_error"]),
                    fmt(row["fusion_angle_rel_error"]),
                    str(row["matcher_covered"]),
                    str(row["matcher_fallback"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-report", required=True)
    parser.add_argument("--coverage-summary", required=True)
    parser.add_argument("--slice-metrics", required=True)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--fusion-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    schema = load_json(args.schema_report)
    coverage = load_json(args.coverage_summary)
    metrics = load_json(args.slice_metrics)
    verdict, reason = decide(metrics, coverage)
    slices = metrics["slices"]

    content = f"""# Phase26 B1 Frozen External Matcher Fusion

## Status

Status: complete
Verdict: `{verdict}`
Reason: {reason}

## Important Caveat

The configured checkpoint path was `/media/jgzn/SSD_lexar/RZ/UAVM/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth`, but probe evidence showed its nested keys are `backbone.embedder...`, not Reloc3r/CroCo keys. The matched comparison is therefore valid as a bounded matched run against the prior Phase26 B1 baseline setup, but it must not be described as a true Reloc3r longrun-best initialization.

## Matcher Schema And Coverage

- cache_root: `{schema["cache_root"]}`
- detected_format: `{schema["detected_format"]}`
- available_fields: `{", ".join(schema["available_fields"])}`
- schema_verdict: `{schema["schema_verdict"]}`
- train coverage: `{coverage["train"]["covered"]}/{coverage["train"]["rows"]}` = `{coverage["train"]["coverage_rate"]:.6f}`
- eval coverage: `{coverage["eval"]["covered"]}/{coverage["eval"]["rows"]}` = `{coverage["eval"]["coverage_rate"]:.6f}`

## Training Provenance

- baseline_run_dir: `{args.baseline_run_dir}`
- fusion_run_dir: `{args.fusion_run_dir}`
- baseline_eval_root: `{metrics["baseline_eval_root"]}`
- fusion_eval_root: `{metrics["fusion_eval_root"]}`
- joined_rows: `{metrics["joined_rows"]}`
- phase11_joined_rows: `{metrics["phase11_joined_rows"]}`
- phase14_joined_rows: `{metrics["phase14_joined_rows"]}`

## Slice Metrics

{table(list(slices.items()))}

## Comparison To B1 v1 Token Local Alignment

- B1 v1 aggregate delta: `{B1_V1["aggregate_delta"]:.6f}`
- B1 v1 Phase11 semantic-decoupled delta: `{B1_V1["phase11_semantic_decoupled_delta"]:.6f}`
- B1 v1 Phase11 low-sim/low-error control delta: `{B1_V1["phase11_control_low_sim_low_error_delta"]:.6f}`
- B1 v1 verdict: `{B1_V1["verdict"]}`

## Decision

`{verdict}`. {reason}

This is bounded method evidence only. It is not leaderboard evidence.

## Next Action

If `method-promising`, promote only after fixing checkpoint provenance and rerunning a larger matched eval. If `weak-inconclusive`, diagnose slice size and initialization. If `rejected`, move to B2 pose-supervised contrastive learning.
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
