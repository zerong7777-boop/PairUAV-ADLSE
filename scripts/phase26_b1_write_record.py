#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_run_dirs(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def metric(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def slices_by_name(payload):
    return {row["slice"]: row for row in payload.get("slices", [])}


def value(slices, name, key):
    row = slices.get(name, {})
    raw = row.get(key)
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def fmt(raw):
    if raw is None:
        return "NA"
    if isinstance(raw, float):
        return f"{raw:.6f}"
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if math.isnan(parsed):
        return "NA"
    return f"{parsed:.6f}"


def decide(baseline_slice, alignment_slice):
    if baseline_slice.get("joined_phase11_rows", 0) == 0 or alignment_slice.get("joined_phase11_rows", 0) == 0:
        return "blocked", "Phase11 slice join is zero."
    if baseline_slice.get("joined_phase14_rows", 0) == 0 or alignment_slice.get("joined_phase14_rows", 0) == 0:
        return "blocked", "Phase14 slice join is zero."

    b = slices_by_name(baseline_slice)
    a = slices_by_name(alignment_slice)
    b_sem = value(b, "phase11_semantic_decoupled", "final_score_mean")
    a_sem = value(a, "phase11_semantic_decoupled", "final_score_mean")
    b_ctrl = value(b, "phase11_control_low_sim_low_error", "final_score_mean")
    a_ctrl = value(a, "phase11_control_low_sim_low_error", "final_score_mean")
    b_align = value(b, "phase14_alignment_sensitive", "final_score_mean")
    a_align = value(a, "phase14_alignment_sensitive", "final_score_mean")

    if b_sem is None or a_sem is None:
        return "blocked", "Phase11 semantic-decoupled slice metric is missing."
    if b_ctrl is None or a_ctrl is None:
        return "weak-inconclusive", "Phase11 control slice metric is missing."

    sem_delta = a_sem - b_sem
    ctrl_delta = a_ctrl - b_ctrl
    align_delta = None if b_align is None or a_align is None else a_align - b_align

    if sem_delta < 0 and ctrl_delta <= 0.01 and (align_delta is None or align_delta <= 0.01):
        return "method-promising", "Local alignment improves Phase11 semantic-decoupled cases without material control degradation."
    if sem_delta < 0:
        return "weak-inconclusive", "Local alignment improves semantic-decoupled cases but control or Phase14 evidence is weak."
    return "rejected", "Local alignment does not improve Phase11 semantic-decoupled final_score_mean."


def slice_table(b_slices, a_slices, names):
    lines = ["| slice | sample_count | baseline_final | alignment_final | delta alignment-baseline |", "|---|---:|---:|---:|---:|"]
    for name in names:
        b = b_slices.get(name, {})
        a = a_slices.get(name, {})
        b_final = value(b_slices, name, "final_score_mean")
        a_final = value(a_slices, name, "final_score_mean")
        delta = None if b_final is None or a_final is None else a_final - b_final
        count = b.get("sample_count", a.get("sample_count"))
        lines.append(f"| {name} | {count} | {fmt(b_final)} | {fmt(a_final)} | {fmt(delta)} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-summary", required=True)
    parser.add_argument("--run-dirs", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    subset = read_json(args.subset_summary)
    runs = read_run_dirs(args.run_dirs)
    baseline_metrics = read_json(Path(args.baseline_dir) / "official_metrics.json")
    alignment_metrics = read_json(Path(args.alignment_dir) / "official_metrics.json")
    baseline_slice = read_json(Path(args.baseline_dir) / "phase26_b1_slice_metrics.json")
    alignment_slice = read_json(Path(args.alignment_dir) / "phase26_b1_slice_metrics.json")
    verdict, reason = decide(baseline_slice, alignment_slice)
    b_slices = slices_by_name(baseline_slice)
    a_slices = slices_by_name(alignment_slice)

    aggregate_delta = metric(alignment_metrics, "final_score") - metric(baseline_metrics, "final_score")
    lines = [
        "# Phase26 B1 Geometry-Aware Local Alignment",
        "",
        "## Status",
        "",
        "Status: complete",
        f"Verdict: `{verdict}`",
        f"Reason: {reason}",
        "",
        "## Subset",
        "",
        f"- subset root: `{Path(args.subset_summary).parent}`",
        f"- train_rows: `{subset.get('train_rows')}`",
        f"- subset_manifest_eval_rows: `{subset.get('subset_manifest_eval_rows')}`",
        f"- official_eval_rows: `{subset.get('official_eval_rows')}`",
        f"- phase11_eval_overlap: `{subset.get('phase11_eval_overlap')}`",
        f"- phase14_eval_overlap: `{subset.get('phase14_eval_overlap')}`",
        "",
        "## Training Provenance",
        "",
        f"- baseline_run_dir: `{runs.get('baseline_run_dir')}`",
        f"- alignment_run_dir: `{runs.get('alignment_run_dir')}`",
        "- pretrained: `/media/jgzn/SSD_lexar/RZ/UAVM/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth`",
        "- budget: `100` train steps each, same bounded train/eval roots, same seed/config except model output mode.",
        "",
        "## Aggregate Metrics",
        "",
        "| model | rows | final_score | distance_rel_error | angle_rel_error |",
        "|---|---:|---:|---:|---:|",
        f"| baseline | {baseline_metrics.get('num_prediction_rows')} | {fmt(metric(baseline_metrics, 'final_score'))} | {fmt(metric(baseline_metrics, 'distance_rel_error'))} | {fmt(metric(baseline_metrics, 'angle_rel_error'))} |",
        f"| local_alignment | {alignment_metrics.get('num_prediction_rows')} | {fmt(metric(alignment_metrics, 'final_score'))} | {fmt(metric(alignment_metrics, 'distance_rel_error'))} | {fmt(metric(alignment_metrics, 'angle_rel_error'))} |",
        f"| delta alignment-baseline |  | {fmt(aggregate_delta)} | {fmt(metric(alignment_metrics, 'distance_rel_error') - metric(baseline_metrics, 'distance_rel_error'))} | {fmt(metric(alignment_metrics, 'angle_rel_error') - metric(baseline_metrics, 'angle_rel_error'))} |",
        "",
        "## Phase11 Slice Metrics",
        "",
        f"- baseline joined_phase11_rows: `{baseline_slice.get('joined_phase11_rows')}`",
        f"- alignment joined_phase11_rows: `{alignment_slice.get('joined_phase11_rows')}`",
        "",
        *slice_table(b_slices, a_slices, ["phase11_semantic_decoupled", "phase11_control_low_sim_low_error", "phase11_control_other"]),
        "",
        "## Phase14 Slice Metrics",
        "",
        f"- baseline joined_phase14_rows: `{baseline_slice.get('joined_phase14_rows')}`",
        f"- alignment joined_phase14_rows: `{alignment_slice.get('joined_phase14_rows')}`",
        "",
        *slice_table(b_slices, a_slices, ["phase14_alignment_sensitive", "phase14_alignment_control_middle"]),
        "",
        "## Decision",
        "",
        f"`{verdict}`. {reason}",
        "",
        "This is bounded method evidence only. It is not leaderboard evidence.",
        "",
        "## Risks",
        "",
        "- Eval set is intentionally small and Phase11/Phase14-overlap-biased.",
        "- Token similarity may still encode semantic similarity rather than true geometry.",
        "- Positive evidence would need larger matched eval before method promotion.",
        "",
        "## Next Action",
        "",
    ]
    if verdict == "method-promising":
        lines.append("Enter knowledge-review and decide whether to promote B1 to a larger matched eval.")
    elif verdict in {"weak-inconclusive", "rejected"}:
        lines.append("Decide between B2 pose-supervised contrastive learning and B1 external frozen matcher redesign.")
    else:
        lines.append("Fix identity alignment before drawing any method conclusion.")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(output), "verdict": verdict, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
