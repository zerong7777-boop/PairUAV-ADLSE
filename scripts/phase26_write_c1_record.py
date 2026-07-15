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
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def metric(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def slices_by_name(payload):
    return {row["slice"]: row for row in payload.get("slices", [])}


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def slice_value(slices, name, key):
    row = slices.get(name, {})
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide(baseline_slice, adapter_slice):
    if baseline_slice.get("joined_phase13_rows", 0) == 0 or adapter_slice.get("joined_phase13_rows", 0) == 0:
        return "blocked", "Phase13 slice join is zero."

    base_slices = slices_by_name(baseline_slice)
    adapter_slices = slices_by_name(adapter_slice)
    b_target = slice_value(base_slices, "target_heterogeneous", "final_score_mean")
    a_target = slice_value(adapter_slices, "target_heterogeneous", "final_score_mean")
    b_middle = slice_value(base_slices, "middle_or_ordinary", "final_score_mean")
    a_middle = slice_value(adapter_slices, "middle_or_ordinary", "final_score_mean")
    target_count = adapter_slices.get("target_heterogeneous", {}).get("sample_count", 0) or 0

    if b_target is None or a_target is None:
        return "blocked", "Target-heterogeneous slice metric is missing."

    target_delta = a_target - b_target
    middle_delta = None if b_middle is None or a_middle is None else a_middle - b_middle
    middle_ok = middle_delta is None or middle_delta <= 0.01

    if target_delta < 0 and middle_ok and target_count >= 20:
        return "method-promising", "Adapter improves target-heterogeneous final_score_mean without material middle degradation."
    if target_delta < 0:
        return "weak-inconclusive", "Adapter improves target-heterogeneous slice, but sample count or middle degradation makes evidence weak."
    return "rejected", "Adapter does not improve target-heterogeneous final_score_mean versus matched baseline."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-summary", required=True)
    parser.add_argument("--run-dirs", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    subset = read_json(args.subset_summary)
    runs = read_run_dirs(args.run_dirs)
    baseline_metrics = read_json(Path(args.baseline_dir) / "official_metrics.json")
    adapter_metrics = read_json(Path(args.adapter_dir) / "official_metrics.json")
    baseline_slice = read_json(Path(args.baseline_dir) / "phase26_slice_metrics.json")
    adapter_slice = read_json(Path(args.adapter_dir) / "phase26_slice_metrics.json")
    verdict, reason = decide(baseline_slice, adapter_slice)

    b_slices = slices_by_name(baseline_slice)
    a_slices = slices_by_name(adapter_slice)
    slice_rows = []
    for name in ["all", "target_heterogeneous", "heading_hard_range_easy", "range_hard_heading_easy", "middle_or_ordinary"]:
        b = b_slices.get(name, {})
        a = a_slices.get(name, {})
        b_final = b.get("final_score_mean")
        a_final = a.get("final_score_mean")
        delta = None
        if b_final is not None and a_final is not None:
            delta = float(a_final) - float(b_final)
        slice_rows.append(
            f"| {name} | {fmt(b.get('sample_count'))} | {fmt(b_final)} | {fmt(a_final)} | {fmt(delta)} |"
        )

    aggregate_delta = metric(adapter_metrics, "final_score") - metric(baseline_metrics, "final_score")
    lines = [
        "# Phase26 C1 Bounded Matched Eval",
        "",
        "## Status",
        "",
        f"Status: complete",
        f"Verdict: `{verdict}`",
        f"Reason: {reason}",
        "",
        "## Subset",
        "",
        f"- subset root: `{Path(args.subset_summary).parent}`",
        f"- train_rows: `{subset.get('train_rows')}`",
        f"- subset_manifest_eval_rows: `{subset.get('eval_rows')}`",
        f"- official_eval_rows: `{baseline_metrics.get('num_manifest_rows')}`",
        f"- phase13_eval_overlap: `{subset.get('phase13_eval_overlap')}`",
        "",
        "## Training Provenance",
        "",
        f"- baseline_run_dir: `{runs.get('baseline_run_dir')}`",
        f"- adapter_run_dir: `{runs.get('adapter_run_dir')}`",
        "- pretrained: `/media/jgzn/SSD_lexar/RZ/UAVM/synced_results/reloc3r_v2_long_20260423_232457_result_bundle/best_checkpoint.pth`",
        "- budget: `100` train steps each, same bounded train/eval roots, same seed/config except model output mode.",
        "",
        "## Aggregate Metrics",
        "",
        "| model | rows | final_score | distance_rel_error | angle_rel_error |",
        "|---|---:|---:|---:|---:|",
        f"| baseline | {fmt(baseline_metrics.get('num_prediction_rows'))} | {fmt(metric(baseline_metrics, 'final_score'))} | {fmt(metric(baseline_metrics, 'distance_rel_error'))} | {fmt(metric(baseline_metrics, 'angle_rel_error'))} |",
        f"| adapter | {fmt(adapter_metrics.get('num_prediction_rows'))} | {fmt(metric(adapter_metrics, 'final_score'))} | {fmt(metric(adapter_metrics, 'distance_rel_error'))} | {fmt(metric(adapter_metrics, 'angle_rel_error'))} |",
        f"| delta adapter-baseline |  | {fmt(aggregate_delta)} | {fmt(metric(adapter_metrics, 'distance_rel_error') - metric(baseline_metrics, 'distance_rel_error'))} | {fmt(metric(adapter_metrics, 'angle_rel_error') - metric(baseline_metrics, 'angle_rel_error'))} |",
        "",
        "## Slice Metrics",
        "",
        f"- baseline joined_phase13_rows: `{baseline_slice.get('joined_phase13_rows')}`",
        f"- adapter joined_phase13_rows: `{adapter_slice.get('joined_phase13_rows')}`",
        "",
        "| slice | sample_count | baseline_final | adapter_final | delta adapter-baseline |",
        "|---|---:|---:|---:|---:|",
        *slice_rows,
        "",
        "## Decision",
        "",
        f"`{verdict}`. {reason}",
        "",
        "This is bounded method evidence only. It is not leaderboard evidence and should not be promoted to full training unless the decision is `method-promising`.",
        "",
        "## Risks",
        "",
        "- Eval set is intentionally small and Phase13-overlap-biased.",
        "- The subset manifest had 39 eval rows, but PairUAV's actual dataset-order eval root contains 33 unique JSON samples; official eval used the 33-row dataset-order manifest.",
        "- Positive evidence would still need a larger matched full eval before architectural promotion.",
        "- Negative evidence should stop this C1 adapter route, not Pillar 3 itself.",
        "",
        "## Next Action",
        "",
    ]
    if verdict == "method-promising":
        lines.append("Enter knowledge-review, then design a larger matched eval before any leaderboard-oriented training.")
    elif verdict in {"weak-inconclusive", "rejected"}:
        lines.append("Stop C1 adapter optimization and move to B1 geometry-aware local alignment design.")
    else:
        lines.append("Fix identity alignment before drawing any method verdict.")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(output), "verdict": verdict, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
