#!/usr/bin/env python3
"""Phase100 H0/H1 checkpoint-trajectory heterogeneity audit.

This CPU-only diagnostic expands existing local-val checkpoint predictions into
per-sample best-checkpoint maps. It uses fixed val811 labels and already
generated prediction files only; it does not read official hidden-test labels or
leaderboard feedback.

H0: Different checkpoints are optimal for different samples.
H1: Heading-optimal and range-optimal checkpoints may not coincide.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CASES = (
    "step050000,step100000,step150000,step200000,step250000,"
    "step300000,step350000,step400000,step450000,final"
)
DEFAULT_STEP_TEMPLATE = (
    "{prediction_root}/metrics/phase95_A0_phase89_H8_{case_id}_val811/"
    "val_predict_output.txt"
)
DEFAULT_FINAL_PRED = (
    "{prediction_root}/metrics/"
    "phase89_Wstrip_T2_H8_mid_late_from_phase88_10k_fulltrain_1epoch_"
    "lr1e-5_bs4_pf20_20260618_val811/val_predict_output.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-per-sample", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-list", default=DEFAULT_CASES)
    parser.add_argument("--step-prediction-template", default=DEFAULT_STEP_TEMPLATE)
    parser.add_argument("--final-prediction-path", default=DEFAULT_FINAL_PRED)
    parser.add_argument("--max-pairs", type=int, default=811)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def median(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(statistics.median(clean))


def quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    pos = min(max(q, 0.0), 1.0) * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def angle_abs_error_deg(a: float, b: float) -> float:
    return abs(wrap_angle_diff_deg(a, b))


def read_prediction_file(path: Path, max_pairs: int) -> list[tuple[float, float]]:
    preds: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            preds.append((float(parts[0]), float(parts[1])))
            if len(preds) >= max_pairs:
                break
    if not preds:
        raise ValueError(f"No predictions read from {path}")
    return preds


def case_prediction_path(args: argparse.Namespace, case_id: str) -> Path:
    template = args.final_prediction_path if case_id == "final" else args.step_prediction_template
    return Path(template.format(prediction_root=args.prediction_root, case_id=case_id))


def infer_range_span(rows: list[dict[str, Any]]) -> float:
    candidates = []
    true_ranges = []
    for row in rows:
        true_range = f(row, "true_range")
        if math.isfinite(true_range):
            true_ranges.append(true_range)
        abs_err = f(row, "range_abs_error")
        rel_err = f(row, "distance_rel_error")
        if math.isfinite(abs_err) and math.isfinite(rel_err) and rel_err > 0:
            candidates.append(abs_err / rel_err)
    if candidates:
        return float(statistics.median(candidates))
    if len(true_ranges) >= 2:
        return max(max(true_ranges) - min(true_ranges), 1e-12)
    return 1.0


def case_family(case_id: str) -> str:
    if case_id == "final":
        return "final"
    step = int(case_id.replace("step", ""))
    if step <= 200000:
        return "early_050_200k"
    if step <= 300000:
        return "mid_250_300k"
    return "late_350_450k"


def step_value(case_id: str) -> int:
    if case_id == "final":
        return 459999
    return int(case_id.replace("step", ""))


def top_two(items: list[tuple[str, float]]) -> tuple[tuple[str, float], tuple[str, float]]:
    ordered = sorted(items, key=lambda item: (item[1], step_value(item[0])))
    if len(ordered) == 1:
        return ordered[0], ("", math.nan)
    return ordered[0], ordered[1]


def win_summary(
    per_sample: list[dict[str, Any]],
    case_ids: list[str],
    axis_key: str,
    error_key: str,
    margin_key: str,
    baseline_key: str | None = None,
) -> list[dict[str, Any]]:
    counts = Counter(str(row[axis_key]) for row in per_sample)
    rows = []
    for case_id in case_ids:
        selected = [row for row in per_sample if str(row[axis_key]) == case_id]
        rows.append(
            {
                "axis": axis_key,
                "case_id": case_id,
                "family": case_family(case_id),
                "train_step": step_value(case_id),
                "win_count": counts.get(case_id, 0),
                "win_frac": counts.get(case_id, 0) / max(len(per_sample), 1),
                "mean_selected_error": mean([f(row, error_key) for row in selected]),
                "median_margin_to_second": median([f(row, margin_key) for row in selected]),
                "mean_margin_to_second": mean([f(row, margin_key) for row in selected]),
                "mean_baseline_minus_selected": (
                    mean([f(row, baseline_key) - f(row, error_key) for row in selected])
                    if baseline_key
                    else math.nan
                ),
            }
        )
    return rows


def group_summary(per_sample: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        groups[str(row.get(group_key, ""))].append(row)
    rows = []
    for value, selected in groups.items():
        axis_mismatch_count = sum(int(f(row, "heading_range_best_case_mismatch", 0.0)) for row in selected)
        final_winner_count = sum(1 for row in selected if row["best_final_case"] == "final")
        rows.append(
            {
                "group_key": group_key,
                "group_value": value,
                "count": len(selected),
                "mean_baseline_final_error": mean([f(row, "baseline_final_error") for row in selected]),
                "mean_best_checkpoint_error": mean([f(row, "best_final_error") for row in selected]),
                "mean_axiswise_oracle_error": mean([f(row, "axiswise_oracle_error") for row in selected]),
                "mean_final_minus_best_checkpoint": mean(
                    [f(row, "baseline_minus_best_final_error") for row in selected]
                ),
                "mean_best_checkpoint_minus_axiswise": mean(
                    [f(row, "best_checkpoint_minus_axiswise_oracle") for row in selected]
                ),
                "axis_mismatch_count": axis_mismatch_count,
                "axis_mismatch_frac": axis_mismatch_count / max(len(selected), 1),
                "final_winner_count": final_winner_count,
                "final_winner_frac": final_winner_count / max(len(selected), 1),
                "best_final_case_counts": json.dumps(
                    dict(sorted(Counter(str(row["best_final_case"]) for row in selected).items())),
                    sort_keys=True,
                ),
                "best_heading_case_counts": json.dumps(
                    dict(sorted(Counter(str(row["best_heading_case"]) for row in selected).items())),
                    sort_keys=True,
                ),
                "best_range_case_counts": json.dumps(
                    dict(sorted(Counter(str(row["best_range_case"]) for row in selected).items())),
                    sort_keys=True,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["count"]),
            -float(row["axis_mismatch_frac"]),
            -float(row["mean_final_minus_best_checkpoint"]),
            str(row["group_value"]),
        ),
    )


def build_per_sample(
    rows: list[dict[str, str]],
    predictions: dict[str, list[tuple[float, float]]],
    case_ids: list[str],
    range_span: float,
) -> list[dict[str, Any]]:
    output = []
    for idx, row in enumerate(rows):
        true_h = f(row, "true_heading_deg")
        true_r = f(row, "true_range")
        case_errors = []
        heading_errors = []
        range_errors = []
        pred_lookup: dict[str, tuple[float, float]] = {}
        for case_id in case_ids:
            pred_h, pred_r = predictions[case_id][idx]
            pred_lookup[case_id] = (pred_h, pred_r)
            angle_rel = angle_abs_error_deg(pred_h, true_h) / 180.0
            distance_rel = abs(pred_r - true_r) / max(range_span, 1e-12)
            final_score = 0.5 * (angle_rel + distance_rel)
            case_errors.append((case_id, final_score))
            heading_errors.append((case_id, angle_rel))
            range_errors.append((case_id, distance_rel))

        (best_final_case, best_final_error), (_second_final_case, second_final_error) = top_two(case_errors)
        (best_heading_case, best_heading_error), (_second_heading_case, second_heading_error) = top_two(heading_errors)
        (best_range_case, best_range_error), (_second_range_case, second_range_error) = top_two(range_errors)
        final_error = dict(case_errors)["final"]
        final_angle_error = dict(heading_errors)["final"]
        final_range_error = dict(range_errors)["final"]
        final_rank = [case for case, _score in sorted(case_errors, key=lambda item: item[1])].index("final") + 1

        axiswise_oracle_error = 0.5 * (best_heading_error + best_range_error)
        best_h_pred, _ = pred_lookup[best_heading_case]
        _, best_r_pred = pred_lookup[best_range_case]
        out = dict(row)
        out.update(
            {
                "sample_index": row.get("sample_index", idx),
                "range_span": range_span,
                "baseline_final_error": final_error,
                "baseline_final_angle_rel_error": final_angle_error,
                "baseline_final_distance_rel_error": final_range_error,
                "best_final_case": best_final_case,
                "best_final_family": case_family(best_final_case),
                "best_final_train_step": step_value(best_final_case),
                "best_final_error": best_final_error,
                "best_final_margin_to_second": second_final_error - best_final_error,
                "baseline_minus_best_final_error": final_error - best_final_error,
                "baseline_final_error_rank": final_rank,
                "best_heading_case": best_heading_case,
                "best_heading_family": case_family(best_heading_case),
                "best_heading_train_step": step_value(best_heading_case),
                "best_heading_angle_rel_error": best_heading_error,
                "best_heading_margin_to_second": second_heading_error - best_heading_error,
                "baseline_minus_best_heading_angle": final_angle_error - best_heading_error,
                "best_range_case": best_range_case,
                "best_range_family": case_family(best_range_case),
                "best_range_train_step": step_value(best_range_case),
                "best_range_distance_rel_error": best_range_error,
                "best_range_margin_to_second": second_range_error - best_range_error,
                "baseline_minus_best_range_distance": final_range_error - best_range_error,
                "heading_range_best_case_mismatch": int(best_heading_case != best_range_case),
                "best_checkpoint_axes_aligned": int(best_final_case == best_heading_case == best_range_case),
                "axiswise_oracle_heading_case": best_heading_case,
                "axiswise_oracle_range_case": best_range_case,
                "axiswise_oracle_heading_pred": best_h_pred,
                "axiswise_oracle_range_pred": best_r_pred,
                "axiswise_oracle_error": axiswise_oracle_error,
                "baseline_minus_axiswise_oracle": final_error - axiswise_oracle_error,
                "best_checkpoint_minus_axiswise_oracle": best_final_error - axiswise_oracle_error,
                "r5_traj_best_case_agrees": int(str(row.get("traj_best_case", "")) == best_final_case),
            }
        )
        output.append(out)
    return output


def summarize(per_sample: list[dict[str, Any]], case_ids: list[str], prediction_paths: dict[str, str]) -> dict[str, Any]:
    baseline = mean([f(row, "baseline_final_error") for row in per_sample])
    best_checkpoint = mean([f(row, "best_final_error") for row in per_sample])
    axiswise = mean([f(row, "axiswise_oracle_error") for row in per_sample])
    axis_mismatch = sum(int(f(row, "heading_range_best_case_mismatch", 0.0)) for row in per_sample)
    final_winner = sum(1 for row in per_sample if row["best_final_case"] == "final")
    aligned = sum(int(f(row, "best_checkpoint_axes_aligned", 0.0)) for row in per_sample)
    r5_agree = sum(int(f(row, "r5_traj_best_case_agrees", 0.0)) for row in per_sample)
    return {
        "phase": "phase100_h0_h1_checkpoint_heterogeneity",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "processed_pairs": len(per_sample),
        "case_ids": case_ids,
        "prediction_paths": prediction_paths,
        "baseline_final_error": baseline,
        "oracle_best_checkpoint_error": best_checkpoint,
        "oracle_best_checkpoint_delta": best_checkpoint - baseline,
        "axiswise_heading_range_oracle_error": axiswise,
        "axiswise_heading_range_oracle_delta": axiswise - baseline,
        "axiswise_extra_delta_over_best_checkpoint": axiswise - best_checkpoint,
        "final_is_best_checkpoint_count": final_winner,
        "final_is_best_checkpoint_frac": final_winner / max(len(per_sample), 1),
        "heading_range_best_case_mismatch_count": axis_mismatch,
        "heading_range_best_case_mismatch_frac": axis_mismatch / max(len(per_sample), 1),
        "all_axes_same_best_checkpoint_count": aligned,
        "all_axes_same_best_checkpoint_frac": aligned / max(len(per_sample), 1),
        "r5_traj_best_case_agreement_count": r5_agree,
        "r5_traj_best_case_agreement_frac": r5_agree / max(len(per_sample), 1),
        "best_final_case_counts": dict(sorted(Counter(str(row["best_final_case"]) for row in per_sample).items())),
        "best_heading_case_counts": dict(sorted(Counter(str(row["best_heading_case"]) for row in per_sample).items())),
        "best_range_case_counts": dict(sorted(Counter(str(row["best_range_case"]) for row in per_sample).items())),
        "best_final_family_counts": dict(sorted(Counter(str(row["best_final_family"]) for row in per_sample).items())),
        "best_heading_family_counts": dict(sorted(Counter(str(row["best_heading_family"]) for row in per_sample).items())),
        "best_range_family_counts": dict(sorted(Counter(str(row["best_range_family"]) for row in per_sample).items())),
        "baseline_minus_best_final_q50": quantile([f(row, "baseline_minus_best_final_error") for row in per_sample], 0.5),
        "baseline_minus_best_final_q90": quantile([f(row, "baseline_minus_best_final_error") for row in per_sample], 0.9),
        "best_checkpoint_minus_axiswise_q50": quantile(
            [f(row, "best_checkpoint_minus_axiswise_oracle") for row in per_sample], 0.5
        ),
        "best_checkpoint_minus_axiswise_q90": quantile(
            [f(row, "best_checkpoint_minus_axiswise_oracle") for row in per_sample], 0.9
        ),
        "no_hidden_test_labels": True,
    }


def write_markdown(
    output_dir: Path,
    summary: dict[str, Any],
    win_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in win_rows:
        by_axis[str(row["axis"])].append(row)

    md = [
        "# Phase100 H0/H1 Checkpoint Heterogeneity",
        "",
        "CPU-only fixed-val diagnostic. No official hidden-test labels or leaderboard feedback are used.",
        "",
        "## Key Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| processed pairs | {summary['processed_pairs']} |",
        f"| H8 final baseline | {float(summary['baseline_final_error']):.10g} |",
        f"| oracle best checkpoint | {float(summary['oracle_best_checkpoint_error']):.10g} |",
        f"| best-checkpoint delta | {float(summary['oracle_best_checkpoint_delta']):.10g} |",
        f"| axiswise heading/range oracle | {float(summary['axiswise_heading_range_oracle_error']):.10g} |",
        f"| axiswise delta | {float(summary['axiswise_heading_range_oracle_delta']):.10g} |",
        f"| axiswise extra over best checkpoint | {float(summary['axiswise_extra_delta_over_best_checkpoint']):.10g} |",
        f"| final is best checkpoint | {summary['final_is_best_checkpoint_count']} ({float(summary['final_is_best_checkpoint_frac']):.3f}) |",
        f"| heading/range best mismatch | {summary['heading_range_best_case_mismatch_count']} ({float(summary['heading_range_best_case_mismatch_frac']):.3f}) |",
        f"| all axes same best | {summary['all_axes_same_best_checkpoint_count']} ({float(summary['all_axes_same_best_checkpoint_frac']):.3f}) |",
        "",
        "## Win Counts",
    ]
    for axis in ("best_final_case", "best_heading_case", "best_range_case"):
        md.extend(["", f"### {axis}", "", "| case | family | wins | frac | mean selected error | median margin |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in by_axis.get(axis, []):
            if int(row["win_count"]) == 0:
                continue
            md.append(
                f"| `{row['case_id']}` | `{row['family']}` | {int(row['win_count'])} | "
                f"{float(row['win_frac']):.4f} | {float(row['mean_selected_error']):.10g} | "
                f"{float(row['median_margin_to_second']):.10g} |"
            )

    md.extend(
        [
            "",
            "## High-Signal Regime Slices",
            "",
            "| group | value | count | axis mismatch | final winner | mean final-best | mean best-axiswise |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_rows[:18]:
        md.append(
            f"| `{row['group_key']}` | `{row['group_value']}` | {int(row['count'])} | "
            f"{float(row['axis_mismatch_frac']):.3f} | {float(row['final_winner_frac']):.3f} | "
            f"{float(row['mean_final_minus_best_checkpoint']):.10g} | "
            f"{float(row['mean_best_checkpoint_minus_axiswise']):.10g} |"
        )

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- H0 is supported if final is not the dominant per-sample winner and oracle best-checkpoint delta is materially negative.",
            "- H1 is supported if heading/range best-checkpoint mismatch is common and the axiswise oracle improves over the best single checkpoint.",
            "- The axiswise oracle is label-assisted and only measures headroom; a publishable method still needs a non-label predictor or learned readout.",
        ]
    )
    (output_dir / "phase100_h0_h1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(args.r5_per_sample))[: args.max_pairs]
    if not rows:
        raise ValueError(f"No rows read from {args.r5_per_sample}")
    case_ids = [item.strip() for item in args.case_list.split(",") if item.strip()]

    predictions: dict[str, list[tuple[float, float]]] = {}
    prediction_paths: dict[str, str] = {}
    missing: dict[str, str] = {}
    for case_id in case_ids:
        path = case_prediction_path(args, case_id)
        if path.exists():
            preds = read_prediction_file(path, args.max_pairs)
            if len(preds) < len(rows):
                raise ValueError(f"Prediction file has {len(preds)} rows but need {len(rows)}: {path}")
            predictions[case_id] = preds
            prediction_paths[case_id] = str(path)
        else:
            missing[case_id] = str(path)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2, sort_keys=True))

    range_span = infer_range_span(rows)
    per_sample = build_per_sample(rows, predictions, case_ids, range_span)
    win_rows: list[dict[str, Any]] = []
    win_rows.extend(
        win_summary(
            per_sample,
            case_ids,
            "best_final_case",
            "best_final_error",
            "best_final_margin_to_second",
            "baseline_final_error",
        )
    )
    win_rows.extend(
        win_summary(
            per_sample,
            case_ids,
            "best_heading_case",
            "best_heading_angle_rel_error",
            "best_heading_margin_to_second",
            "baseline_final_angle_rel_error",
        )
    )
    win_rows.extend(
        win_summary(
            per_sample,
            case_ids,
            "best_range_case",
            "best_range_distance_rel_error",
            "best_range_margin_to_second",
            "baseline_final_distance_rel_error",
        )
    )

    group_keys = [
        "true_heading_bin_idx",
        "pred_heading_bin_idx",
        "true_range_abs_bucket",
        "pred_range_abs_bucket",
        "true_range_sign",
        "pred_true_heading_bin_mismatch",
        "pred_true_range_abs_bucket_mismatch",
        "pred_true_range_sign_mismatch",
        "traj_best_case",
        "traj_final_error_rank",
    ]
    group_rows: list[dict[str, Any]] = []
    for key in group_keys:
        if key in per_sample[0]:
            group_rows.extend(group_summary(per_sample, key))
    group_rows = sorted(
        group_rows,
        key=lambda row: (
            -float(row["count"]),
            -float(row["axis_mismatch_frac"]),
            -float(row["mean_final_minus_best_checkpoint"]),
        ),
    )

    summary = summarize(per_sample, case_ids, prediction_paths)
    summary["output_dir"] = str(output_dir)
    summary["r5_per_sample"] = args.r5_per_sample
    summary["prediction_root"] = args.prediction_root
    summary["range_span"] = range_span
    summary["elapsed_sec"] = round(time.time() - started, 3)

    write_csv(output_dir / "phase100_h0_per_sample_best_checkpoint.csv", per_sample)
    write_csv(output_dir / "phase100_h0_checkpoint_axis_win_summary.csv", win_rows)
    write_csv(output_dir / "phase100_h1_regime_summary.csv", group_rows)
    (output_dir / "phase100_h0_h1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir, summary, win_rows, group_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
