#!/usr/bin/env python3
"""Phase103-R1 axis learning clock audit.

This fixed-val diagnostic tests whether heading and range choose different
checkpoints along the H8 training trajectory. It consumes existing Phase100
tabular artifacts only. It is not a deployable postprocess rule and does not
use hidden official-test labels or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MIN_GROUP_COUNT = 10
DEFAULT_BOOTSTRAP = 500
BUCKET_KEYS = [
    "true_heading_bin_idx",
    "pred_heading_bin_idx",
    "true_range_abs_bucket",
    "pred_range_abs_bucket",
    "true_range_sign",
    "pred_true_heading_bin_mismatch",
    "pred_true_range_abs_bucket_mismatch",
    "pred_true_range_sign_mismatch",
    "best_final_family",
    "best_final_case",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase100-per-sample", required=True)
    parser.add_argument("--phase100-axis-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-group-count", type=int, default=DEFAULT_MIN_GROUP_COUNT)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=1031)
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
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def case_step(case_id: str) -> int:
    if case_id == "final":
        return 459999
    digits = "".join(ch for ch in str(case_id) if ch.isdigit())
    return int(digits) if digits else 0


def case_family(case_id: str) -> str:
    if case_id == "final":
        return "final"
    step = case_step(case_id)
    if step <= 200000:
        return "early_050_200k"
    if step <= 300000:
        return "mid_250_300k"
    return "late_350_450k"


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def safe_median(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(statistics.median(clean)) if clean else math.nan


def quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def exact_sign_test_two_sided(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return math.nan
    if n > 1000:
        # Normal approximation for large train/hash audits avoids overflow from
        # exact binomial coefficients while preserving the diagnostic gate.
        z_abs = abs(pos - neg) / math.sqrt(n)
        return math.erfc(z_abs / math.sqrt(2.0))
    k = min(pos, neg)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * prob)


def bootstrap_ci(values: list[float], rng: random.Random, n_boot: int) -> dict[str, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean or n_boot <= 0:
        return {"mean_ci_low": math.nan, "mean_ci_high": math.nan}
    means: list[float] = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(sample) / len(sample))
    return {
        "mean_ci_low": quantile(means, 0.025),
        "mean_ci_high": quantile(means, 0.975),
    }


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        best_heading = row.get("best_heading_case", "")
        best_range = row.get("best_range_case", "")
        heading_step = int(f(row, "best_heading_train_step", case_step(best_heading)))
        range_step = int(f(row, "best_range_train_step", case_step(best_range)))
        lag = range_step - heading_step
        enriched = dict(row)
        enriched.update(
            {
                "best_heading_family_derived": case_family(best_heading),
                "best_range_family_derived": case_family(best_range),
                "best_final_family_derived": case_family(row.get("best_final_case", "")),
                "axis_step_lag": lag,
                "axis_abs_step_lag": abs(lag),
                "range_later": int(lag > 0),
                "heading_later": int(lag < 0),
                "same_axis_step": int(lag == 0),
                "axis_mismatch_derived": int(best_heading != best_range),
                "axis_family_pair": f"{case_family(best_heading)}->{case_family(best_range)}",
                "axis_case_pair": f"{best_heading}->{best_range}",
            }
        )
        out.append(enriched)
    return out


def summarize_axis(rows: list[dict[str, Any]], rng: random.Random, n_boot: int) -> dict[str, Any]:
    lags = [f(row, "axis_step_lag") for row in rows]
    range_later = sum(int(f(row, "range_later", 0.0)) for row in rows)
    heading_later = sum(int(f(row, "heading_later", 0.0)) for row in rows)
    same = sum(int(f(row, "same_axis_step", 0.0)) for row in rows)
    mismatch = sum(int(f(row, "axis_mismatch_derived", 0.0)) for row in rows)
    mean_lag_ci = bootstrap_ci(lags, rng, n_boot)
    return {
        "summary_key": "all_samples",
        "summary_value": "all",
        "count": len(rows),
        "mean_axis_step_lag": safe_mean(lags),
        "median_axis_step_lag": safe_median(lags),
        "mean_abs_axis_step_lag": safe_mean([abs(value) for value in lags]),
        "range_later_count": range_later,
        "range_later_frac": range_later / max(len(rows), 1),
        "heading_later_count": heading_later,
        "heading_later_frac": heading_later / max(len(rows), 1),
        "same_axis_step_count": same,
        "same_axis_step_frac": same / max(len(rows), 1),
        "axis_mismatch_count": mismatch,
        "axis_mismatch_frac": mismatch / max(len(rows), 1),
        "sign_test_two_sided_p": exact_sign_test_two_sided(range_later, heading_later),
        "mean_axis_step_lag_ci_low": mean_lag_ci["mean_ci_low"],
        "mean_axis_step_lag_ci_high": mean_lag_ci["mean_ci_high"],
        "best_heading_case_counts": json.dumps(
            dict(sorted(Counter(str(row.get("best_heading_case", "")) for row in rows).items())),
            sort_keys=True,
        ),
        "best_range_case_counts": json.dumps(
            dict(sorted(Counter(str(row.get("best_range_case", "")) for row in rows).items())),
            sort_keys=True,
        ),
        "axis_family_pair_counts": json.dumps(
            dict(sorted(Counter(str(row.get("axis_family_pair", "")) for row in rows).items())),
            sort_keys=True,
        ),
    }


def summarize_axis_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["axis_case_pair"])].append(row)
    out: list[dict[str, Any]] = []
    for pair, selected in groups.items():
        out.append(
            {
                "axis_case_pair": pair,
                "count": len(selected),
                "frac": len(selected) / max(len(rows), 1),
                "mean_axis_step_lag": safe_mean([f(row, "axis_step_lag") for row in selected]),
                "mean_final_error": safe_mean([f(row, "final_error_proxy") for row in selected]),
                "mean_axiswise_headroom": safe_mean([f(row, "baseline_minus_axiswise_oracle") for row in selected]),
                "mean_best_heading_gain": safe_mean([f(row, "baseline_minus_best_heading_angle") for row in selected]),
                "mean_best_range_gain": safe_mean([f(row, "baseline_minus_best_range_distance") for row in selected]),
            }
        )
    return sorted(out, key=lambda row: (-int(row["count"]), str(row["axis_case_pair"])))


def summarize_buckets(rows: list[dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in BUCKET_KEYS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if key in row:
                groups[str(row.get(key, ""))].append(row)
        for value, selected in groups.items():
            if value == "" or len(selected) < min_count:
                continue
            final_wins = sum(1 for row in selected if row.get("best_final_case") == "final")
            range_later = sum(int(f(row, "range_later", 0.0)) for row in selected)
            heading_later = sum(int(f(row, "heading_later", 0.0)) for row in selected)
            mismatch = sum(int(f(row, "axis_mismatch_derived", 0.0)) for row in selected)
            out.append(
                {
                    "bucket_key": key,
                    "bucket_value": value,
                    "count": len(selected),
                    "frac": len(selected) / max(len(rows), 1),
                    "mean_axis_step_lag": safe_mean([f(row, "axis_step_lag") for row in selected]),
                    "median_axis_step_lag": safe_median([f(row, "axis_step_lag") for row in selected]),
                    "range_later_frac": range_later / max(len(selected), 1),
                    "heading_later_frac": heading_later / max(len(selected), 1),
                    "axis_mismatch_frac": mismatch / max(len(selected), 1),
                    "final_winner_frac": final_wins / max(len(selected), 1),
                    "mean_final_error": safe_mean([f(row, "final_error_proxy") for row in selected]),
                    "mean_axiswise_headroom": safe_mean([f(row, "baseline_minus_axiswise_oracle") for row in selected]),
                    "best_heading_case_counts": json.dumps(
                        dict(sorted(Counter(str(row.get("best_heading_case", "")) for row in selected).items())),
                        sort_keys=True,
                    ),
                    "best_range_case_counts": json.dumps(
                        dict(sorted(Counter(str(row.get("best_range_case", "")) for row in selected).items())),
                        sort_keys=True,
                    ),
                }
            )
    return sorted(
        out,
        key=lambda row: (
            -abs(float(row["mean_axis_step_lag"])),
            -float(row["axis_mismatch_frac"]),
            -int(row["count"]),
            str(row["bucket_key"]),
            str(row["bucket_value"]),
        ),
    )


def convert_axis_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        converted = dict(row)
        converted["curve_source"] = "axis_win_summary_selected_error"
        converted["limitation"] = "no_full_per_checkpoint_all_sample_error_curve_in_phase100"
        converted["clock_axis"] = {
            "best_final_case": "joint_final_score",
            "best_heading_case": "heading",
            "best_range_case": "range",
        }.get(str(row.get("axis", "")), str(row.get("axis", "")))
        out.append(converted)
    return out


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    axis_summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase103-R1 Axis Learning Clock Audit",
        "",
        "Fixed-val811 mechanism diagnostic over existing Phase100 trajectory artifacts.",
        "",
        "## Run",
        "",
        f"- per_sample: `{summary['phase100_per_sample']}`",
        f"- axis_summary: `{summary['phase100_axis_summary']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- rows: `{summary['rows']}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "",
        "## Core Axis Clock",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| axis mismatch | {axis_summary['axis_mismatch_count']} ({float(axis_summary['axis_mismatch_frac']):.4f}) |",
        f"| mean range-heading step lag | {float(axis_summary['mean_axis_step_lag']):.4f} |",
        f"| median range-heading step lag | {float(axis_summary['median_axis_step_lag']):.4f} |",
        f"| mean abs step lag | {float(axis_summary['mean_abs_axis_step_lag']):.4f} |",
        f"| range later | {axis_summary['range_later_count']} ({float(axis_summary['range_later_frac']):.4f}) |",
        f"| heading later | {axis_summary['heading_later_count']} ({float(axis_summary['heading_later_frac']):.4f}) |",
        f"| same step | {axis_summary['same_axis_step_count']} ({float(axis_summary['same_axis_step_frac']):.4f}) |",
        f"| sign-test p | {float(axis_summary['sign_test_two_sided_p']):.8g} |",
        "",
        "## Top Axis Case Pairs",
        "",
        "| pair | count | frac | mean lag | final error | axis headroom |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pair_rows[:15]:
        lines.append(
            f"| `{row['axis_case_pair']}` | {row['count']} | {float(row['frac']):.4f} | "
            f"{float(row['mean_axis_step_lag']):.2f} | {float(row['mean_final_error']):.8g} | "
            f"{float(row['mean_axiswise_headroom']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Strong Bucket Clocks",
            "",
            "| bucket | value | count | mean lag | range later | heading later | mismatch | headroom |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bucket_rows[:20]:
        lines.append(
            f"| `{row['bucket_key']}` | `{row['bucket_value']}` | {row['count']} | "
            f"{float(row['mean_axis_step_lag']):.2f} | {float(row['range_later_frac']):.3f} | "
            f"{float(row['heading_later_frac']):.3f} | {float(row['axis_mismatch_frac']):.3f} | "
            f"{float(row['mean_axiswise_headroom']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This run uses Phase100 winner distributions and selected-error summaries. "
            "Phase100 does not contain a full per-checkpoint all-sample error matrix, "
            "so `phase103_r1_checkpoint_axis_error_curves.csv` is a winner-distribution "
            "clock surface, not a complete error curve.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = Path(args.phase100_per_sample)
    axis_summary_path = Path(args.phase100_axis_summary)
    rows = enrich_rows(read_csv(per_sample_path))
    axis_rows = read_csv(axis_summary_path)

    axis_summary = summarize_axis(rows, rng, args.bootstrap)
    axis_summary_rows = [axis_summary]
    pair_rows = summarize_axis_pairs(rows)
    bucket_rows = summarize_buckets(rows, args.min_group_count)
    curve_rows = convert_axis_summary(axis_rows)

    summary = {
        "phase": "phase103_r1_axis_learning_clock_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase100_per_sample": str(per_sample_path),
        "phase100_axis_summary": str(axis_summary_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "min_group_count": args.min_group_count,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "axis_summary": axis_summary,
        "top_axis_pairs": pair_rows[:10],
        "top_bucket_clocks": bucket_rows[:10],
        "curve_source": "axis_win_summary_selected_error",
        "limitation": "no_full_per_checkpoint_all_sample_error_curve_in_phase100",
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "diagnostic_only": True,
        "elapsed_sec": round(time.time() - started, 3),
    }

    write_csv(output_dir / "phase103_r1_axis_learning_clock_summary.csv", axis_summary_rows)
    write_csv(output_dir / "phase103_r1_checkpoint_axis_error_curves.csv", curve_rows)
    write_csv(output_dir / "phase103_r1_bucket_axis_clock_summary.csv", bucket_rows)
    write_csv(output_dir / "phase103_r1_axis_pair_lag_summary.csv", pair_rows)
    (output_dir / "phase103_r1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_r1_summary.md", summary, axis_summary, pair_rows, bucket_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
