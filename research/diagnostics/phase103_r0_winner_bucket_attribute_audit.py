#!/usr/bin/env python3
"""Phase103-R0 winner-bucket attribute audit.

This fixed-val diagnostic asks whether checkpoint winner buckets have
interpretable pose/regime/error attributes. It is not a deployable postprocess
rule and does not use hidden official-test labels or leaderboard feedback.
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


CATEGORICAL_ATTRS = [
    "true_heading_bin_idx",
    "pred_heading_bin_idx",
    "true_range_abs_bucket",
    "pred_range_abs_bucket",
    "true_range_sign",
    "pred_range_sign",
    "heading_label",
    "range_abs_label",
    "range_signed_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase100-per-sample", required=True)
    parser.add_argument("--output-dir", required=True)
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


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


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


def case_step(case_id: str) -> int:
    if case_id == "final":
        return 459999
    digits = "".join(ch for ch in case_id if ch.isdigit())
    if not digits:
        return 0
    return int(digits)


def case_family(case_id: str) -> str:
    if case_id == "final":
        return "final"
    step = case_step(case_id)
    if step <= 200000:
        return "early_050_200k"
    if step <= 300000:
        return "mid_250_300k"
    return "late_350_450k"


def add_derived(rows: list[dict[str, str]]) -> dict[str, float]:
    axis_headrooms = [f(row, "baseline_minus_axiswise_oracle") for row in rows]
    if all(not math.isfinite(value) for value in axis_headrooms):
        axis_headrooms = [f(row, "baseline_minus_best_final_error") for row in rows]
    traj_instabilities = [
        max(
            f(row, "traj_heading_circ_std_deg", 0.0),
            f(row, "traj_range_std", 0.0),
            f(row, "traj_range_abs_std", 0.0),
        )
        for row in rows
    ]
    final_errors = [f(row, "final_error_proxy") for row in rows]
    thresholds = {
        "axis_headroom_q75": quantile(axis_headrooms, 0.75),
        "traj_instability_q75": quantile(traj_instabilities, 0.75),
        "final_error_q90": quantile(final_errors, 0.90),
    }
    for row, axis_headroom, instability in zip(rows, axis_headrooms, traj_instabilities):
        best_final = str(row.get("best_final_case", row.get("teacher_final_case", "")))
        best_heading = str(row.get("best_heading_case", row.get("teacher_heading_case", "")))
        best_range = str(row.get("best_range_case", row.get("teacher_range_case", "")))
        row["best_final_family_derived"] = case_family(best_final)
        row["best_heading_family_derived"] = case_family(best_heading)
        row["best_range_family_derived"] = case_family(best_range)
        row["final_wins"] = str(int(best_final == "final"))
        row["late_wins"] = str(int(case_family(best_final) in ("late_350_450k", "final") and best_final != "final"))
        row["mid_or_early_wins"] = str(int(case_family(best_final) in ("early_050_200k", "mid_250_300k")))
        row["axis_mismatch"] = str(int(best_heading != best_range))
        row["high_axiswise_headroom_q75"] = str(int(math.isfinite(axis_headroom) and axis_headroom >= thresholds["axis_headroom_q75"]))
        row["high_trajectory_instability_q75"] = str(
            int(math.isfinite(instability) and instability >= thresholds["traj_instability_q75"])
        )
        row["top_final_error_q90"] = str(int(f(row, "final_error_proxy") >= thresholds["final_error_q90"]))
        row["heading_signed_error_deg"] = str(wrap_angle_diff_deg(f(row, "pred_heading_deg"), f(row, "true_heading_deg")))
        row["range_signed_error"] = str(f(row, "pred_range") - f(row, "true_range"))
    return thresholds


def group_specs(rows: list[dict[str, str]]) -> list[tuple[str, str, list[dict[str, str]]]]:
    specs: list[tuple[str, str, list[dict[str, str]]]] = []
    group_keys = [
        "best_final_case",
        "best_final_family_derived",
        "best_heading_case",
        "best_range_case",
        "axis_mismatch",
        "high_axiswise_headroom_q75",
        "high_trajectory_instability_q75",
        "top_final_error_q90",
    ]
    for key in group_keys:
        values = sorted({str(row.get(key, "")) for row in rows})
        for value in values:
            selected = [row for row in rows if str(row.get(key, "")) == value]
            if selected:
                specs.append((key, value, selected))
    return specs


def summarize_winner_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_key, group_value, selected in group_specs(rows):
        out.append(
            {
                "group_key": group_key,
                "group_value": group_value,
                "count": len(selected),
                "frac": len(selected) / max(len(rows), 1),
                "mean_final_error": safe_mean([f(row, "final_error_proxy") for row in selected]),
                "mean_baseline_minus_best_final": safe_mean(
                    [f(row, "baseline_minus_best_final_error") for row in selected]
                ),
                "mean_baseline_minus_axiswise": safe_mean(
                    [f(row, "baseline_minus_axiswise_oracle") for row in selected]
                ),
                "mean_traj_heading_circ_std_deg": safe_mean(
                    [f(row, "traj_heading_circ_std_deg") for row in selected]
                ),
                "mean_heading_signed_error_deg": safe_mean(
                    [f(row, "heading_signed_error_deg") for row in selected]
                ),
                "mean_range_signed_error": safe_mean([f(row, "range_signed_error") for row in selected]),
                "axis_mismatch_frac": safe_mean([f(row, "axis_mismatch") for row in selected]),
            }
        )
    return sorted(out, key=lambda row: (str(row["group_key"]), -int(row["count"]), str(row["group_value"])))


def summarize_enrichment(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    global_counts = {attr: Counter(str(row.get(attr, "")) for row in rows) for attr in CATEGORICAL_ATTRS}
    for group_key, group_value, selected in group_specs(rows):
        for attr in CATEGORICAL_ATTRS:
            selected_counts = Counter(str(row.get(attr, "")) for row in selected)
            for attr_value, selected_count in selected_counts.items():
                if attr_value == "":
                    continue
                global_count = global_counts[attr][attr_value]
                selected_frac = selected_count / max(len(selected), 1)
                global_frac = global_count / max(len(rows), 1)
                out.append(
                    {
                        "group_key": group_key,
                        "group_value": group_value,
                        "attribute": attr,
                        "attribute_value": attr_value,
                        "group_count": len(selected),
                        "selected_count": selected_count,
                        "selected_frac": selected_frac,
                        "global_count": global_count,
                        "global_frac": global_frac,
                        "enrichment": selected_frac / global_frac if global_frac > 0.0 else math.nan,
                    }
                )
    return sorted(
        out,
        key=lambda row: (
            str(row["group_key"]),
            str(row["group_value"]),
            -abs(float(row["enrichment"]) - 1.0) if math.isfinite(float(row["enrichment"])) else 0.0,
        ),
    )


def summarize_axis_mismatch(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("axis_mismatch") == "1"]
    same = [row for row in rows if row.get("axis_mismatch") == "0"]
    out: list[dict[str, Any]] = []
    for attr in CATEGORICAL_ATTRS:
        mismatch_counts = Counter(str(row.get(attr, "")) for row in selected)
        same_counts = Counter(str(row.get(attr, "")) for row in same)
        values = sorted(set(mismatch_counts) | set(same_counts))
        for value in values:
            if value == "":
                continue
            mismatch_frac = mismatch_counts[value] / max(len(selected), 1)
            same_frac = same_counts[value] / max(len(same), 1)
            out.append(
                {
                    "attribute": attr,
                    "attribute_value": value,
                    "mismatch_count": mismatch_counts[value],
                    "mismatch_frac": mismatch_frac,
                    "same_count": same_counts[value],
                    "same_frac": same_frac,
                    "mismatch_over_same_enrichment": mismatch_frac / same_frac if same_frac > 0.0 else math.nan,
                }
            )
    return sorted(out, key=lambda row: -abs(float(row["mismatch_over_same_enrichment"]) - 1.0) if math.isfinite(float(row["mismatch_over_same_enrichment"])) else 0.0)


def summarize_bias(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_key, group_value, selected in group_specs(rows):
        heading_signed = [f(row, "heading_signed_error_deg") for row in selected]
        range_signed = [f(row, "range_signed_error") for row in selected]
        out.append(
            {
                "group_key": group_key,
                "group_value": group_value,
                "count": len(selected),
                "mean_heading_signed_error_deg": safe_mean(heading_signed),
                "median_heading_signed_error_deg": safe_median(heading_signed),
                "mean_abs_heading_error_deg": safe_mean([abs(value) for value in heading_signed]),
                "heading_positive_frac": safe_mean([1.0 if value > 0 else 0.0 for value in heading_signed]),
                "mean_range_signed_error": safe_mean(range_signed),
                "median_range_signed_error": safe_median(range_signed),
                "mean_abs_range_error": safe_mean([abs(value) for value in range_signed]),
                "range_positive_frac": safe_mean([1.0 if value > 0 else 0.0 for value in range_signed]),
            }
        )
    return sorted(out, key=lambda row: (str(row["group_key"]), -int(row["count"]), str(row["group_value"])))


def top_enrichments(enrichment_rows: list[dict[str, Any]], min_group_count: int = 20, topn: int = 20) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in enrichment_rows
        if int(row["group_count"]) >= min_group_count
        and math.isfinite(float(row["enrichment"]))
        and float(row["global_frac"]) >= 0.02
    ]
    return sorted(filtered, key=lambda row: -abs(float(row["enrichment"]) - 1.0))[:topn]


def write_markdown(path: Path, summary: dict[str, Any], group_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase103-R0 Winner-Bucket Attribute Audit",
        "",
        "Fixed-val811 mechanism diagnostic. Not a deployable postprocess rule.",
        "",
        "## Run",
        "",
        f"- input: `{summary['phase100_per_sample']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- rows: `{summary['rows']}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "",
        "## Group Counts",
        "",
        "| group | value | count | frac | mean final error | mean axis headroom | axis mismatch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        if row["group_key"] not in ("best_final_case", "best_final_family_derived", "axis_mismatch", "high_axiswise_headroom_q75"):
            continue
        lines.append(
            f"| `{row['group_key']}` | `{row['group_value']}` | {row['count']} | "
            f"{float(row['frac']):.4f} | {float(row['mean_final_error']):.8g} | "
            f"{float(row['mean_baseline_minus_axiswise']):.8g} | {float(row['axis_mismatch_frac']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Attribute Enrichments",
            "",
            "| group | value | attribute | attr value | group count | selected frac | global frac | enrichment |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows[:20]:
        lines.append(
            f"| `{row['group_key']}` | `{row['group_value']}` | `{row['attribute']}` | "
            f"`{row['attribute_value']}` | {row['group_count']} | {float(row['selected_frac']):.4f} | "
            f"{float(row['global_frac']):.4f} | {float(row['enrichment']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.phase100_per_sample))
    thresholds = add_derived(rows)
    group_rows = summarize_winner_groups(rows)
    enrichment_rows = summarize_enrichment(rows)
    mismatch_rows = summarize_axis_mismatch(rows)
    bias_rows = summarize_bias(rows)
    top_rows = top_enrichments(enrichment_rows)
    summary = {
        "phase": "phase103_r0_winner_bucket_attribute_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase100_per_sample": str(Path(args.phase100_per_sample)),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "thresholds": thresholds,
        "best_final_case_counts": dict(sorted(Counter(str(row.get("best_final_case", "")) for row in rows).items())),
        "best_final_family_counts": dict(sorted(Counter(str(row.get("best_final_family_derived", "")) for row in rows).items())),
        "axis_mismatch_count": sum(1 for row in rows if row.get("axis_mismatch") == "1"),
        "axis_mismatch_frac": safe_mean([f(row, "axis_mismatch") for row in rows]),
        "top_enrichment_rows": top_rows[:10],
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "diagnostic_only": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_csv(output_dir / "phase103_r0_winner_bucket_attribute_summary.csv", group_rows)
    write_csv(output_dir / "phase103_r0_axis_mismatch_attribute_summary.csv", mismatch_rows)
    write_csv(output_dir / "phase103_r0_final_bias_direction_summary.csv", bias_rows)
    write_csv(output_dir / "phase103_r0_bucket_enrichment.csv", enrichment_rows)
    (output_dir / "phase103_r0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_r0_summary.md", summary, group_rows, top_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
