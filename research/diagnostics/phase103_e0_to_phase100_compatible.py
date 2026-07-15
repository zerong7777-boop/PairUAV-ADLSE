#!/usr/bin/env python3
"""Convert Phase103-E0 train-hash artifacts to a Phase100-compatible table.

This lets the Phase103-R0/R1/R2a lab diagnostics run on the train4096 hash
subset without regenerating predictions. It only normalizes columns and derives
diagnostic buckets from already-generated train labels/predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


CASE_ORDER = [
    "step050000",
    "step100000",
    "step150000",
    "step200000",
    "step250000",
    "step300000",
    "step350000",
    "step400000",
    "step450000",
    "final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-features", required=True)
    parser.add_argument("--e0-case-manifest", required=True)
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


def wrap_heading(deg: float) -> float:
    value = deg % 360.0
    return value + 360.0 if value < 0.0 else value


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def heading_bin_idx(deg: float) -> int:
    return int(min(7, max(0, math.floor(wrap_heading(deg) / 45.0))))


def heading_label(deg: float) -> str:
    idx = heading_bin_idx(deg)
    return f"h_{idx:02d}_{idx * 45}_{(idx + 1) * 45}"


def range_abs_bucket(value: float) -> str:
    v = abs(value)
    if v <= 1:
        return "d_01_le_1"
    if v <= 5:
        return "d_02_le_5"
    if v <= 10:
        return "d_03_le_10"
    if v <= 25:
        return "d_04_le_25"
    if v <= 50:
        return "d_05_le_50"
    if v <= 100:
        return "d_06_le_100"
    return "d_07_gt_100"


def range_sign(value: float) -> str:
    return "neg" if value < 0 else "pos"


def range_signed_label(value: float) -> str:
    return f"{range_sign(value)}_{range_abs_bucket(value)}"


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def safe_median(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(statistics.median(clean)) if clean else math.nan


def build_case_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out[str(row["case_id"])] = row
    return out


def convert_rows(features: list[dict[str, str]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    case_ids = [case for case in CASE_ORDER if case in manifest or case == "final"]
    output: list[dict[str, Any]] = []
    for row in features:
        true_h = f(row, "true_heading_deg")
        true_r = f(row, "true_range")
        pred_h = f(row, "pred_heading_final")
        pred_r = f(row, "pred_range_final")
        best_final = str(row.get("best_final_case", ""))
        best_heading = str(row.get("best_heading_case", ""))
        best_range = str(row.get("best_range_case", ""))
        true_h_bin = heading_bin_idx(true_h)
        pred_h_bin = heading_bin_idx(pred_h)
        true_abs_bucket = range_abs_bucket(true_r)
        pred_abs_bucket = range_abs_bucket(pred_r)
        true_sign = range_sign(true_r)
        pred_sign = range_sign(pred_r)
        final_errors = [(case, f(row, f"final_error_{case}")) for case in case_ids]
        final_rank = [case for case, _ in sorted(final_errors, key=lambda item: item[1])].index("final") + 1
        converted = dict(row)
        converted.update(
            {
                "pred_heading_deg": pred_h,
                "pred_range": pred_r,
                "heading_abs_error_deg": abs(wrap_angle_diff_deg(pred_h, true_h)),
                "angle_rel_error": abs(wrap_angle_diff_deg(pred_h, true_h)) / 180.0,
                "range_abs_error": abs(pred_r - true_r),
                "final_error_proxy": f(row, "baseline_final_error"),
                "heading_label": heading_label(true_h),
                "range_abs_label": true_abs_bucket,
                "range_signed_label": range_signed_label(true_r),
                "true_heading_bin_idx": true_h_bin,
                "pred_heading_bin_idx": pred_h_bin,
                "pred_true_heading_bin_mismatch": int(true_h_bin != pred_h_bin),
                "true_range_abs_bucket": true_abs_bucket,
                "pred_range_abs_bucket": pred_abs_bucket,
                "pred_true_range_abs_bucket_mismatch": int(true_abs_bucket != pred_abs_bucket),
                "true_range_sign": true_sign,
                "pred_range_sign": pred_sign,
                "pred_true_range_sign_mismatch": int(true_sign != pred_sign),
                "traj_case_count": len(case_ids),
                "traj_present_cases": ",".join(case_ids),
                "traj_heading_circ_std_deg": f(row, "full_heading_circ_std_deg"),
                "traj_heading_max_drift_to_final_deg": max(
                    abs(f(row, f"delta_heading_{case}_minus_final", 0.0)) for case in case_ids
                ),
                "traj_heading_mean_drift_to_final_deg": safe_mean(
                    [abs(f(row, f"delta_heading_{case}_minus_final", 0.0)) for case in case_ids]
                ),
                "traj_range_std": f(row, "full_range_std"),
                "traj_range_span": max(f(row, f"pred_range_{case}") for case in case_ids)
                - min(f(row, f"pred_range_{case}") for case in case_ids),
                "traj_range_max_drift_to_final": max(
                    abs(f(row, f"delta_range_{case}_minus_final", 0.0)) for case in case_ids
                ),
                "traj_range_mean_drift_to_final": safe_mean(
                    [abs(f(row, f"delta_range_{case}_minus_final", 0.0)) for case in case_ids]
                ),
                "traj_best_case": best_final,
                "traj_best_error_proxy": f(row, "best_final_error"),
                "traj_final_error_from_predictions": f(row, "baseline_final_error"),
                "traj_final_minus_best_error": f(row, "baseline_minus_best_final_error"),
                "traj_final_error_rank": final_rank,
                "best_final_family": case_family(best_final),
                "best_final_train_step": case_step(best_final),
                "best_heading_family": case_family(best_heading),
                "best_heading_train_step": case_step(best_heading),
                "best_range_family": case_family(best_range),
                "best_range_train_step": case_step(best_range),
                "heading_range_best_case_mismatch": int(best_heading != best_range),
                "best_checkpoint_axes_aligned": int(best_final == best_heading == best_range),
                "axiswise_oracle_heading_case": best_heading,
                "axiswise_oracle_range_case": best_range,
                "best_checkpoint_minus_axiswise_oracle": f(row, "best_final_error") - f(row, "axiswise_oracle_error"),
            }
        )
        for case in ("step400000", "step450000"):
            if case in case_ids:
                converted[f"traj_{case}_heading_to_final_deg"] = f(row, f"delta_heading_{case}_minus_final")
                converted[f"traj_{case}_range_to_final"] = f(row, f"delta_range_{case}_minus_final")
                converted[f"traj_final_minus_{case}_error"] = (
                    f(row, "baseline_final_error") - f(row, f"final_error_{case}")
                )
        output.append(converted)
    return output


def win_summary(rows: list[dict[str, Any]], case_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    configs = [
        ("best_final_case", "best_final_error", "baseline_minus_best_final_error"),
        ("best_heading_case", "best_heading_angle_rel_error", "baseline_minus_best_heading_angle"),
        ("best_range_case", "best_range_distance_rel_error", "baseline_minus_best_range_distance"),
    ]
    for axis_key, selected_error_key, improvement_key in configs:
        counts = Counter(str(row.get(axis_key, "")) for row in rows)
        for case in case_ids:
            selected = [row for row in rows if str(row.get(axis_key, "")) == case]
            out.append(
                {
                    "axis": axis_key,
                    "case_id": case,
                    "family": case_family(case),
                    "train_step": case_step(case),
                    "win_count": counts.get(case, 0),
                    "win_frac": counts.get(case, 0) / max(len(rows), 1),
                    "mean_selected_error": safe_mean([f(row, selected_error_key) for row in selected]),
                    "median_margin_to_second": safe_median(
                        [
                            f(
                                row,
                                {
                                    "best_final_case": "best_final_margin_to_second",
                                    "best_heading_case": "best_heading_margin_to_second",
                                    "best_range_case": "best_range_margin_to_second",
                                }[axis_key],
                            )
                            for row in selected
                        ]
                    ),
                    "mean_margin_to_second": safe_mean(
                        [
                            f(
                                row,
                                {
                                    "best_final_case": "best_final_margin_to_second",
                                    "best_heading_case": "best_heading_margin_to_second",
                                    "best_range_case": "best_range_margin_to_second",
                                }[axis_key],
                            )
                            for row in selected
                        ]
                    ),
                    "mean_baseline_minus_selected": safe_mean([f(row, improvement_key) for row in selected]),
                }
            )
    return out


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = read_csv(Path(args.e0_features))
    manifest = build_case_manifest(read_csv(Path(args.e0_case_manifest)))
    rows = convert_rows(features, manifest)
    case_ids = [case for case in CASE_ORDER if case in manifest or case == "final"]
    axis_summary = win_summary(rows, case_ids)
    summary = {
        "phase": "phase103_e0_to_phase100_compatible",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "e0_features": args.e0_features,
        "e0_case_manifest": args.e0_case_manifest,
        "output_dir": str(output_dir),
        "rows": len(rows),
        "case_ids": case_ids,
        "best_final_case_counts": dict(sorted(Counter(str(row.get("best_final_case", "")) for row in rows).items())),
        "best_heading_case_counts": dict(sorted(Counter(str(row.get("best_heading_case", "")) for row in rows).items())),
        "best_range_case_counts": dict(sorted(Counter(str(row.get("best_range_case", "")) for row in rows).items())),
        "axis_mismatch_count": sum(int(f(row, "heading_range_best_case_mismatch", 0.0)) for row in rows),
        "axis_mismatch_frac": safe_mean([f(row, "heading_range_best_case_mismatch") for row in rows]),
        "baseline_final_error": safe_mean([f(row, "baseline_final_error") for row in rows]),
        "oracle_best_checkpoint_error": safe_mean([f(row, "best_final_error") for row in rows]),
        "axiswise_oracle_error": safe_mean([f(row, "axiswise_oracle_error") for row in rows]),
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "train_hash_diagnostic": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_csv(output_dir / "phase103_train_hash_phase100_compatible_per_sample.csv", rows)
    write_csv(output_dir / "phase103_train_hash_checkpoint_axis_win_summary.csv", axis_summary)
    (output_dir / "phase103_train_hash_phase100_compatible_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
