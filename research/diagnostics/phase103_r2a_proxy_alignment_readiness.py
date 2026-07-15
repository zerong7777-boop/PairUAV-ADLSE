#!/usr/bin/env python3
"""Phase103-R2a proxy alignment and full-R2 readiness audit.

This lab-side diagnostic asks whether Phase103-R1 axis learning-clock effects
align with available prediction-space proxies. It also records whether the
feature/head swap artifacts required for the full Phase103-R2 probe are present.
It is not the complete representation-readout phase-lag experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


PROXY_COLUMNS = [
    "traj_heading_circ_std_deg",
    "traj_heading_max_drift_to_final_deg",
    "traj_heading_mean_drift_to_final_deg",
    "traj_range_std",
    "traj_range_span",
    "traj_range_max_drift_to_final",
    "traj_range_mean_drift_to_final",
    "heading_margin_l6",
    "range_abs_margin_l11",
    "range_signed_margin_l12",
    "combined_min_margin",
    "true_heading_boundary_dist_deg",
    "pred_heading_boundary_dist_deg",
    "true_range_abs_threshold_dist",
    "pred_range_abs_threshold_dist",
    "pred_true_heading_bin_mismatch",
    "pred_true_range_abs_bucket_mismatch",
    "pred_true_range_sign_mismatch",
    "heading_centroid_mismatch",
    "range_abs_centroid_mismatch",
    "range_signed_centroid_mismatch",
]

TARGET_COLUMNS = [
    "axis_step_lag",
    "axis_abs_step_lag",
    "axis_mismatch_derived",
    "baseline_minus_axiswise_oracle",
    "baseline_minus_best_final_error",
    "final_error_proxy",
]

EXPECTED_FULL_R2_FILES = [
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase102_pasc_mechanism_v1/r5_repr_readout_attribution_phase89_h8_val811_20260622/phase102_r5_attribution_per_sample.csv",
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase102_pasc_mechanism_v1/r5_repr_readout_attribution_phase89_h8_val811_20260622/phase102_r5_hybrid_per_sample.csv",
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase102_pasc_mechanism_v1/r6_feature_head_coadapt_phase89_h8_val811_20260622/phase102_r6_prediction_per_sample.csv",
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase102_pasc_mechanism_v1/r7_head_submodule_swap_phase89_h8_val811_20260622/phase102_r7_prediction_per_sample.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase100-per-sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-valid-pairs", type=int, default=50)
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


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    deny = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if denx == 0.0 or deny == 0.0:
        return math.nan
    return num / (denx * deny)


def ranks(values: list[float]) -> list[float]:
    indexed = [(value, idx) for idx, value in enumerate(values)]
    indexed.sort(key=lambda item: item[0])
    result = [math.nan] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for _value, idx in indexed[i:j]:
            result[idx] = rank
        i = j
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


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
                "axis_step_lag": lag,
                "axis_abs_step_lag": abs(lag),
                "axis_mismatch_derived": int(best_heading != best_range),
            }
        )
        out.append(enriched)
    return out


def summarize_correlations(rows: list[dict[str, Any]], min_valid_pairs: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for proxy in PROXY_COLUMNS:
        if proxy not in rows[0]:
            continue
        xs_all = [f(row, proxy) for row in rows]
        for target in TARGET_COLUMNS:
            ys_all = [f(row, target) for row in rows]
            valid = [(x, y) for x, y in zip(xs_all, ys_all) if math.isfinite(x) and math.isfinite(y)]
            if len(valid) < min_valid_pairs:
                continue
            xs = [p[0] for p in valid]
            ys = [p[1] for p in valid]
            out.append(
                {
                    "proxy": proxy,
                    "target": target,
                    "valid_pairs": len(valid),
                    "proxy_mean": safe_mean(xs),
                    "target_mean": safe_mean(ys),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    "abs_spearman": abs(spearman(xs, ys)) if math.isfinite(spearman(xs, ys)) else math.nan,
                }
            )
    return sorted(out, key=lambda row: (-float(row["abs_spearman"]), str(row["proxy"]), str(row["target"])))


def summarize_quartiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for proxy in PROXY_COLUMNS:
        if proxy not in rows[0]:
            continue
        values = [f(row, proxy) for row in rows]
        q25 = quantile(values, 0.25)
        q75 = quantile(values, 0.75)
        for bucket_name, predicate in (
            ("bottom_q25", lambda value, q=q25: math.isfinite(value) and value <= q),
            ("top_q75", lambda value, q=q75: math.isfinite(value) and value >= q),
        ):
            selected = [row for row in rows if predicate(f(row, proxy))]
            if not selected:
                continue
            out.append(
                {
                    "proxy": proxy,
                    "bucket": bucket_name,
                    "threshold": q25 if bucket_name == "bottom_q25" else q75,
                    "count": len(selected),
                    "mean_proxy": safe_mean([f(row, proxy) for row in selected]),
                    "mean_axis_step_lag": safe_mean([f(row, "axis_step_lag") for row in selected]),
                    "mean_abs_axis_step_lag": safe_mean([f(row, "axis_abs_step_lag") for row in selected]),
                    "axis_mismatch_frac": safe_mean([f(row, "axis_mismatch_derived") for row in selected]),
                    "mean_axiswise_headroom": safe_mean([f(row, "baseline_minus_axiswise_oracle") for row in selected]),
                    "mean_best_checkpoint_headroom": safe_mean(
                        [f(row, "baseline_minus_best_final_error") for row in selected]
                    ),
                    "mean_final_error": safe_mean([f(row, "final_error_proxy") for row in selected]),
                }
            )
    return sorted(
        out,
        key=lambda row: (
            -float(row["mean_axiswise_headroom"]) if math.isfinite(float(row["mean_axiswise_headroom"])) else 0.0,
            -float(row["axis_mismatch_frac"]) if math.isfinite(float(row["axis_mismatch_frac"])) else 0.0,
            str(row["proxy"]),
        ),
    )


def audit_readiness() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in EXPECTED_FULL_R2_FILES:
        path = Path(path_text)
        rows.append(
            {
                "artifact": path.name,
                "path": path_text,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "required_for": "full_phase103_r2_feature_head_phase_lag",
            }
        )
    return rows


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    corr_rows: list[dict[str, Any]],
    quartile_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase103-R2a Proxy Alignment And Readiness",
        "",
        "Lab-side proxy diagnostic. This is not the full feature-head phase-lag probe.",
        "",
        "## Run",
        "",
        f"- phase100_per_sample: `{summary['phase100_per_sample']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- rows: `{summary['rows']}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        f"- full_r2_ready: `{summary['full_r2_ready']}`",
        "",
        "## Top Proxy Correlations",
        "",
        "| proxy | target | n | pearson | spearman |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in corr_rows[:20]:
        lines.append(
            f"| `{row['proxy']}` | `{row['target']}` | {row['valid_pairs']} | "
            f"{float(row['pearson']):.4f} | {float(row['spearman']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strong Quartile Buckets",
            "",
            "| proxy | bucket | count | mean lag | abs lag | mismatch | headroom | final error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in quartile_rows[:20]:
        lines.append(
            f"| `{row['proxy']}` | `{row['bucket']}` | {row['count']} | "
            f"{float(row['mean_axis_step_lag']):.2f} | {float(row['mean_abs_axis_step_lag']):.2f} | "
            f"{float(row['axis_mismatch_frac']):.3f} | {float(row['mean_axiswise_headroom']):.8g} | "
            f"{float(row['mean_final_error']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Full R2 Readiness",
            "",
            "| artifact | exists | size bytes |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in readiness_rows:
        lines.append(f"| `{row['artifact']}` | `{row['exists']}` | {row['size_bytes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase100_path = Path(args.phase100_per_sample)
    rows = enrich_rows(read_csv(phase100_path))
    corr_rows = summarize_correlations(rows, args.min_valid_pairs)
    quartile_rows = summarize_quartiles(rows)
    readiness_rows = audit_readiness()
    full_r2_ready = all(bool(row["exists"]) for row in readiness_rows)
    summary = {
        "phase": "phase103_r2a_proxy_alignment_readiness",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase100_per_sample": str(phase100_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "min_valid_pairs": args.min_valid_pairs,
        "full_r2_ready": full_r2_ready,
        "missing_full_r2_artifacts": [row["path"] for row in readiness_rows if not row["exists"]],
        "top_proxy_correlations": corr_rows[:10],
        "top_quartile_buckets": quartile_rows[:10],
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "diagnostic_only": True,
        "proxy_stage_not_full_r2": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_csv(output_dir / "phase103_r2a_proxy_correlation_summary.csv", corr_rows)
    write_csv(output_dir / "phase103_r2a_proxy_quartile_summary.csv", quartile_rows)
    write_csv(output_dir / "phase103_r2a_full_r2_readiness.csv", readiness_rows)
    (output_dir / "phase103_r2a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "phase103_r2a_summary.md",
        summary,
        corr_rows,
        quartile_rows,
        readiness_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
