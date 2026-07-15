#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEPLOYABLE_DEFAULT_SIGNALS = [
    "rank1_distance",
    "abs_rank1_distance",
    "rank1_heading_abs",
    "rank1_heading_sin",
    "rank1_heading_cos",
]

ANALYSIS_ONLY_SIGNALS = {
    "rank1_distance_abs_error",
    "target_heading",
    "target_distance",
}


def wrapped_signed_angle_error_deg(pred: float, target: float) -> float:
    return (float(pred) - float(target) + 180.0) % 360.0 - 180.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    pred_heading = safe_float(out.get("rank1_heading"))
    target_heading = safe_float(out.get("target_heading"))
    rank1_distance = safe_float(out.get("rank1_distance"))
    if pred_heading is not None:
        wrapped_heading = wrapped_signed_angle_error_deg(pred_heading, 0.0)
        out["rank1_heading_abs"] = abs(wrapped_heading)
        out["rank1_heading_sin"] = math.sin(math.radians(wrapped_heading))
        out["rank1_heading_cos"] = math.cos(math.radians(wrapped_heading))
    if rank1_distance is not None:
        out["abs_rank1_distance"] = abs(rank1_distance)
    if pred_heading is not None and target_heading is not None:
        signed = wrapped_signed_angle_error_deg(pred_heading, target_heading)
        out["rank1_angle_signed_error"] = signed
        out["rank1_angle_abs_error"] = abs(signed)
    reverse_heading = safe_float(out.get("reverse_heading"))
    if pred_heading is not None and reverse_heading is not None:
        out["reverse_heading_inverse_disagreement"] = abs(wrapped_signed_angle_error_deg(pred_heading, -reverse_heading))
    reverse_distance = safe_float(out.get("reverse_distance"))
    if rank1_distance is not None and reverse_distance is not None:
        out["reverse_distance_inverse_disagreement"] = abs(rank1_distance + reverse_distance)
    return out


def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [enrich_row(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["prediction_csv"] = str(path)
    return rows


def compute_base_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signed_errors = [safe_float(row.get("rank1_angle_signed_error")) for row in rows]
    signed = [value for value in signed_errors if value is not None]
    abs_errors = [abs(value) for value in signed]
    distance_errors = [safe_float(row.get("rank1_distance_abs_error")) for row in rows]
    dist = [value for value in distance_errors if value is not None]
    return {
        "rows": len(rows),
        "valid_angle_rows": len(abs_errors),
        "angle_mae": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "angle_signed_mean": sum(signed) / len(signed) if signed else None,
        "angle_signed_median": statistics.median(signed) if signed else None,
        "angle_p90_abs": percentile(abs_errors, 0.90),
        "angle_p95_abs": percentile(abs_errors, 0.95),
        "angle_max_abs": max(abs_errors) if abs_errors else None,
        "angle_ge_0p5": sum(value >= 0.5 for value in abs_errors),
        "angle_ge_1p0": sum(value >= 1.0 for value in abs_errors),
        "angle_ge_2p0": sum(value >= 2.0 for value in abs_errors),
        "distance_mae": sum(dist) / len(dist) if dist else None,
        "valid_distance_rows": len(dist),
    }


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for original_idx, _ in indexed[i:j]:
            ranks[original_idx] = avg_rank
        i = j
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    return cov / math.sqrt(x_var * y_var)


def spearman(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    return pearson(_ranks(x_values), _ranks(y_values))


def rank_auc(scores: list[float], labels: list[bool]) -> float:
    valid = [(float(score), bool(label)) for score, label in zip(scores, labels)]
    n_pos = sum(1 for _, label in valid if label)
    n_neg = len(valid) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    ascending = sorted(valid, key=lambda item: item[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(ascending):
        j = i + 1
        while j < len(ascending) and ascending[j][0] == ascending[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_rank_sum += avg_rank * sum(1 for _, label in ascending[i:j] if label)
        i = j
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def top_decile_lift(scores: list[float], labels: list[bool]) -> float | None:
    if not scores:
        return None
    base = sum(labels) / len(labels)
    if base <= 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    top_n = max(1, math.ceil(len(ordered) * 0.10))
    top_rate = sum(label for _, label in ordered[:top_n]) / top_n
    return top_rate / base


def compute_signal_metrics(rows: list[dict[str, Any]], signal_fields: list[str]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for field in signal_fields:
        valid: list[tuple[float, float]] = []
        for row in rows:
            score = safe_float(row.get(field))
            abs_error = safe_float(row.get("rank1_angle_abs_error"))
            if score is None or abs_error is None:
                continue
            valid.append((score, abs_error))
        scores = [score for score, _ in valid]
        abs_errors = [err for _, err in valid]
        row = {
            "signal_field": field,
            "deployable": field not in ANALYSIS_ONLY_SIGNALS,
            "valid_rows": len(valid),
            "pearson_abs_error": pearson(scores, abs_errors),
            "spearman_abs_error": spearman(scores, abs_errors),
        }
        for threshold in (0.5, 1.0, 2.0):
            labels = [err >= threshold for err in abs_errors]
            row[f"tail_ge_{str(threshold).replace('.', 'p')}_positives"] = sum(labels)
            auc = rank_auc(scores, labels) if valid else None
            row[f"tail_ge_{str(threshold).replace('.', 'p')}_auc"] = auc
            row[f"tail_ge_{str(threshold).replace('.', 'p')}_best_auc"] = max(auc, 1.0 - auc) if auc is not None else None
            row[f"tail_ge_{str(threshold).replace('.', 'p')}_top_decile_lift"] = top_decile_lift(scores, labels)
        metrics.append(row)
    return metrics


def _bin_label(value: float, bins: list[tuple[float, float]]) -> str:
    for low, high in bins:
        if low <= value < high:
            high_text = "inf" if math.isinf(high) else f"{high:g}"
            return f"{low:g}:{high_text}"
    return "unbinned"


def compute_bin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = [
        ("abs_rank1_distance", [(0, 50), (50, 75), (75, 100), (100, 115), (115, 130), (130, math.inf)]),
        ("rank1_heading_abs", [(0, 45), (45, 90), (90, 135), (135, 180.000001)]),
    ]
    out: list[dict[str, Any]] = []
    for field, bins in configs:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            value = safe_float(row.get(field))
            err = safe_float(row.get("rank1_angle_signed_error"))
            if value is None or err is None:
                continue
            grouped.setdefault(_bin_label(value, bins), []).append(err)
        for label, errors in sorted(grouped.items()):
            abs_errors = [abs(err) for err in errors]
            out.append(
                {
                    "bin_field": field,
                    "bin": label,
                    "rows": len(errors),
                    "signed_mean": sum(errors) / len(errors),
                    "angle_mae": sum(abs_errors) / len(abs_errors),
                    "angle_ge_0p5": sum(err >= 0.5 for err in abs_errors),
                    "angle_ge_1p0": sum(err >= 1.0 for err in abs_errors),
                    "angle_ge_2p0": sum(err >= 2.0 for err in abs_errors),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, report: dict[str, Any]) -> None:
    base = report["base_metrics"]
    best = report["best_signal_metrics"][:8]
    lines = [
        "# Phase57 Residual Structure Audit",
        "",
        "This report is audit-only. It does not fit or apply an angle correction.",
        "",
        "## Base Metrics",
        "",
        f"- rows: {base['rows']}",
        f"- valid_angle_rows: {base['valid_angle_rows']}",
        f"- angle_mae: {base['angle_mae']}",
        f"- angle_signed_mean: {base['angle_signed_mean']}",
        f"- angle_p95_abs: {base['angle_p95_abs']}",
        f"- angle_max_abs: {base['angle_max_abs']}",
        f"- angle_ge_0p5: {base['angle_ge_0p5']}",
        f"- angle_ge_1p0: {base['angle_ge_1p0']}",
        f"- angle_ge_2p0: {base['angle_ge_2p0']}",
        "",
        "## Top Signal Metrics",
        "",
    ]
    for row in best:
        lines.append(
            "- "
            f"{row['signal_field']} deployable={row['deployable']} "
            f"spearman={row['spearman_abs_error']} "
            f"auc_ge_0p5={row['tail_ge_0p5_auc']} "
            f"best_auc_ge_0p5={row['tail_ge_0p5_best_auc']} "
            f"lift_ge_0p5={row['tail_ge_0p5_top_decile_lift']}"
        )
    lines += [
        "",
        "Analysis-only signals use target-derived fields and must not be used for hidden/test correction.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def available_default_signals(rows: list[dict[str, Any]]) -> list[str]:
    fields = set()
    for row in rows:
        fields.update(row.keys())
    signals = [field for field in DEPLOYABLE_DEFAULT_SIGNALS if field in fields]
    for field in ("reverse_heading_inverse_disagreement", "reverse_distance_inverse_disagreement", "rank1_distance_abs_error"):
        if field in fields:
            signals.append(field)
    return signals


def run_audit(prediction_csvs: list[Path], output_dir: Path, signal_fields: list[str] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in prediction_csvs:
        rows.extend(read_prediction_csv(path))
    fields = signal_fields or available_default_signals(rows)
    base = compute_base_metrics(rows)
    signal_metrics = compute_signal_metrics(rows, fields)
    bin_rows = compute_bin_rows(rows)
    best = sorted(
        signal_metrics,
        key=lambda row: (
            row.get("tail_ge_0p5_best_auc") or 0.0,
            abs(row.get("spearman_abs_error") or 0.0),
            row.get("valid_rows") or 0,
        ),
        reverse=True,
    )
    report = {
        "prediction_csvs": [str(path) for path in prediction_csvs],
        "output_dir": str(output_dir),
        "signal_fields": fields,
        "base_metrics": base,
        "best_signal_metrics": best,
        "bin_row_count": len(bin_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "residual_structure_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    signal_fields_out = [
        "signal_field",
        "deployable",
        "valid_rows",
        "pearson_abs_error",
        "spearman_abs_error",
        "tail_ge_0p5_positives",
        "tail_ge_0p5_auc",
        "tail_ge_0p5_best_auc",
        "tail_ge_0p5_top_decile_lift",
        "tail_ge_1p0_positives",
        "tail_ge_1p0_auc",
        "tail_ge_1p0_best_auc",
        "tail_ge_1p0_top_decile_lift",
        "tail_ge_2p0_positives",
        "tail_ge_2p0_auc",
        "tail_ge_2p0_best_auc",
        "tail_ge_2p0_top_decile_lift",
    ]
    write_csv(output_dir / "residual_signal_metrics.csv", signal_metrics, signal_fields_out)
    write_csv(
        output_dir / "residual_bins.csv",
        bin_rows,
        ["bin_field", "bin", "rows", "signed_mean", "angle_mae", "angle_ge_0p5", "angle_ge_1p0", "angle_ge_2p0"],
    )
    write_report(output_dir / "residual_structure_report.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether frozen rank1 PairUAV residuals are predictable.")
    parser.add_argument("--prediction-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-field", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(args.prediction_csv, args.output_dir, args.signal_field)
    print(json.dumps(report["base_metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
