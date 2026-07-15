#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DISTANCE_BINS = [(0.0, 50.0), (50.0, 75.0), (75.0, 100.0), (100.0, 115.0), (115.0, 130.0), (130.0, math.inf)]
GATED_PERCENTILES = (70.0, 80.0, 90.0)
REVERSE_DISAGREEMENT_PERCENTILES = (50.0, 70.0, 90.0)


def wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def signed_error(pred: float, target: float) -> float:
    return wrap_angle_deg(float(pred) - float(target))


def safe_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite value: {value!r}")
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def circular_mean_deg(a: float, b: float) -> float:
    ar = math.radians(float(a))
    br = math.radians(float(b))
    return math.degrees(math.atan2(math.sin(ar) + math.sin(br), math.cos(ar) + math.cos(br)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def distance_bin(value: float) -> str:
    abs_value = abs(float(value))
    for low, high in DISTANCE_BINS:
        if low <= abs_value < high:
            return f"{low:g}:{'inf' if math.isinf(high) else f'{high:g}'}"
    return "unbinned"


def read_prediction_csv(path: Path, fold_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["fold_id"] = fold_id
            row["rank1_heading_float"] = safe_float(row["rank1_heading"])
            row["rank1_distance_float"] = safe_float(row["rank1_distance"])
            row["target_heading_float"] = safe_float(row["target_heading"])
            row["target_distance_float"] = safe_float(row["target_distance"])
            row["rank1_signed_error"] = signed_error(row["rank1_heading_float"], row["target_heading_float"])
            row["rank1_abs_error"] = abs(row["rank1_signed_error"])
            row["abs_rank1_distance"] = abs(row["rank1_distance_float"])
            if row.get("reverse_forward_heading") not in (None, ""):
                row["reverse_forward_heading_float"] = safe_float(row["reverse_forward_heading"])
            elif row.get("reverse_heading") not in (None, ""):
                row["reverse_forward_heading_float"] = wrap_angle_deg(-safe_float(row["reverse_heading"]))
            if row.get("reverse_forward_distance") not in (None, ""):
                row["reverse_forward_distance_float"] = safe_float(row["reverse_forward_distance"])
            elif row.get("reverse_distance") not in (None, ""):
                row["reverse_forward_distance_float"] = -safe_float(row["reverse_distance"])
            if "reverse_forward_heading_float" in row:
                row["same_forward_heading_disagreement_float"] = abs(
                    signed_error(row["rank1_heading_float"], row["reverse_forward_heading_float"])
                )
                row["same_forward_avg_heading_float"] = circular_mean_deg(
                    row["rank1_heading_float"],
                    row["reverse_forward_heading_float"],
                )
            rows.append(row)
    return rows


def load_fold_predictions(fold_specs: list[str]) -> dict[int, list[dict[str, Any]]]:
    folds: dict[int, list[dict[str, Any]]] = {}
    for spec in fold_specs:
        if "=" not in spec:
            raise ValueError(f"fold prediction spec must be fold_id=path: {spec!r}")
        fold_text, path_text = spec.split("=", 1)
        fold_id = int(fold_text)
        if fold_id in folds:
            raise ValueError(f"duplicate fold id: {fold_id}")
        folds[fold_id] = read_prediction_csv(Path(path_text), fold_id)
    if not folds:
        raise ValueError("no fold predictions supplied")
    return dict(sorted(folds.items()))


def base_metrics(rows: list[dict[str, Any]], heading_key: str) -> dict[str, Any]:
    angle_errors = [abs(signed_error(row[heading_key], row["target_heading_float"])) for row in rows]
    distance_errors = [abs(row["rank1_distance_float"] - row["target_distance_float"]) for row in rows]
    return {
        "rows": len(rows),
        "angle_mae": mean(angle_errors),
        "angle_p90_abs": percentile(angle_errors, 90.0),
        "angle_p95_abs": percentile(angle_errors, 95.0),
        "angle_max_abs": max(angle_errors) if angle_errors else 0.0,
        "angle_ge_0p5": sum(err >= 0.5 for err in angle_errors),
        "angle_ge_1p0": sum(err >= 1.0 for err in angle_errors),
        "angle_ge_2p0": sum(err >= 2.0 for err in angle_errors),
        "distance_mae": mean(distance_errors),
    }


def fit_noop(calib_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"method": "m0_noop", "calib_rows": len(calib_rows)}


def apply_noop(row: dict[str, Any], params: dict[str, Any]) -> float:
    return row["rank1_heading_float"]


def fit_global_bias(calib_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bias = mean([row["rank1_signed_error"] for row in calib_rows])
    return {"method": "m1_global_bias", "bias_deg": bias, "calib_rows": len(calib_rows)}


def apply_global_bias(row: dict[str, Any], params: dict[str, Any]) -> float:
    return wrap_angle_deg(row["rank1_heading_float"] - float(params["bias_deg"]))


def fit_gated_bias(calib_rows: list[dict[str, Any]], percentile_value: float) -> dict[str, Any]:
    threshold = percentile([row["abs_rank1_distance"] for row in calib_rows], percentile_value)
    selected = [row for row in calib_rows if row["abs_rank1_distance"] >= threshold]
    bias = mean([row["rank1_signed_error"] for row in selected])
    return {
        "method": f"m2_gated_abs_distance_p{int(percentile_value)}",
        "percentile": percentile_value,
        "threshold_abs_rank1_distance": threshold,
        "bias_deg": bias,
        "calib_rows": len(calib_rows),
        "selected_rows": len(selected),
    }


def apply_gated_bias(row: dict[str, Any], params: dict[str, Any]) -> float:
    if row["abs_rank1_distance"] >= float(params["threshold_abs_rank1_distance"]):
        return wrap_angle_deg(row["rank1_heading_float"] - float(params["bias_deg"]))
    return row["rank1_heading_float"]


def fit_distance_bin_bias(calib_rows: list[dict[str, Any]], prior_strength: float = 64.0) -> dict[str, Any]:
    global_bias = mean([row["rank1_signed_error"] for row in calib_rows])
    grouped: dict[str, list[float]] = {}
    for row in calib_rows:
        grouped.setdefault(distance_bin(row["rank1_distance_float"]), []).append(row["rank1_signed_error"])
    bins: dict[str, dict[str, Any]] = {}
    for name, values in grouped.items():
        raw_bias = mean(values)
        shrink = len(values) / (len(values) + prior_strength)
        bias = shrink * raw_bias + (1.0 - shrink) * global_bias
        bins[name] = {"rows": len(values), "raw_bias_deg": raw_bias, "bias_deg": bias}
    return {
        "method": "m3_abs_distance_bin_bias",
        "global_bias_deg": global_bias,
        "prior_strength": prior_strength,
        "calib_rows": len(calib_rows),
        "bins": bins,
    }


def apply_distance_bin_bias(row: dict[str, Any], params: dict[str, Any]) -> float:
    payload = params["bins"].get(distance_bin(row["rank1_distance_float"]))
    bias = float(payload["bias_deg"]) if payload else float(params["global_bias_deg"])
    return wrap_angle_deg(row["rank1_heading_float"] - bias)


def fit_same_forward_average(calib_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"method": "m4_same_forward_average", "calib_rows": len(calib_rows)}


def apply_same_forward_average(row: dict[str, Any], params: dict[str, Any]) -> float:
    return float(row.get("same_forward_avg_heading_float", row["rank1_heading_float"]))


def fit_reverse_disagreement_gated_average(calib_rows: list[dict[str, Any]], percentile_value: float) -> dict[str, Any]:
    values = [row["same_forward_heading_disagreement_float"] for row in calib_rows if "same_forward_heading_disagreement_float" in row]
    threshold = percentile(values, percentile_value)
    return {
        "method": f"m5_reverse_disagreement_gated_avg_p{int(percentile_value)}",
        "percentile": percentile_value,
        "threshold_same_forward_heading_disagreement": threshold,
        "calib_rows": len(calib_rows),
        "reverse_feature_rows": len(values),
    }


def apply_reverse_disagreement_gated_average(row: dict[str, Any], params: dict[str, Any]) -> float:
    disagreement = row.get("same_forward_heading_disagreement_float")
    if disagreement is not None and float(disagreement) <= float(params["threshold_same_forward_heading_disagreement"]):
        return float(row.get("same_forward_avg_heading_float", row["rank1_heading_float"]))
    return row["rank1_heading_float"]


def correction_methods(has_reverse_features: bool = False) -> list[tuple[str, Any, Any]]:
    methods: list[tuple[str, Any, Any]] = []
    methods.append(("m0_noop", fit_noop, apply_noop))
    methods.append(("m1_global_bias", fit_global_bias, apply_global_bias))
    for value in GATED_PERCENTILES:
        methods.append((f"m2_gated_abs_distance_p{int(value)}", lambda rows, v=value: fit_gated_bias(rows, v), apply_gated_bias))
    methods.append(("m3_abs_distance_bin_bias", fit_distance_bin_bias, apply_distance_bin_bias))
    if has_reverse_features:
        methods.append(("m4_same_forward_average", fit_same_forward_average, apply_same_forward_average))
        for value in REVERSE_DISAGREEMENT_PERCENTILES:
            methods.append(
                (
                    f"m5_reverse_disagreement_gated_avg_p{int(value)}",
                    lambda rows, v=value: fit_reverse_disagreement_gated_average(rows, v),
                    apply_reverse_disagreement_gated_average,
                )
            )
    return methods


def evaluate_cv(folds: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in folds.values() for row in rows]
    baseline = base_metrics(all_rows, "rank1_heading_float")
    method_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    corrected_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    has_reverse_features = any("same_forward_avg_heading_float" in row for row in all_rows)

    for method_name, fit_fn, apply_fn in correction_methods(has_reverse_features=has_reverse_features):
        corrected_all: list[dict[str, Any]] = []
        configs[method_name] = {"method": method_name, "fold_params": {}}
        for fold_id, holdout_rows in folds.items():
            calib_rows = [row for other_fold, rows in folds.items() if other_fold != fold_id for row in rows]
            params = fit_fn(calib_rows)
            configs[method_name]["fold_params"][str(fold_id)] = params
            corrected_fold: list[dict[str, Any]] = []
            for row in holdout_rows:
                corrected_heading = apply_fn(row, params)
                out = dict(row)
                out["corrected_heading"] = corrected_heading
                out["corrected_distance"] = row["rank1_distance_float"]
                out["corrected_angle_abs_error"] = abs(signed_error(corrected_heading, row["target_heading_float"]))
                out["corrected_distance_abs_error"] = abs(row["rank1_distance_float"] - row["target_distance_float"])
                corrected_fold.append(out)
            corrected_all.extend(corrected_fold)
            metrics = base_metrics(corrected_fold, "corrected_heading")
            base = base_metrics(holdout_rows, "rank1_heading_float")
            fold_rows.append(
                {
                    "method": method_name,
                    "fold_id": fold_id,
                    **{f"baseline_{k}": v for k, v in base.items()},
                    **{f"corrected_{k}": v for k, v in metrics.items()},
                    "angle_mae_delta": metrics["angle_mae"] - base["angle_mae"],
                    "angle_mae_rel_improvement": (base["angle_mae"] - metrics["angle_mae"]) / base["angle_mae"] if base["angle_mae"] else 0.0,
                    "distance_mae_delta": metrics["distance_mae"] - base["distance_mae"],
                }
            )
        corrected_rows_by_method[method_name] = corrected_all
        metrics = base_metrics(corrected_all, "corrected_heading")
        method_rows.append(
            {
                "method": method_name,
                **{f"baseline_{k}": v for k, v in baseline.items()},
                **{f"corrected_{k}": v for k, v in metrics.items()},
                "angle_mae_delta": metrics["angle_mae"] - baseline["angle_mae"],
                "angle_mae_rel_improvement": (baseline["angle_mae"] - metrics["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0,
                "distance_mae_delta": metrics["distance_mae"] - baseline["distance_mae"],
                "folds_improved": sum(1 for row in fold_rows if row["method"] == method_name and row["angle_mae_delta"] < 0.0),
                "worst_fold_rel_regression": max(
                    [-(row["angle_mae_rel_improvement"]) for row in fold_rows if row["method"] == method_name] or [0.0]
                ),
            }
        )

    best_non_noop = sorted(
        [row for row in method_rows if row["method"] != "m0_noop"],
        key=lambda row: (row["corrected_angle_mae"], -row["folds_improved"], row["corrected_angle_ge_1p0"]),
    )[0]
    return {
        "baseline": baseline,
        "method_metrics": method_rows,
        "fold_metrics": fold_rows,
        "configs": configs,
        "corrected_rows_by_method": corrected_rows_by_method,
        "best_non_noop": best_non_noop,
    }


def evaluate_calibrated_apply(calib_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = base_metrics(eval_rows, "rank1_heading_float")
    method_rows: list[dict[str, Any]] = []
    corrected_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    has_reverse_features = any("same_forward_avg_heading_float" in row for row in calib_rows + eval_rows)

    for method_name, fit_fn, apply_fn in correction_methods(has_reverse_features=has_reverse_features):
        params = fit_fn(calib_rows)
        configs[method_name] = {"method": method_name, "params": params}
        corrected_rows: list[dict[str, Any]] = []
        for row in eval_rows:
            corrected_heading = apply_fn(row, params)
            out = dict(row)
            out["corrected_heading"] = corrected_heading
            out["corrected_distance"] = row["rank1_distance_float"]
            out["corrected_angle_abs_error"] = abs(signed_error(corrected_heading, row["target_heading_float"]))
            out["corrected_distance_abs_error"] = abs(row["rank1_distance_float"] - row["target_distance_float"])
            corrected_rows.append(out)
        corrected_rows_by_method[method_name] = corrected_rows
        metrics = base_metrics(corrected_rows, "corrected_heading")
        method_rows.append(
            {
                "method": method_name,
                **{f"baseline_{k}": v for k, v in baseline.items()},
                **{f"corrected_{k}": v for k, v in metrics.items()},
                "angle_mae_delta": metrics["angle_mae"] - baseline["angle_mae"],
                "angle_mae_rel_improvement": (baseline["angle_mae"] - metrics["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0,
                "distance_mae_delta": metrics["distance_mae"] - baseline["distance_mae"],
            }
        )

    best_non_noop = sorted(
        [row for row in method_rows if row["method"] != "m0_noop"],
        key=lambda row: (row["corrected_angle_mae"], row["corrected_angle_ge_1p0"]),
    )[0]
    return {
        "baseline": baseline,
        "method_metrics": method_rows,
        "fold_metrics": [],
        "configs": configs,
        "corrected_rows_by_method": corrected_rows_by_method,
        "best_non_noop": best_non_noop,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_corrected_predictions(output_dir: Path, method: str, rows: list[dict[str, Any]]) -> None:
    path = output_dir / "corrected_predictions" / f"{method}.csv"
    fields = [
        "pair_id",
        "fold_id",
        "group_id",
        "json_path",
        "target_heading",
        "target_distance",
        "rank1_heading",
        "rank1_distance",
        "corrected_heading",
        "corrected_distance",
        "rank1_abs_error",
        "corrected_angle_abs_error",
        "corrected_distance_abs_error",
    ]
    write_csv(path, rows, fields)


def write_report(output_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Phase57 G2 Angle-Only Correction Report",
        "",
        "This report evaluates low-capacity corrections only. Distance is copied from raw rank1 for every method.",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for row in sorted(result["method_metrics"], key=lambda item: item["corrected_angle_mae"]):
        folds_text = f", folds_improved={row['folds_improved']}" if "folds_improved" in row else ""
        lines.append(
            f"- {row['method']}: angle_mae {row['baseline_angle_mae']:.12f} -> {row['corrected_angle_mae']:.12f}, "
            f"rel_improvement={row['angle_mae_rel_improvement']:.6f}{folds_text}, "
            f"distance_delta={row['distance_mae_delta']:.12g}"
        )
    best = result["best_non_noop"]
    lines += [
        "",
        "## Best Non-Noop",
        "",
        f"- method: {best['method']}",
        f"- corrected_angle_mae: {best['corrected_angle_mae']}",
        f"- relative_improvement: {best['angle_mae_rel_improvement']}",
        f"- folds_improved: {best.get('folds_improved', 'n/a')}",
        f"- worst_fold_rel_regression: {best.get('worst_fold_rel_regression', 'n/a')}",
        f"- distance_mae_delta: {best['distance_mae_delta']}",
        "",
        "## Gate Note",
        "",
        "G2 should pass only if improvement is >=2%, at least 3/5 folds improve, worst-fold regression <=0.5%, and distance remains unchanged.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "method_metrics.csv", result["method_metrics"])
    write_csv(output_dir / "fold_metrics.csv", result["fold_metrics"])
    payload = {key: value for key, value in result.items() if key != "corrected_rows_by_method"}
    (output_dir / "angle_correction_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "angle_correction_configs.json").write_text(
        json.dumps(result["configs"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for method, rows in result["corrected_rows_by_method"].items():
        write_corrected_predictions(output_dir, method, rows)
    write_report(output_dir, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase57 G2 low-capacity angle-only corrections under CV.")
    parser.add_argument("--fold-prediction-csv", action="append")
    parser.add_argument("--calibration-prediction-csv", action="append")
    parser.add_argument("--eval-prediction-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fold_prediction_csv:
        folds = load_fold_predictions(args.fold_prediction_csv)
        result = evaluate_cv(folds)
    else:
        if not args.calibration_prediction_csv or not args.eval_prediction_csv:
            raise SystemExit("Either --fold-prediction-csv or both --calibration-prediction-csv/--eval-prediction-csv are required")
        calib_rows: list[dict[str, Any]] = []
        for idx, path_text in enumerate(args.calibration_prediction_csv):
            calib_rows.extend(read_prediction_csv(Path(path_text), idx))
        eval_rows = read_prediction_csv(args.eval_prediction_csv, 0)
        result = evaluate_calibrated_apply(calib_rows, eval_rows)
    write_outputs(args.output_dir, result)
    print(json.dumps({"baseline": result["baseline"], "best_non_noop": result["best_non_noop"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
