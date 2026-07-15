#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00)
TAIL_THRESHOLDS = (0.5, 1.0, 2.0)


def is_deployable_feature(feature: str) -> bool:
    return not feature.endswith("_analysis_only")


def wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_abs_error_deg(pred: float, target: float) -> float:
    return abs(wrap_angle_deg(float(pred) - float(target)))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def circular_mean_deg(a: float, b: float) -> float:
    ar = math.radians(float(a))
    br = math.radians(float(b))
    return math.degrees(math.atan2(math.sin(ar) + math.sin(br), math.cos(ar) + math.cos(br)))


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("json_path") or "")


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row_key(row).split("/")[0])


def enrich_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    target_heading = safe_float(row.get("target_heading"))
    rank1_heading = safe_float(row.get("rank1_heading"))
    rank1_distance = safe_float(row.get("rank1_distance"))
    target_distance = safe_float(row.get("target_distance"))
    if target_heading is None or rank1_heading is None:
        raise ValueError(f"missing target/rank1 heading for row {row_key(row)!r}")

    row["target_heading_float"] = target_heading
    row["rank1_heading_float"] = rank1_heading
    row["rank1_abs_error_float"] = angle_abs_error_deg(rank1_heading, target_heading)
    row["rank1_signed_error_float"] = wrap_angle_deg(rank1_heading - target_heading)

    if rank1_distance is not None:
        row["rank1_distance_float"] = rank1_distance
        row["abs_rank1_distance_float"] = abs(rank1_distance)
    if target_distance is not None:
        row["target_distance_float"] = target_distance
    if rank1_distance is not None and target_distance is not None:
        row["rank1_distance_abs_error_float"] = abs(rank1_distance - target_distance)

    wrapped_heading = wrap_angle_deg(rank1_heading)
    row["rank1_heading_abs_float"] = abs(wrapped_heading)
    row["rank1_heading_sin_float"] = math.sin(math.radians(wrapped_heading))
    row["rank1_heading_cos_float"] = math.cos(math.radians(wrapped_heading))

    alt_heading = safe_float(row.get("same_forward_avg_heading"))
    if alt_heading is None:
        reverse_forward = safe_float(row.get("reverse_forward_heading"))
        if reverse_forward is not None:
            alt_heading = circular_mean_deg(rank1_heading, reverse_forward)
    if alt_heading is None:
        reverse_heading = safe_float(row.get("reverse_heading"))
        if reverse_heading is not None:
            alt_heading = circular_mean_deg(rank1_heading, -reverse_heading)
    if alt_heading is not None:
        row["alt_heading_float"] = alt_heading
        row["alt_abs_error_float"] = angle_abs_error_deg(alt_heading, target_heading)
        row["oracle_alt_better"] = row["alt_abs_error_float"] < row["rank1_abs_error_float"]
        row["oracle_alt_gain_float"] = row["rank1_abs_error_float"] - row["alt_abs_error_float"]

    feature_values: dict[str, float] = {}
    for source, name in [
        ("rank1_distance", "rank1_distance"),
        ("abs_rank1_distance_float", "abs_rank1_distance"),
        ("rank1_heading_abs_float", "rank1_heading_abs"),
        ("rank1_heading_sin_float", "rank1_heading_sin"),
        ("rank1_heading_cos_float", "rank1_heading_cos"),
        ("rank1_distance_abs_error_float", "rank1_distance_abs_error_analysis_only"),
        ("same_forward_heading_disagreement", "same_forward_heading_disagreement"),
        ("same_forward_distance_disagreement", "same_forward_distance_disagreement"),
    ]:
        value = safe_float(row.get(source))
        if value is not None:
            feature_values[name] = value
    reverse_heading = safe_float(row.get("reverse_heading"))
    if reverse_heading is not None:
        feature_values["reverse_heading_inverse_disagreement"] = angle_abs_error_deg(rank1_heading, -reverse_heading)
    reverse_distance = safe_float(row.get("reverse_distance"))
    if reverse_distance is not None and rank1_distance is not None:
        feature_values["reverse_distance_inverse_disagreement"] = abs(rank1_distance + reverse_distance)
    row["feature_values"] = feature_values
    return row


def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [enrich_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def split_folds(rows: list[dict[str, Any]], folds: int) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(folds)}
    for row in rows:
        digest = hashlib.sha1(group_key(row).encode("utf-8")).hexdigest()
        fold_id = int(digest[:8], 16) % folds
        out[fold_id].append(row)
    return out


def base_metrics(rows: list[dict[str, Any]], *, use_alt: bool = False, selected: set[str] | None = None) -> dict[str, Any]:
    angle_errors: list[float] = []
    distance_errors: list[float] = []
    selected = selected or set()
    for row in rows:
        if use_alt and row_key(row) in selected and "alt_abs_error_float" in row:
            angle_errors.append(float(row["alt_abs_error_float"]))
        else:
            angle_errors.append(float(row["rank1_abs_error_float"]))
        distance_err = safe_float(row.get("rank1_distance_abs_error_float"))
        if distance_err is not None:
            distance_errors.append(distance_err)
    return {
        "rows": len(rows),
        "angle_mae": mean(angle_errors),
        "angle_p90_abs": percentile(angle_errors, 0.90),
        "angle_p95_abs": percentile(angle_errors, 0.95),
        "angle_max_abs": max(angle_errors) if angle_errors else 0.0,
        "angle_ge_0p5": sum(err >= 0.5 for err in angle_errors),
        "angle_ge_1p0": sum(err >= 1.0 for err in angle_errors),
        "angle_ge_2p0": sum(err >= 2.0 for err in angle_errors),
        "distance_mae": mean(distance_errors),
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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 2 else None


def rank_auc(scores: list[float], labels: list[bool]) -> float:
    pairs = list(zip(scores, labels))
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ascending = sorted(pairs, key=lambda item: item[0])
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


def top_fraction_lift(scores: list[float], labels: list[bool], fraction: float = 0.10) -> float | None:
    if not scores:
        return None
    base = sum(labels) / len(labels)
    if base <= 0.0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    top_n = max(1, math.ceil(len(ordered) * fraction))
    return (sum(label for _, label in ordered[:top_n]) / top_n) / base


def available_features(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row["feature_values"])
    return sorted(names)


def signal_metrics(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features:
        valid = [
            (float(row["feature_values"][feature]), float(row["rank1_abs_error_float"]))
            for row in rows
            if feature in row["feature_values"]
        ]
        scores = [score for score, _ in valid]
        errors = [err for _, err in valid]
        item: dict[str, Any] = {
            "feature": feature,
            "deployable": is_deployable_feature(feature),
            "valid_rows": len(valid),
            "pearson_abs_error": pearson(scores, errors),
            "spearman_abs_error": spearman(scores, errors),
        }
        for threshold in TAIL_THRESHOLDS:
            labels = [err >= threshold for err in errors]
            auc = rank_auc(scores, labels) if valid else 0.5
            suffix = str(threshold).replace(".", "p")
            item[f"tail_ge_{suffix}_positives"] = sum(labels)
            item[f"tail_ge_{suffix}_auc"] = auc
            item[f"tail_ge_{suffix}_best_auc"] = max(auc, 1.0 - auc)
            item[f"tail_ge_{suffix}_top_decile_lift"] = top_fraction_lift(scores, labels, 0.10)
        out.append(item)
    return out


def select_keys(rows: list[dict[str, Any]], feature: str, direction: str, fraction: float) -> set[str]:
    valid = [row for row in rows if feature in row["feature_values"] and "alt_abs_error_float" in row]
    reverse = direction == "high"
    ordered = sorted(valid, key=lambda row: float(row["feature_values"][feature]), reverse=reverse)
    take = max(1, math.ceil(len(ordered) * fraction)) if ordered else 0
    return {row_key(row) for row in ordered[:take]}


def evaluate_selector(rows: list[dict[str, Any]], feature: str, direction: str, fraction: float) -> dict[str, Any]:
    selected = select_keys(rows, feature, direction, fraction)
    metrics = base_metrics(rows, use_alt=True, selected=selected)
    baseline = base_metrics(rows)
    return {
        "feature": feature,
        "direction": direction,
        "fraction": fraction,
        "selected_rows": len(selected),
        **{f"baseline_{k}": v for k, v in baseline.items()},
        **{f"corrected_{k}": v for k, v in metrics.items()},
        "angle_mae_delta": metrics["angle_mae"] - baseline["angle_mae"],
        "angle_mae_rel_improvement": (baseline["angle_mae"] - metrics["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0,
        "distance_mae_delta": metrics["distance_mae"] - baseline["distance_mae"],
    }


def fit_best_selector(calib_rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature in [name for name in features if is_deployable_feature(name)]:
        for direction in ("high", "low"):
            for fraction in FRACTIONS:
                candidates.append(evaluate_selector(calib_rows, feature, direction, fraction))
    candidates.sort(key=lambda row: (-row["angle_mae_rel_improvement"], row["corrected_angle_mae"], row["selected_rows"]))
    if not candidates:
        raise ValueError("no selector candidates")
    best = candidates[0]
    return {
        "feature": best["feature"],
        "direction": best["direction"],
        "fraction": best["fraction"],
        "calib_angle_mae_rel_improvement": best["angle_mae_rel_improvement"],
        "calib_selected_rows": best["selected_rows"],
    }


def evaluate_cv(rows: list[dict[str, Any]], folds: int = 5) -> dict[str, Any]:
    features = available_features(rows)
    selector_features = [name for name in features if is_deployable_feature(name)]
    fold_rows = split_folds(rows, folds)
    baseline = base_metrics(rows)
    full_alt = base_metrics(rows, use_alt=True, selected={row_key(row) for row in rows})
    oracle_selected = {row_key(row) for row in rows if row.get("oracle_alt_better")}
    oracle = base_metrics(rows, use_alt=True, selected=oracle_selected)
    method_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for fold_id, holdout_rows in fold_rows.items():
        if not holdout_rows:
            continue
        calib_rows = [row for other_id, part in fold_rows.items() if other_id != fold_id for row in part]
        selector = fit_best_selector(calib_rows, selector_features)
        holdout_result = evaluate_selector(holdout_rows, selector["feature"], selector["direction"], selector["fraction"])
        fold_metric_rows.append({"fold_id": fold_id, **selector, **holdout_result})

    selected_cv_metrics = base_metrics(rows)
    cv_angle_values: list[float] = []
    cv_distance_values: list[float] = []
    for fold in fold_metric_rows:
        cv_angle_values.append(float(fold["corrected_angle_mae"]) * int(fold["corrected_rows"]))
        cv_distance_values.append(float(fold["corrected_distance_mae"]) * int(fold["corrected_rows"]))
    rows_total = sum(int(fold["corrected_rows"]) for fold in fold_metric_rows)
    if rows_total:
        selected_cv_metrics["angle_mae"] = sum(cv_angle_values) / rows_total
        selected_cv_metrics["distance_mae"] = sum(cv_distance_values) / rows_total

    selected_rel = (baseline["angle_mae"] - selected_cv_metrics["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0
    full_alt_rel = (baseline["angle_mae"] - full_alt["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0
    oracle_rel = (baseline["angle_mae"] - oracle["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0
    conversion = selected_rel / oracle_rel if oracle_rel > 0.0 else 0.0

    method_rows.append({"method": "m0_rank1_noop", **{f"corrected_{k}": v for k, v in baseline.items()}, "angle_mae_rel_improvement": 0.0})
    method_rows.append({"method": "m1_full_alt_average", **{f"corrected_{k}": v for k, v in full_alt.items()}, "angle_mae_rel_improvement": full_alt_rel})
    method_rows.append({"method": "m2_oracle_alt_if_better", **{f"corrected_{k}": v for k, v in oracle.items()}, "angle_mae_rel_improvement": oracle_rel})
    method_rows.append(
        {
            "method": "m3_cv_feature_selector",
            **{f"corrected_{k}": v for k, v in selected_cv_metrics.items()},
            "angle_mae_rel_improvement": selected_rel,
            "oracle_conversion": conversion,
            "folds_improved": sum(1 for row in fold_metric_rows if row["angle_mae_rel_improvement"] > 0.0),
            "worst_fold_rel_regression": max([-row["angle_mae_rel_improvement"] for row in fold_metric_rows] or [0.0]),
        }
    )

    decision = "route_c_hold"
    if selected_rel >= 0.025 and conversion >= 0.30 and method_rows[-1]["folds_improved"] >= 3:
        decision = "promote_selector_probe"
    elif oracle_rel >= 0.05 and selected_rel < 0.015:
        decision = "observability_gap_still_dominant"
    elif oracle_rel < 0.05:
        decision = "alternate_source_too_weak"

    return {
        "baseline": baseline,
        "features": features,
        "selector_features": selector_features,
        "signal_metrics": signal_metrics(rows, features),
        "method_metrics": method_rows,
        "fold_metrics": fold_metric_rows,
        "decision": decision,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(output_dir: Path, result: dict[str, Any], input_csv: Path) -> None:
    methods = {row["method"]: row for row in result["method_metrics"]}
    selector = methods["m3_cv_feature_selector"]
    oracle = methods["m2_oracle_alt_if_better"]
    full_alt = methods["m1_full_alt_average"]
    best_signals = sorted(
        [row for row in result["signal_metrics"] if row.get("deployable")],
        key=lambda row: max(
            float(row.get("tail_ge_0p5_best_auc") or 0.5),
            float(row.get("tail_ge_1p0_best_auc") or 0.5),
            float(row.get("tail_ge_2p0_best_auc") or 0.5),
        ),
        reverse=True,
    )[:8]
    lines = [
        "# Phase59 Route C Observability Audit",
        "",
        f"Input CSV: `{input_csv}`",
        "",
        "## Summary",
        "",
        f"- baseline angle MAE: {result['baseline']['angle_mae']:.12f}",
        f"- full alternate angle MAE: {full_alt['corrected_angle_mae']:.12f}, rel_improvement={full_alt['angle_mae_rel_improvement']:.6f}",
        f"- oracle alternate angle MAE: {oracle['corrected_angle_mae']:.12f}, rel_improvement={oracle['angle_mae_rel_improvement']:.6f}",
        f"- CV deployable selector angle MAE: {selector['corrected_angle_mae']:.12f}, rel_improvement={selector['angle_mae_rel_improvement']:.6f}",
        f"- oracle conversion: {selector['oracle_conversion']:.6f}",
        f"- folds improved: {selector['folds_improved']}",
        f"- worst fold rel regression: {selector['worst_fold_rel_regression']:.6f}",
        f"- decision: {result['decision']}",
        "",
        "## Best Deployable Tail Signals",
        "",
    ]
    for row in best_signals:
        lines.append(
            f"- {row['feature']}: spearman={row['spearman_abs_error']}, "
            f"best_auc@0.5={row['tail_ge_0p5_best_auc']}, "
            f"best_auc@1.0={row['tail_ge_1p0_best_auc']}, "
            f"best_auc@2.0={row['tail_ge_2p0_best_auc']}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This audit is deployment-style: selector parameters are chosen on calibration folds and evaluated on held-out groups.",
        "Oracle rows use ground truth only to measure complementarity upper bound; they are not deployable.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, result: dict[str, Any], input_csv: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "signal_metrics.csv", result["signal_metrics"])
    write_csv(output_dir / "method_metrics.csv", result["method_metrics"])
    write_csv(output_dir / "fold_metrics.csv", result["fold_metrics"])
    payload = {
        "input_csv": str(input_csv),
        "baseline": result["baseline"],
        "decision": result["decision"],
        "features": result["features"],
        "selector_features": result["selector_features"],
        "method_metrics": result["method_metrics"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, result, input_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase59 Route C observability audit for rank1-compatible angle selectors.")
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_prediction_csv(args.prediction_csv)
    if not any("alt_abs_error_float" in row for row in rows):
        raise SystemExit("prediction CSV must contain same_forward/reverse fields to define an alternate heading")
    result = evaluate_cv(rows, folds=args.folds)
    write_outputs(args.output_dir, result, args.prediction_csv)
    print(json.dumps({"decision": result["decision"], "method_metrics": result["method_metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
