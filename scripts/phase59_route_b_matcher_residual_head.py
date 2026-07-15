#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.datasets.pairuav_matcher_features import (
    BSCR_ANCHOR_DIM,
    BSCR_GLOBAL_FEATURE_NAMES,
    BSCR_GRID_SIZE,
    BSCR_SPATIAL_CHANNELS,
    BSCR_TOPK,
    MATCHER_FEATURE_NAMES,
    extract_bscr_packet,
    extract_matcher_features,
    sample_to_match_path,
)


# Rank1 residuals are already very small; weakly regularized residual heads overfit.
# Keep this grid narrow because Phase59 Route B is a gate, not an HPO phase.
ALPHAS = (100.0, 1000.0)
CLIPS_DEG: tuple[float | None, ...] = (0.25, 0.5, 1.0)
FEATURE_SETS = {
    "rank1_only": [
        "rank1_heading_sin",
        "rank1_heading_cos",
        "rank1_heading_abs",
        "rank1_distance",
        "abs_rank1_distance",
    ],
    "matcher_only": [f"matcher:{name}" for name in MATCHER_FEATURE_NAMES],
    "bscr_only": [],
    "rank1_matcher": [
        "rank1_heading_sin",
        "rank1_heading_cos",
        "rank1_heading_abs",
        "rank1_distance",
        "abs_rank1_distance",
    ]
    + [f"matcher:{name}" for name in MATCHER_FEATURE_NAMES],
    "rank1_bscr": [
        "rank1_heading_sin",
        "rank1_heading_cos",
        "rank1_heading_abs",
        "rank1_distance",
        "abs_rank1_distance",
    ],
}


BSCR_FLAT_NAMES = (
    [f"bscr_global:{name}" for name in BSCR_GLOBAL_FEATURE_NAMES]
    + [f"bscr_spatial:{idx}" for idx in range(BSCR_GRID_SIZE * BSCR_GRID_SIZE * BSCR_SPATIAL_CHANNELS)]
    + [f"bscr_topk:{idx}" for idx in range(BSCR_TOPK * BSCR_ANCHOR_DIM)]
    + ["bscr_quality_mask", "bscr_fallback_used"]
)
FEATURE_SETS["bscr_only"] = list(BSCR_FLAT_NAMES)
FEATURE_SETS["rank1_bscr"] = FEATURE_SETS["rank1_bscr"] + list(BSCR_FLAT_NAMES)


def wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_abs_error_deg(pred: float, target: float) -> float:
    return abs(wrap_angle_deg(float(pred) - float(target)))


def safe_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite float: {value!r}")
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def row_key(row: dict[str, Any]) -> str:
    return str(row["pair_id"])


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row_key(row).split("/")[0])


def split_folds(rows: list[dict[str, Any]], folds: int) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(folds)}
    for row in rows:
        digest = hashlib.sha1(group_key(row).encode("utf-8")).hexdigest()
        out[int(digest[:8], 16) % folds].append(row)
    return out


def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            pair_id = str(raw.get("pair_id") or "")
            if not pair_id:
                raise ValueError(f"missing pair_id in {path}")
            target_heading = safe_float(raw["target_heading"])
            rank1_heading = safe_float(raw["rank1_heading"])
            target_distance = safe_float(raw["target_distance"])
            rank1_distance = safe_float(raw["rank1_distance"])
            row = dict(raw)
            row["pair_id"] = pair_id
            row["target_heading_float"] = target_heading
            row["rank1_heading_float"] = rank1_heading
            row["target_distance_float"] = target_distance
            row["rank1_distance_float"] = rank1_distance
            row["rank1_angle_abs_error_float"] = angle_abs_error_deg(rank1_heading, target_heading)
            row["rank1_distance_abs_error_float"] = abs(rank1_distance - target_distance)
            row["target_residual_deg"] = wrap_angle_deg(target_heading - rank1_heading)
            rows.append(row)
    return rows


def read_many_prediction_csv(paths: list[Path]) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_prediction_csv(path):
            key = row_key(row)
            if key in rows_by_key:
                raise ValueError(f"duplicate pair_id across train CSVs: {key}")
            rows_by_key[key] = row
    return list(rows_by_key.values())


def attach_matcher_features(rows: list[dict[str, Any]], cache_root: Path) -> dict[str, Any]:
    covered = 0
    fallback = 0
    for row in rows:
        match_path = sample_to_match_path(cache_root, row_key(row))
        raw = extract_matcher_features(match_path, image_width=512, image_height=512)
        bscr = extract_bscr_packet(match_path, image_size=(512, 512))
        row["matcher_features"] = [float(raw[name]) for name in MATCHER_FEATURE_NAMES]
        row["bscr_features"] = np.concatenate(
            [
                np.asarray(bscr["global_stats"], dtype=np.float32).reshape(-1),
                np.asarray(bscr["spatial_bins"], dtype=np.float32).reshape(-1),
                np.asarray(bscr["topk_anchors"], dtype=np.float32).reshape(-1),
                np.asarray(bscr["quality_mask"], dtype=np.float32).reshape(-1),
                np.asarray([float(bscr["fallback_used"])], dtype=np.float32),
            ]
        ).astype(float).tolist()
        row["matcher_path"] = str(match_path)
        row["matcher_fallback_used"] = bool(raw.get("fallback_used", 0.0) >= 0.5)
        if row["matcher_fallback_used"]:
            fallback += 1
        else:
            covered += 1
    return {
        "rows": len(rows),
        "covered": covered,
        "fallback": fallback,
        "coverage_rate": covered / len(rows) if rows else 0.0,
    }


def base_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    heading = wrap_angle_deg(row["rank1_heading_float"])
    rank1_distance = float(row["rank1_distance_float"])
    values = {
        "rank1_heading_sin": math.sin(math.radians(heading)),
        "rank1_heading_cos": math.cos(math.radians(heading)),
        "rank1_heading_abs": abs(heading) / 180.0,
        "rank1_distance": rank1_distance / 140.0,
        "abs_rank1_distance": abs(rank1_distance) / 140.0,
    }
    for name, value in zip(MATCHER_FEATURE_NAMES, row["matcher_features"]):
        values[f"matcher:{name}"] = float(value)
    for name, value in zip(BSCR_FLAT_NAMES, row["bscr_features"]):
        values[name] = float(value)
    return values


def matrix_for(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    matrix = [[base_feature_dict(row).get(name, 0.0) for name in feature_names] for row in rows]
    return np.asarray(matrix, dtype=np.float64)


def targets_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["target_residual_deg"]) for row in rows], dtype=np.float64)


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_values = x.mean(axis=0) if x.size else np.zeros((x.shape[1],), dtype=np.float64)
    std_values = x.std(axis=0) if x.size else np.ones((x.shape[1],), dtype=np.float64)
    std_values[std_values < 1e-8] = 1.0
    return mean_values, std_values


def normalize_apply(x: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mean_values) / std_values, nan=0.0, posinf=0.0, neginf=0.0)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean_values, std_values = normalize_fit(x)
    x_norm = normalize_apply(x, mean_values, std_values)
    design = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return {"coef": coef, "mean": mean_values, "std": std_values, "alpha": float(alpha)}


def predict_ridge(model: dict[str, Any], x: np.ndarray, clip_deg: float | None) -> np.ndarray:
    x_norm = normalize_apply(x, model["mean"], model["std"])
    design = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
    pred = design @ model["coef"]
    if clip_deg is not None:
        pred = np.clip(pred, -float(clip_deg), float(clip_deg))
    return pred


def metrics_for(rows: list[dict[str, Any]], residual_pred: np.ndarray | None = None) -> dict[str, Any]:
    angle_errors: list[float] = []
    distance_errors: list[float] = []
    for idx, row in enumerate(rows):
        heading = float(row["rank1_heading_float"])
        if residual_pred is not None:
            heading = wrap_angle_deg(heading + float(residual_pred[idx]))
        angle_errors.append(angle_abs_error_deg(heading, row["target_heading_float"]))
        distance_errors.append(float(row["rank1_distance_abs_error_float"]))
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


def evaluate_candidate(
    train_rows: list[dict[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    clip_deg: float | None,
    folds: int,
) -> dict[str, Any]:
    feature_names = FEATURE_SETS[feature_set]
    fold_rows = split_folds(train_rows, folds)
    weighted_angle = 0.0
    weighted_distance = 0.0
    total_rows = 0
    improved_folds = 0
    worst_fold_rel_regression = 0.0
    for fold_id, holdout_rows in fold_rows.items():
        if not holdout_rows:
            continue
        calib_rows = [row for other_id, part in fold_rows.items() if other_id != fold_id for row in part]
        x_calib = matrix_for(calib_rows, feature_names)
        y_calib = targets_for(calib_rows)
        model = fit_ridge(x_calib, y_calib, alpha)
        pred = predict_ridge(model, matrix_for(holdout_rows, feature_names), clip_deg)
        baseline = metrics_for(holdout_rows)
        corrected = metrics_for(holdout_rows, pred)
        rel = (baseline["angle_mae"] - corrected["angle_mae"]) / baseline["angle_mae"] if baseline["angle_mae"] else 0.0
        improved_folds += int(rel > 0.0)
        worst_fold_rel_regression = max(worst_fold_rel_regression, -rel)
        rows = len(holdout_rows)
        weighted_angle += corrected["angle_mae"] * rows
        weighted_distance += corrected["distance_mae"] * rows
        total_rows += rows
    baseline_all = metrics_for(train_rows)
    corrected_angle = weighted_angle / total_rows if total_rows else 0.0
    corrected_distance = weighted_distance / total_rows if total_rows else 0.0
    rel_improvement = (baseline_all["angle_mae"] - corrected_angle) / baseline_all["angle_mae"] if baseline_all["angle_mae"] else 0.0
    return {
        "feature_set": feature_set,
        "alpha": alpha,
        "clip_deg": clip_deg if clip_deg is not None else "none",
        "cv_corrected_angle_mae": corrected_angle,
        "cv_corrected_distance_mae": corrected_distance,
        "cv_angle_mae_rel_improvement": rel_improvement,
        "cv_improved_folds": improved_folds,
        "cv_worst_fold_rel_regression": worst_fold_rel_regression,
    }


def select_candidate(train_rows: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for alpha in ALPHAS:
            for clip_deg in CLIPS_DEG:
                candidates.append(evaluate_candidate(train_rows, feature_set=feature_set, alpha=alpha, clip_deg=clip_deg, folds=folds))
    candidates.sort(
        key=lambda row: (
            -float(row["cv_angle_mae_rel_improvement"]),
            float(row["cv_corrected_angle_mae"]),
            float(row["cv_worst_fold_rel_regression"]),
        )
    )
    return {"best": candidates[0], "candidates": candidates}


def apply_selected(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    feature_set = str(candidate["feature_set"])
    feature_names = FEATURE_SETS[feature_set]
    clip_raw = candidate["clip_deg"]
    clip_deg = None if clip_raw == "none" else float(clip_raw)
    model = fit_ridge(matrix_for(train_rows, feature_names), targets_for(train_rows), float(candidate["alpha"]))
    pred = predict_ridge(model, matrix_for(eval_rows, feature_names), clip_deg)
    metrics = metrics_for(eval_rows, pred)
    metrics["selected_feature_set"] = feature_set
    metrics["selected_alpha"] = float(candidate["alpha"])
    metrics["selected_clip_deg"] = clip_raw
    return metrics, pred


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


def write_predictions(path: Path, rows: list[dict[str, Any]], residual_pred: np.ndarray) -> None:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        corrected_heading = wrap_angle_deg(row["rank1_heading_float"] + float(residual_pred[idx]))
        out.append(
            {
                "pair_id": row["pair_id"],
                "group_id": group_key(row),
                "target_heading": row["target_heading_float"],
                "target_distance": row["target_distance_float"],
                "rank1_heading": row["rank1_heading_float"],
                "rank1_distance": row["rank1_distance_float"],
                "predicted_residual_deg": float(residual_pred[idx]),
                "corrected_heading": corrected_heading,
                "corrected_distance": row["rank1_distance_float"],
                "rank1_angle_abs_error": row["rank1_angle_abs_error_float"],
                "corrected_angle_abs_error": angle_abs_error_deg(corrected_heading, row["target_heading_float"]),
                "rank1_distance_abs_error": row["rank1_distance_abs_error_float"],
                "matcher_fallback_used": row["matcher_fallback_used"],
            }
        )
    write_csv(path, out)


def write_report(output_dir: Path, result: dict[str, Any]) -> None:
    best = result["selected_candidate"]
    val = result["eval_corrected"]
    baseline = result["eval_baseline"]
    rel = result["eval_angle_mae_rel_improvement"]
    lines = [
        "# Phase59 Route B Matcher Residual Head",
        "",
        "This is a minimal trainable matching-aware angle mechanism. It keeps rank1 distance fixed and trains only a ridge residual head for heading.",
        "",
        "## Inputs",
        "",
        f"- train rows: {result['train_rows']}",
        f"- eval rows: {result['eval_rows']}",
        f"- train matcher/BSCR coverage: {result['train_matcher_coverage']['coverage_rate']:.6f}",
        f"- eval matcher/BSCR coverage: {result['eval_matcher_coverage']['coverage_rate']:.6f}",
        "",
        "## Selected Candidate",
        "",
        f"- feature_set: {best['feature_set']}",
        f"- alpha: {best['alpha']}",
        f"- clip_deg: {best['clip_deg']}",
        f"- train CV angle rel improvement: {best['cv_angle_mae_rel_improvement']:.6f}",
        f"- train CV improved folds: {best['cv_improved_folds']}",
        f"- train CV worst fold rel regression: {best['cv_worst_fold_rel_regression']:.6f}",
        "",
        "## Fixed Val811 Result",
        "",
        f"- rank1 angle MAE: {baseline['angle_mae']:.12f}",
        f"- corrected angle MAE: {val['angle_mae']:.12f}",
        f"- angle rel improvement: {rel:.6f}",
        f"- rank1/corrected distance MAE: {baseline['distance_mae']:.12f}",
        f"- decision: {result['decision']}",
        "",
        "## Gate Note",
        "",
        "Promote only if fixed val811 angle passes Phase59 G1 while distance remains copied from rank1.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_many_prediction_csv(args.train_prediction_csv)
    eval_rows = read_prediction_csv(args.eval_prediction_csv)
    train_cov = attach_matcher_features(train_rows, args.cache_root)
    eval_cov = attach_matcher_features(eval_rows, args.cache_root)
    selection = select_candidate(train_rows, args.folds)
    eval_baseline = metrics_for(eval_rows)
    eval_corrected, eval_pred = apply_selected(train_rows, eval_rows, selection["best"])
    eval_rel = (eval_baseline["angle_mae"] - eval_corrected["angle_mae"]) / eval_baseline["angle_mae"] if eval_baseline["angle_mae"] else 0.0
    decision = "route_b_hold"
    if eval_corrected["angle_mae"] <= args.g1_angle_mae and eval_corrected["distance_mae"] <= args.g3_distance_mae:
        decision = "promote_to_scale_review"
    elif eval_rel <= 0.0:
        decision = "kill_or_redesign"
    result = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "cache_root": str(args.cache_root),
        "train_prediction_csv": [str(path) for path in args.train_prediction_csv],
        "eval_prediction_csv": str(args.eval_prediction_csv),
        "train_matcher_coverage": train_cov,
        "eval_matcher_coverage": eval_cov,
        "selected_candidate": selection["best"],
        "eval_baseline": eval_baseline,
        "eval_corrected": eval_corrected,
        "eval_angle_mae_rel_improvement": eval_rel,
        "decision": decision,
        "gates": {
            "g1_direct_angle": {"threshold": args.g1_angle_mae, "pass": eval_corrected["angle_mae"] <= args.g1_angle_mae},
            "g3_distance": {"threshold": args.g3_distance_mae, "pass": eval_corrected["distance_mae"] <= args.g3_distance_mae},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "candidate_metrics.csv", selection["candidates"])
    write_predictions(args.output_dir / "predictions_val.csv", eval_rows, eval_pred)
    write_report(args.output_dir, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase59 Route B minimal matching-aware residual angle head.")
    parser.add_argument("--train-prediction-csv", type=Path, action="append", required=True)
    parser.add_argument("--eval-prediction-csv", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--g1-angle-mae", type=float, default=0.125347)
    parser.add_argument("--g3-distance-mae", type=float, default=0.043750)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ["selected_candidate", "eval_baseline", "eval_corrected", "decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
