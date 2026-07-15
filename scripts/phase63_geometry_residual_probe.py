#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def wrap_deg(values):
    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def angle_abs_error(pred, target):
    return np.abs(wrap_deg(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)))


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def standardize(train_x, eval_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std


def fit_ridge(x, y, alpha):
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def predict_ridge(x, coef):
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ coef


def feature_matrix(rows, feature_set):
    base_names = []
    for key in rows[0].keys():
        if key.startswith("global_") or key.startswith("hyp_") or key in ("valid_matches", "total_matches"):
            base_names.append(key)
    base_names = sorted(base_names)
    columns = []
    names = []
    if feature_set in ("quality", "all"):
        for name in base_names:
            columns.append([f(row, name) for row in rows])
            names.append(name)
    if feature_set in ("angles", "all"):
        angle_names = ["translation_angle_xy", "translation_angle_yx", "similarity_rotation", "affine_rotation"]
        for name in angle_names:
            values = np.asarray([f(row, name) for row in rows], dtype=np.float64)
            rank1 = np.asarray([f(row, "rank1_heading") for row in rows], dtype=np.float64)
            for suffix, angle_values in (
                ("source", values),
                ("source_minus_rank1", wrap_deg(values - rank1)),
            ):
                radians = np.deg2rad(angle_values)
                columns.append(np.sin(radians).tolist())
                names.append(f"{name}_{suffix}_sin")
                columns.append(np.cos(radians).tolist())
                names.append(f"{name}_{suffix}_cos")
    if feature_set in ("rank1_context", "all"):
        rank1 = np.asarray([f(row, "rank1_heading") for row in rows], dtype=np.float64)
        radians = np.deg2rad(rank1)
        columns.append(np.sin(radians).tolist())
        names.append("rank1_heading_sin")
        columns.append(np.cos(radians).tolist())
        names.append("rank1_heading_cos")
    matrix = np.asarray(columns, dtype=np.float64).T
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def split_indices(n, folds):
    idx = np.arange(n)
    return [idx[i::folds] for i in range(folds)]


def select_config(x, y, train_idx, alphas, clips):
    inner_val = train_idx[::5]
    inner_train = np.setdiff1d(train_idx, inner_val, assume_unique=False)
    best = None
    for alpha in alphas:
        for clip in clips:
            x_train, x_val = standardize(x[inner_train], x[inner_val])
            coef = fit_ridge(x_train, y[inner_train], alpha)
            pred = np.clip(predict_ridge(x_val, coef), -clip, clip)
            mae = float(np.abs(pred - y[inner_val]).mean())
            candidate = (mae, alpha, clip)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return {"inner_mae": best[0], "alpha": best[1], "clip": best[2]}


def run_probe(rows, feature_set, folds=5):
    x, feature_names = feature_matrix(rows, feature_set)
    target = np.asarray([f(row, "target_heading") for row in rows], dtype=np.float64)
    rank1 = np.asarray([f(row, "rank1_heading") for row in rows], dtype=np.float64)
    residual = wrap_deg(target - rank1)
    rank1_err = angle_abs_error(rank1, target)
    pred_residual = np.zeros(len(rows), dtype=np.float64)
    configs = []
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    clips = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    for val_idx in split_indices(len(rows), folds):
        train_idx = np.setdiff1d(np.arange(len(rows)), val_idx, assume_unique=False)
        config = select_config(x, residual, train_idx, alphas, clips)
        x_train, x_val = standardize(x[train_idx], x[val_idx])
        coef = fit_ridge(x_train, residual[train_idx], config["alpha"])
        pred_residual[val_idx] = np.clip(predict_ridge(x_val, coef), -config["clip"], config["clip"])
        configs.append(config)
    corrected = wrap_deg(rank1 + pred_residual)
    corrected_err = angle_abs_error(corrected, target)
    hard_threshold = float(np.percentile(rank1_err, 80))
    hard = rank1_err >= hard_threshold
    return {
        "feature_set": feature_set,
        "feature_dim": int(x.shape[1]),
        "feature_names": feature_names,
        "rank1_mae": float(rank1_err.mean()),
        "corrected_mae": float(corrected_err.mean()),
        "delta_mae": float(corrected_err.mean() - rank1_err.mean()),
        "rank1_hard_mae_p80": float(rank1_err[hard].mean()),
        "corrected_hard_mae_p80": float(corrected_err[hard].mean()),
        "delta_hard_mae_p80": float(corrected_err[hard].mean() - rank1_err[hard].mean()),
        "improved_count": int((corrected_err < rank1_err).sum()),
        "worsened_count": int((corrected_err > rank1_err).sum()),
        "mean_abs_pred_residual": float(np.abs(pred_residual).mean()),
        "p90_abs_pred_residual": float(np.percentile(np.abs(pred_residual), 90)),
        "fold_configs": configs,
    }, pred_residual, corrected_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = read_rows(args.joined_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    per_row = {row["pair_id"]: dict(row) for row in rows}
    for feature_set in ("quality", "angles", "rank1_context", "all"):
        result, pred_residual, corrected_err = run_probe(rows, feature_set, folds=args.folds)
        results.append(result)
        for row, pred, err in zip(rows, pred_residual, corrected_err):
            per_row[row["pair_id"]][f"{feature_set}_pred_residual"] = float(pred)
            per_row[row["pair_id"]][f"{feature_set}_corrected_abs_error"] = float(err)

    summary = {
        "joined_csv": str(Path(args.joined_csv)),
        "rows": len(rows),
        "folds": int(args.folds),
        "results": results,
    }
    (output_dir / "g2b_residual_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(next(iter(per_row.values())).keys())
    with (output_dir / "g2b_residual_probe_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_row.values():
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
