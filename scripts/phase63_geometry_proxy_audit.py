#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def _load_tokens_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "datasets" / "pairuav_correspondence_tokens.py"
    spec = importlib.util.spec_from_file_location("pairuav_correspondence_tokens_g2", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOKENS = _load_tokens_module()


def wrap_deg(values):
    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def angle_abs_error(pred, target):
    return np.abs(wrap_deg(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)))


def circular_offset(source, target):
    delta = np.deg2rad(wrap_deg(np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64)))
    return float(np.rad2deg(math.atan2(float(np.sin(delta).mean()), float(np.cos(delta).mean()))))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def sample_to_match_path(cache_root, pair_id):
    return TOKENS.sample_to_train_match_path(cache_root, pair_id)


def read_predictions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def build_row(row, cache_root, image_size, topk, grid_size, residual_threshold):
    pair_id = row["pair_id"]
    match_path = sample_to_match_path(cache_root, pair_id)
    packet = TOKENS.build_correspondence_token_packet(
        match_path,
        image_size=image_size,
        topk=topk,
        grid_size=grid_size,
        residual_threshold=residual_threshold,
    )
    hypo = np.asarray(packet["hypothesis_features"], dtype=np.float64)
    global_stats = np.asarray(packet["global_stats"], dtype=np.float64)
    target = safe_float(row, "target_heading")
    rank1 = safe_float(row, "rank1_heading")

    trans_tx = float(hypo[0, 0])
    trans_ty = float(hypo[0, 1])
    sim = hypo[1, 2:6].reshape(2, 2)
    aff = hypo[2, 2:6].reshape(2, 2)
    sim_angle = math.degrees(math.atan2(float(sim[1, 0]), float(sim[0, 0]))) if hypo[1, 8] > 0.5 else 0.0
    aff_angle = math.degrees(math.atan2(float(aff[1, 0]), float(aff[0, 0]))) if hypo[2, 8] > 0.5 else 0.0
    trans_angle_xy = math.degrees(math.atan2(trans_ty, trans_tx)) if hypo[0, 8] > 0.5 else 0.0
    trans_angle_yx = math.degrees(math.atan2(trans_tx, trans_ty)) if hypo[0, 8] > 0.5 else 0.0

    result = {
        "pair_id": pair_id,
        "group_id": row.get("group_id", ""),
        "target_heading": target,
        "rank1_heading": rank1,
        "rank1_angle_abs_error": safe_float(row, "rank1_angle_abs_error", angle_abs_error(rank1, target)),
        "match_path": str(match_path),
        "fallback_used": float(packet["fallback_used"]),
        "valid_matches": int(packet["raw_counts"].get("valid_matches", 0)),
        "total_matches": int(packet["raw_counts"].get("total_matches", 0)),
        "translation_angle_xy": wrap_deg(trans_angle_xy).item(),
        "translation_angle_yx": wrap_deg(trans_angle_yx).item(),
        "similarity_rotation": wrap_deg(sim_angle).item(),
        "affine_rotation": wrap_deg(aff_angle).item(),
    }
    for name, value in zip(TOKENS.GLOBAL_FEATURE_NAMES, global_stats):
        result[f"global_{name}"] = float(value)
    for h_idx, h_name in enumerate(TOKENS.HYPOTHESIS_NAMES):
        for f_idx, f_name in enumerate(TOKENS.HYPOTHESIS_FEATURE_NAMES):
            result[f"hyp_{h_name}_{f_name}"] = float(hypo[h_idx, f_idx])
    return result


def kfold_indices(n, folds):
    indices = np.arange(n)
    return [indices[i::folds] for i in range(folds)]


def calibrate_angle_sources(rows, folds):
    target = np.asarray([r["target_heading"] for r in rows], dtype=np.float64)
    rank1 = np.asarray([r["rank1_heading"] for r in rows], dtype=np.float64)
    rank1_err = angle_abs_error(rank1, target)
    source_names = ["translation_angle_xy", "translation_angle_yx", "similarity_rotation", "affine_rotation"]
    fold_indices = kfold_indices(len(rows), folds)
    results = {}
    for source_name in source_names:
        source = np.asarray([r[source_name] for r in rows], dtype=np.float64)
        pred = np.zeros_like(source)
        fold_offsets = []
        for val_idx in fold_indices:
            train_mask = np.ones(len(rows), dtype=bool)
            train_mask[val_idx] = False
            best = None
            for sign in (-1.0, 1.0):
                offset = circular_offset(sign * source[train_mask], target[train_mask])
                train_pred = wrap_deg(sign * source[train_mask] + offset)
                train_mae = float(angle_abs_error(train_pred, target[train_mask]).mean())
                candidate = (train_mae, sign, offset)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            _, sign, offset = best
            pred[val_idx] = wrap_deg(sign * source[val_idx] + offset)
            fold_offsets.append({"sign": sign, "offset": offset})
        source_err = angle_abs_error(pred, target)
        oracle_err = np.minimum(rank1_err, source_err)
        better = source_err < rank1_err
        results[source_name] = {
            "cv_mae": float(source_err.mean()),
            "cv_median_abs_error": float(np.median(source_err)),
            "cv_p90_abs_error": float(np.percentile(source_err, 90)),
            "rank1_mae_same_rows": float(rank1_err.mean()),
            "oracle_min_rank1_or_source_mae": float(oracle_err.mean()),
            "oracle_gain_vs_rank1": float(rank1_err.mean() - oracle_err.mean()),
            "source_better_count": int(better.sum()),
            "source_better_rate": float(better.mean()),
            "fold_offsets": fold_offsets,
        }
        for idx, row in enumerate(rows):
            row[f"{source_name}_cv_pred"] = float(pred[idx])
            row[f"{source_name}_cv_abs_error"] = float(source_err[idx])
            row[f"{source_name}_beats_rank1"] = int(bool(better[idx]))
    return results


def observability_audit(rows, topn=20):
    target_error = np.asarray([r["rank1_angle_abs_error"] for r in rows], dtype=np.float64)
    feature_names = []
    for key in rows[0]:
        if key.startswith("global_") or key.startswith("hyp_") or key in ("valid_matches", "total_matches"):
            feature_names.append(key)
    feature_results = []
    for name in feature_names:
        values = np.asarray([r.get(name, 0.0) for r in rows], dtype=np.float64)
        feature_results.append(
            {
                "feature": name,
                "pearson": pearson(values, target_error),
                "spearman": spearman(values, target_error),
            }
        )
    feature_results.sort(key=lambda item: abs(item["spearman"]), reverse=True)
    hard_threshold = float(np.percentile(target_error, 80))
    hard = target_error >= hard_threshold
    threshold_results = []
    for name in feature_names:
        values = np.asarray([r.get(name, 0.0) for r in rows], dtype=np.float64)
        if not np.isfinite(values).all() or np.max(values) <= np.min(values):
            continue
        for direction in ("ge", "le"):
            best = None
            for quantile in np.linspace(0.1, 0.9, 17):
                threshold = float(np.quantile(values, quantile))
                pred = values >= threshold if direction == "ge" else values <= threshold
                tp = int(np.logical_and(pred, hard).sum())
                fp = int(np.logical_and(pred, ~hard).sum())
                fn = int(np.logical_and(~pred, hard).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                candidate = (f1, precision, recall, threshold, direction, int(pred.sum()))
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best:
                threshold_results.append(
                    {
                        "feature": name,
                        "direction": best[4],
                        "threshold": best[3],
                        "hard_f1": best[0],
                        "hard_precision": best[1],
                        "hard_recall": best[2],
                        "selected": best[5],
                    }
                )
    threshold_results.sort(key=lambda item: item["hard_f1"], reverse=True)
    return {
        "top_correlations": feature_results[:topn],
        "hard_threshold_p80": hard_threshold,
        "hard_rows": int(hard.sum()),
        "top_hard_thresholds": threshold_results[:topn],
    }


def write_joined_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank1-predictions", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--residual-threshold", type=float, default=0.035)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_predictions(args.rank1_predictions)
    rows = [
        build_row(
            row,
            args.cache_root,
            image_size=(args.image_width, args.image_height),
            topk=args.topk,
            grid_size=args.grid_size,
            residual_threshold=args.residual_threshold,
        )
        for row in raw_rows
    ]
    covered_rows = [row for row in rows if row["fallback_used"] < 0.5]
    angle_source_results = calibrate_angle_sources(covered_rows, folds=int(args.folds)) if covered_rows else {}
    observability = observability_audit(covered_rows) if covered_rows else {}
    joined_csv = output_dir / "g2_joined_geometry_rank1.csv"
    write_joined_csv(joined_csv, rows)

    rank1_errors = np.asarray([row["rank1_angle_abs_error"] for row in covered_rows], dtype=np.float64)
    summary = {
        "rank1_predictions": str(Path(args.rank1_predictions)),
        "cache_root": str(Path(args.cache_root)),
        "rows": len(rows),
        "covered": len(covered_rows),
        "fallback": len(rows) - len(covered_rows),
        "coverage_rate": float(len(covered_rows) / len(rows)) if rows else 0.0,
        "topk": int(args.topk),
        "grid_size": int(args.grid_size),
        "residual_threshold": float(args.residual_threshold),
        "rank1_angle_mae_covered": float(rank1_errors.mean()) if rank1_errors.size else None,
        "angle_source_results": angle_source_results,
        "observability": observability,
        "joined_csv": str(joined_csv),
    }
    summary_path = output_dir / "g2_geometry_proxy_audit.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
