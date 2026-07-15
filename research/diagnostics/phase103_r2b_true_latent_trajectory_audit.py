#!/usr/bin/env python3
"""Phase103-R2b true latent trajectory CPU audit.

Consumes feature artifacts from phase103_r2b_extract_latent_features.py and
computes CKA/RSA, alignment residual, kNN overlap/purity, probe, and latent
drift/correction summaries. This is mechanism evidence, not a deployable rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-scope", default="val811")
    parser.add_argument("--reference-case", default="final")
    parser.add_argument("--candidate-cases", default="step350000,step400000,step450000")
    parser.add_argument("--layers", default="6,11,12,mid,late")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--ridge-alphas", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    return parser.parse_args()


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


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


def s(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def rankdata(values):
    import numpy as np

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_np(x, y) -> float:
    import numpy as np

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx.dot(ry) / denom) if denom > 0 else math.nan


def center(x):
    return x - x.mean(axis=0, keepdims=True)


def l2_normalize(x):
    import numpy as np

    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom < 1e-12] = 1.0
    return x / denom


def linear_cka(x, y) -> float:
    import numpy as np

    x = center(x.astype("float64"))
    y = center(y.astype("float64"))
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    num = float((xy * xy).sum())
    den = math.sqrt(float((xx * xx).sum()) * float((yy * yy).sum()))
    return num / den if den > 0 else math.nan


def cosine_distance_vector(x):
    import numpy as np

    z = l2_normalize(x.astype("float64"))
    sim = z @ z.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return 1.0 - sim[iu]


def load_features(feature_dir: Path, sample_scope: str, cases: list[str], layers: list[str]):
    import numpy as np

    manifest = read_csv(feature_dir / "phase103_r2b_feature_manifest.csv")
    out = {}
    manifest_rows = []
    for row in manifest:
        if s(row, "sample_scope") != sample_scope:
            continue
        case = s(row, "checkpoint_case")
        layer = s(row, "layer_name")
        if case not in cases or layer not in layers:
            continue
        path = Path(s(row, "feature_path"))
        if not path.is_absolute():
            path = feature_dir / path
        data = np.load(path)
        out[(case, layer)] = data["features"].astype("float64")
        manifest_rows.append(row)
    return out, manifest_rows


def fold_indices(sample_rows: list[dict[str, str]], fold_id: str, fold_count: int) -> tuple[list[int], list[int]]:
    train: list[int] = []
    eval_idx: list[int] = []
    for idx, row in enumerate(sample_rows):
        row_fold = s(row, "fold_id", str(idx % fold_count))
        if row_fold == fold_id:
            eval_idx.append(idx)
        else:
            train.append(idx)
    return train, eval_idx


def ridge_predict(x_train, y_train, x_eval, alpha: float):
    import numpy as np

    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    xt = (x_train - mean) / scale
    xe = (x_eval - mean) / scale
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    xe = np.concatenate([np.ones((xe.shape[0], 1)), xe], axis=1)
    reg = np.eye(xt.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    return xe @ coef


def r2_score(y_true, y_pred) -> float:
    import numpy as np

    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def build_cka_rsa(features, cases: list[str], layers: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cka_rows: list[dict[str, Any]] = []
    rsa_rows: list[dict[str, Any]] = []
    for layer in layers:
        for i, case_a in enumerate(cases):
            for case_b in cases[i + 1 :]:
                if (case_a, layer) not in features or (case_b, layer) not in features:
                    continue
                x = features[(case_a, layer)]
                y = features[(case_b, layer)]
                count = min(len(x), len(y))
                x = x[:count]
                y = y[:count]
                cka_rows.append(
                    {
                        "layer_name": layer,
                        "checkpoint_a": case_a,
                        "checkpoint_b": case_b,
                        "count": count,
                        "linear_cka": linear_cka(x, y),
                    }
                )
                rsa_rows.append(
                    {
                        "layer_name": layer,
                        "checkpoint_a": case_a,
                        "checkpoint_b": case_b,
                        "count": count,
                        "rsa_spearman_cosine_distance": spearman_np(cosine_distance_vector(x), cosine_distance_vector(y)),
                    }
                )
    return cka_rows, rsa_rows


def build_alignment(features, sample_rows, reference_case: str, candidate_cases: list[str], layers: list[str], alphas: list[float], fold_count: int):
    import numpy as np

    folds = sorted({s(row, "fold_id", str(idx % fold_count)) for idx, row in enumerate(sample_rows)})
    rows: list[dict[str, Any]] = []
    for case in candidate_cases:
        for layer in layers:
            if (case, layer) not in features or (reference_case, layer) not in features:
                continue
            x = features[(case, layer)]
            y = features[(reference_case, layer)]
            count = min(len(x), len(y), len(sample_rows))
            x = x[:count]
            y = y[:count]
            sample_sub = sample_rows[:count]
            for fold in folds:
                train, eval_idx = fold_indices(sample_sub, fold, fold_count)
                if not train or not eval_idx:
                    continue
                best = None
                for alpha in alphas:
                    pred = ridge_predict(x[train], y[train], x[eval_idx], alpha)
                    residual = np.linalg.norm(pred - y[eval_idx], axis=1)
                    denom = np.linalg.norm(y[eval_idx] - y[train].mean(axis=0, keepdims=True), axis=1)
                    rel = float(residual.mean() / max(float(denom.mean()), 1e-12))
                    cos = float((l2_normalize(pred) * l2_normalize(y[eval_idx])).sum(axis=1).mean())
                    key = (rel, -cos)
                    if best is None or key < best[0]:
                        best = (key, alpha, rel, cos, residual)
                if best is None:
                    continue
                rows.append(
                    {
                        "checkpoint_case": case,
                        "reference_case": reference_case,
                        "layer_name": layer,
                        "fold_id": fold,
                        "alpha": best[1],
                        "count": len(eval_idx),
                        "relative_residual": best[2],
                        "cosine_after_alignment": best[3],
                    }
                )
    return rows


def topk_neighbors(x, k: int):
    import numpy as np

    z = l2_normalize(x.astype("float64"))
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, max(sim.shape[0] - 1, 1))
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[row, idx], axis=1)
    return idx[row, order]


def build_knn(features, sample_rows, reference_case: str, candidate_cases: list[str], layers: list[str], k: int):
    import numpy as np

    purity_keys = [
        "best_final_case",
        "best_heading_case",
        "best_range_case",
        "heading_range_best_case_mismatch",
        "pred_heading_bin_idx",
        "pred_range_abs_bucket",
        "pred_range_sign",
        "high_axiswise_headroom_q75",
    ]
    overlap_rows: list[dict[str, Any]] = []
    purity_rows: list[dict[str, Any]] = []
    for layer in layers:
        if (reference_case, layer) not in features:
            continue
        ref = features[(reference_case, layer)]
        count = min(len(ref), len(sample_rows))
        ref_nn = topk_neighbors(ref[:count], k)
        for case in [reference_case, *candidate_cases]:
            if (case, layer) not in features:
                continue
            x = features[(case, layer)][:count]
            nn = topk_neighbors(x, k)
            overlaps = [len(set(nn[i]).intersection(set(ref_nn[i]))) / nn.shape[1] for i in range(count)]
            overlap_rows.append(
                {
                    "checkpoint_case": case,
                    "reference_case": reference_case,
                    "layer_name": layer,
                    "count": count,
                    "k": nn.shape[1],
                    "mean_neighbor_overlap_with_final": float(np.mean(overlaps)),
                }
            )
            for key in purity_keys:
                labels = [s(row, key) for row in sample_rows[:count]]
                valid = [idx for idx, value in enumerate(labels) if value != ""]
                if not valid:
                    continue
                same_rates = []
                for idx in valid:
                    denom = 0
                    same = 0
                    for nbr in nn[idx]:
                        if labels[nbr] == "":
                            continue
                        denom += 1
                        same += int(labels[nbr] == labels[idx])
                    if denom:
                        same_rates.append(same / denom)
                if same_rates:
                    purity_rows.append(
                        {
                            "checkpoint_case": case,
                            "layer_name": layer,
                            "label_key": key,
                            "count": len(same_rates),
                            "k": nn.shape[1],
                            "neighbor_same_label_rate": float(np.mean(same_rates)),
                        }
                    )
    return overlap_rows, purity_rows


def add_headroom_flag(sample_rows: list[dict[str, str]]) -> None:
    values = sorted(f(row, "baseline_minus_axiswise_oracle") for row in sample_rows if f(row, "baseline_minus_axiswise_oracle") > 0)
    q75 = values[int(0.75 * (len(values) - 1))] if values else math.inf
    for row in sample_rows:
        row["high_axiswise_headroom_q75"] = str(int(f(row, "baseline_minus_axiswise_oracle") >= q75))


def build_probe_summary(features, sample_rows, cases: list[str], layers: list[str], alphas: list[float], fold_count: int, shuffle_repeats: int):
    import numpy as np

    rng = np.random.default_rng(103)
    targets = [
        "best_final_case",
        "best_heading_case",
        "best_range_case",
        "heading_range_best_case_mismatch",
        "high_axiswise_headroom_q75",
        "pred_heading_bin_idx",
        "pred_range_abs_bucket",
        "pred_range_sign",
        "baseline_minus_axiswise_oracle",
    ]
    rows: list[dict[str, Any]] = []
    folds = sorted({s(row, "fold_id", str(idx % fold_count)) for idx, row in enumerate(sample_rows)})
    for case in cases:
        for layer in layers:
            if (case, layer) not in features:
                continue
            x_all = features[(case, layer)][: len(sample_rows)]
            for target in targets:
                labels = [s(row, target) for row in sample_rows[: len(x_all)]]
                numeric = target == "baseline_minus_axiswise_oracle"
                if numeric:
                    y_raw = np.asarray([f(row, target) for row in sample_rows[: len(x_all)]], dtype="float64")
                    if not np.isfinite(y_raw).all():
                        continue
                    y_matrix = y_raw[:, None]
                    baseline = 0.0
                    metric_name = "oof_r2"
                else:
                    classes = sorted({value for value in labels if value != ""})
                    if len(classes) < 2:
                        continue
                    class_to_idx = {value: idx for idx, value in enumerate(classes)}
                    y_idx = np.asarray([class_to_idx[value] for value in labels], dtype="int64")
                    y_matrix = np.zeros((len(y_idx), len(classes)), dtype="float64")
                    y_matrix[np.arange(len(y_idx)), y_idx] = 1.0
                    baseline = Counter(y_idx.tolist()).most_common(1)[0][1] / len(y_idx)
                    metric_name = "oof_accuracy"

                def score_for(y_mat, y_idx_or_raw):
                    pred = np.zeros_like(y_mat, dtype="float64")
                    for fold in folds:
                        train, eval_idx = fold_indices(sample_rows[: len(x_all)], fold, fold_count)
                        if not train or not eval_idx:
                            continue
                        best_pred = None
                        best_train = -math.inf
                        for alpha in alphas:
                            train_pred = ridge_predict(x_all[train], y_mat[train], x_all[train], alpha)
                            if numeric:
                                train_score = r2_score(y_mat[train], train_pred)
                            else:
                                train_score = float((train_pred.argmax(axis=1) == y_idx_or_raw[train]).mean())
                            if train_score > best_train:
                                best_train = train_score
                                best_pred = ridge_predict(x_all[train], y_mat[train], x_all[eval_idx], alpha)
                        pred[eval_idx] = best_pred
                    if numeric:
                        return r2_score(y_mat, pred)
                    return float((pred.argmax(axis=1) == y_idx_or_raw).mean())

                actual = score_for(y_matrix, y_raw if numeric else y_idx)
                shuffles = []
                for _ in range(shuffle_repeats):
                    if numeric:
                        y_shuf = rng.permutation(y_matrix[:, 0])[:, None]
                        shuffles.append(score_for(y_shuf, y_shuf[:, 0]))
                    else:
                        shuffled_idx = rng.permutation(y_idx)
                        y_shuf = np.zeros_like(y_matrix)
                        y_shuf[np.arange(len(shuffled_idx)), shuffled_idx] = 1.0
                        shuffles.append(score_for(y_shuf, shuffled_idx))
                rows.append(
                    {
                        "checkpoint_case": case,
                        "layer_name": layer,
                        "target_name": target,
                        "metric_name": metric_name,
                        "count": len(x_all),
                        "oof_metric": actual,
                        "majority_or_mean_baseline": baseline,
                        "shuffle_mean": float(np.mean(shuffles)) if shuffles else math.nan,
                        "shuffle_std": float(np.std(shuffles)) if shuffles else math.nan,
                    }
                )
    return rows


def candidate_benefit(row: dict[str, str], case: str) -> float:
    key = f"traj_final_minus_{case}_error"
    if key in row:
        return f(row, key)
    error_key = f"final_error_{case}"
    if error_key in row:
        return f(row, "baseline_final_error") - f(row, error_key)
    return math.nan


def build_latent_drift(features, sample_rows, reference_case: str, candidate_cases: list[str], layers: list[str], alphas: list[float], fold_count: int):
    import numpy as np

    rows: list[dict[str, Any]] = []
    folds = sorted({s(row, "fold_id", str(idx % fold_count)) for idx, row in enumerate(sample_rows)})
    range_span = f(sample_rows[0], "range_span", 264.0) if sample_rows else 264.0
    for case in candidate_cases:
        for layer in layers:
            if (case, layer) not in features or (reference_case, layer) not in features:
                continue
            count = min(len(features[(case, layer)]), len(features[(reference_case, layer)]), len(sample_rows))
            z_delta = features[(case, layer)][:count] - features[(reference_case, layer)][:count]
            latent_norm = np.linalg.norm(z_delta, axis=1)
            output = []
            correction = []
            benefits = []
            for row in sample_rows[:count]:
                final_h = f(row, f"pred_heading_{reference_case}", f(row, "pred_heading_deg"))
                final_r = f(row, f"pred_range_{reference_case}", f(row, "pred_range"))
                cand_h = f(row, f"pred_heading_{case}", math.nan)
                cand_r = f(row, f"pred_range_{case}", math.nan)
                output.append([
                    wrap_angle_diff_deg(cand_h, final_h) / 180.0,
                    (cand_r - final_r) / max(range_span, 1e-12),
                ])
                correction.append([
                    wrap_angle_diff_deg(f(row, "true_heading_deg"), final_h) / 180.0,
                    (f(row, "true_range") - final_r) / max(range_span, 1e-12),
                ])
                benefits.append(candidate_benefit(row, case))
            output_y = np.asarray(output, dtype="float64")
            correction_y = np.asarray(correction, dtype="float64")
            benefits_np = np.asarray(benefits, dtype="float64")

            def oof_r2(y):
                pred = np.zeros_like(y)
                for fold in folds:
                    train, eval_idx = fold_indices(sample_rows[:count], fold, fold_count)
                    if not train or not eval_idx:
                        continue
                    best = None
                    for alpha in alphas:
                        train_pred = ridge_predict(z_delta[train], y[train], z_delta[train], alpha)
                        score = r2_score(y[train], train_pred)
                        if best is None or score > best[0]:
                            best = (score, alpha)
                    pred[eval_idx] = ridge_predict(z_delta[train], y[train], z_delta[eval_idx], best[1])
                return r2_score(y, pred)

            valid_benefit = np.isfinite(benefits_np)
            rows.append(
                {
                    "checkpoint_case": case,
                    "reference_case": reference_case,
                    "layer_name": layer,
                    "count": count,
                    "latent_norm_spearman_candidate_benefit": spearman_np(latent_norm[valid_benefit], benefits_np[valid_benefit]) if valid_benefit.sum() >= 3 else math.nan,
                    "latent_norm_spearman_output_move_norm": spearman_np(latent_norm, np.linalg.norm(output_y, axis=1)),
                    "latent_to_output_oof_r2": oof_r2(output_y),
                    "latent_to_true_correction_oof_r2_diagnostic": oof_r2(correction_y),
                    "latent_bucket_candidate_benefit_gap": float(np.nanmean(benefits_np[latent_norm >= np.nanmedian(latent_norm)]) - np.nanmean(benefits_np[latent_norm < np.nanmedian(latent_norm)])) if valid_benefit.any() else math.nan,
                    "uses_true_labels_for_diagnostic": True,
                }
            )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase103-R2b True Latent Trajectory Audit",
        "",
        f"- feature_dir: `{summary['feature_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- sample_scope: `{summary['sample_scope']}`",
        f"- checkpoint_cases: `{', '.join(summary['checkpoint_cases'])}`",
        f"- layers: `{', '.join(summary['layers'])}`",
        "",
        "## Gates",
        "",
        f"- Gate A real latent change: `{summary['gate_a_real_latent_change']}`",
        f"- Gate B regime structured: `{summary['gate_b_regime_structured']}`",
        f"- Gate C correction flow justified: `{summary['gate_c_correction_flow_justified']}`",
        f"- R3 allowed: `{summary['r3_allowed']}`",
        f"- Paper story recommendation: `{summary['paper_story_recommendation']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    feature_dir = Path(args.feature_dir)
    candidate_cases = parse_str_list(args.candidate_cases)
    checkpoint_cases = [args.reference_case, *candidate_cases]
    layers = parse_str_list(args.layers)
    alphas = parse_float_list(args.ridge_alphas)
    sample_rows = read_csv(feature_dir / "phase103_r2b_sample_manifest.csv")
    sample_rows = [row for row in sample_rows if s(row, "sample_scope", args.sample_scope) == args.sample_scope]
    add_headroom_flag(sample_rows)
    features, manifest_rows = load_features(feature_dir, args.sample_scope, checkpoint_cases, layers)

    cka_rows, rsa_rows = build_cka_rsa(features, checkpoint_cases, layers)
    alignment_rows = build_alignment(features, sample_rows, args.reference_case, candidate_cases, layers, alphas, args.fold_count)
    knn_rows, purity_rows = build_knn(features, sample_rows, args.reference_case, candidate_cases, layers, args.knn_k)
    probe_rows = build_probe_summary(features, sample_rows, checkpoint_cases, layers, alphas, args.fold_count, args.shuffle_repeats)
    drift_rows = build_latent_drift(features, sample_rows, args.reference_case, candidate_cases, layers, alphas, args.fold_count)

    gate_a = "pass" if any(f(row, "linear_cka", 1.0) < 0.98 for row in cka_rows) or any(f(row, "mean_neighbor_overlap_with_final", 1.0) < 0.75 for row in knn_rows if s(row, "checkpoint_case") != args.reference_case) else "weak"
    gate_b = "pass" if any(f(row, "oof_metric") > max(f(row, "majority_or_mean_baseline"), f(row, "shuffle_mean")) + 0.03 for row in probe_rows if s(row, "target_name") in {"best_final_case", "heading_range_best_case_mismatch", "high_axiswise_headroom_q75"}) else "weak"
    gate_c = "pass" if any(f(row, "latent_to_true_correction_oof_r2_diagnostic") > 0.05 or f(row, "latent_to_output_oof_r2") > 0.05 for row in drift_rows) else "fail"
    r3_allowed = gate_a in {"pass", "weak"} and gate_b == "pass" and gate_c in {"pass", "weak"}
    paper_story = "correction_flow" if gate_c == "pass" and gate_b == "pass" else "latent_manifold" if gate_a == "pass" and gate_b == "pass" else "axis_async_feature_head"

    bucket_rows: list[dict[str, Any]] = []
    for key in ["best_final_case", "heading_range_best_case_mismatch", "high_axiswise_headroom_q75", "pred_heading_bin_idx", "pred_range_abs_bucket", "pred_range_sign"]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sample_rows:
            grouped[s(row, key)].append(row)
        for value, rows in sorted(grouped.items()):
            if value == "":
                continue
            bucket_rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "count": len(rows),
                    "mean_baseline_minus_axiswise_oracle": sum(f(row, "baseline_minus_axiswise_oracle") for row in rows) / max(len(rows), 1),
                    "mean_baseline_minus_best_final_error": sum(f(row, "baseline_minus_best_final_error") for row in rows) / max(len(rows), 1),
                }
            )

    write_csv(output_dir / "phase103_r2b_cka_summary.csv", cka_rows)
    write_csv(output_dir / "phase103_r2b_rsa_summary.csv", rsa_rows)
    write_csv(output_dir / "phase103_r2b_alignment_residual_summary.csv", alignment_rows)
    write_csv(output_dir / "phase103_r2b_knn_overlap_summary.csv", knn_rows)
    write_csv(output_dir / "phase103_r2b_regime_purity_summary.csv", purity_rows)
    write_csv(output_dir / "phase103_r2b_probe_summary.csv", probe_rows)
    write_csv(output_dir / "phase103_r2b_latent_drift_correction_summary.csv", drift_rows)
    write_csv(output_dir / "phase103_r2b_bucket_summary.csv", bucket_rows)
    summary = {
        "phase": "phase103_r2b_true_latent_trajectory_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "sample_scope": args.sample_scope,
        "checkpoint_cases": checkpoint_cases,
        "layers": layers,
        "feature_manifest_rows": len(manifest_rows),
        "sample_rows": len(sample_rows),
        "cka_rows": len(cka_rows),
        "rsa_rows": len(rsa_rows),
        "alignment_rows": len(alignment_rows),
        "knn_rows": len(knn_rows),
        "purity_rows": len(purity_rows),
        "probe_rows": len(probe_rows),
        "drift_rows": len(drift_rows),
        "bucket_rows": len(bucket_rows),
        "gate_a_real_latent_change": gate_a,
        "gate_b_regime_structured": gate_b,
        "gate_c_correction_flow_justified": gate_c,
        "r3_allowed": r3_allowed,
        "paper_story_recommendation": paper_story,
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "phase103_r2b_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_r2b_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

