#!/usr/bin/env python3
"""Phase103-E2 train-only correction-direction estimator.

Consumes Phase103-E0 train-hash trajectory artifacts and evaluates deployable
post-hoc policies with deterministic OOF folds. Hidden-test labels, val811
oracle labels, and leaderboard feedback are not used.
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


FORBIDDEN_INPUT_PATTERNS = (
    "true_",
    "_error",
    "error_",
    "best_",
    "baseline_minus_",
    "axiswise_",
    "benefit",
    "positive_benefit",
    "candidate_correction_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-cases", default="final,step350000,step400000,step450000")
    parser.add_argument("--reference-case", default="final")
    parser.add_argument("--range-span", type=float, default=264.0)
    parser.add_argument("--ridge-alphas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--penalties", default="0,0.01,0.03,0.1,0.3,1")
    parser.add_argument("--threshold-quantiles", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--max-rows", type=int, default=0)
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


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def safe_quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def score_pred(pred_h: float, pred_r: float, true_h: float, true_r: float, range_span: float) -> tuple[float, float, float]:
    heading_rel = abs(wrap_angle_diff_deg(pred_h, true_h)) / 180.0
    range_rel = abs(pred_r - true_r) / max(range_span, 1e-12)
    return heading_rel, range_rel, 0.5 * (heading_rel + range_rel)


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


def index_targets(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        out[(s(row, "sample_index"), s(row, "candidate_case"))] = row
    return out


def assert_required_files(e0_dir: Path) -> None:
    for name in [
        "phase103_e0_case_manifest.csv",
        "phase103_e0_per_sample_features.csv",
        "phase103_e0_targets.csv",
        "phase103_e0_split_manifest.csv",
    ]:
        path = e0_dir / name
        if not path.exists():
            raise FileNotFoundError(path)


def one_hot(prefix: str, value: str, values: list[str]) -> dict[str, float]:
    return {f"{prefix}_{item}": 1.0 if value == item else 0.0 for item in values}


def build_sample_feature_record(
    row: dict[str, str],
    candidate_case: str,
    target_row: dict[str, str],
    range_buckets: list[str],
    range_signs: list[str],
    range_span: float,
    reference_case: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    final_heading = f(row, f"pred_heading_{reference_case}")
    final_range = f(row, f"pred_range_{reference_case}")
    heading_bin_idx = int(f(row, "pred_heading_bin_idx", 0.0))
    range_bucket = s(row, "pred_range_abs_bucket")
    range_sign = s(row, "pred_range_sign")
    cand_delta_h = f(target_row, "candidate_delta_heading_norm", 0.0)
    cand_delta_r = f(target_row, "candidate_delta_range_norm", 0.0)
    movement_norm = math.sqrt(cand_delta_h * cand_delta_h + cand_delta_r * cand_delta_r)
    values: dict[str, float] = {
        "feature__bias": 1.0,
        "feature__pred_heading_final_sin": math.sin(math.radians(final_heading)),
        "feature__pred_heading_final_cos": math.cos(math.radians(final_heading)),
        "feature__pred_range_final_norm": final_range / max(range_span, 1e-12),
        "feature__late_heading_circ_std_norm": f(row, "late_heading_circ_std_deg", 0.0) / 180.0,
        "feature__late_range_std_norm": f(row, "late_range_std", 0.0) / max(range_span, 1e-12),
        "feature__full_heading_circ_std_norm": f(row, "full_heading_circ_std_deg", 0.0) / 180.0,
        "feature__full_range_std_norm": f(row, "full_range_std", 0.0) / max(range_span, 1e-12),
        "feature__candidate_step_norm": case_step(candidate_case) / 459999.0,
        "feature__is_late_candidate": f(target_row, "is_late_candidate", float(case_step(candidate_case) >= 350000)),
        "feature__candidate_delta_heading_norm": cand_delta_h,
        "feature__candidate_delta_range_norm": cand_delta_r,
        "feature__candidate_movement_norm": movement_norm,
        "feature__candidate_delta_heading_abs": abs(cand_delta_h),
        "feature__candidate_delta_range_abs": abs(cand_delta_r),
    }
    for idx in range(8):
        values[f"feature__pred_heading_bin_{idx}"] = 1.0 if heading_bin_idx == idx else 0.0
    values.update(one_hot("feature__pred_range_abs_bucket", range_bucket, range_buckets))
    values.update(one_hot("feature__pred_range_sign", range_sign, range_signs))
    for family in ["final", "early_050_200k", "mid_250_300k", "late_350_450k"]:
        values[f"feature__candidate_family_{family}"] = 1.0 if case_family(candidate_case) == family else 0.0

    sources = {
        "feature__bias": "constant",
        "feature__pred_heading_final_sin": f"pred_heading_{reference_case}",
        "feature__pred_heading_final_cos": f"pred_heading_{reference_case}",
        "feature__pred_range_final_norm": f"pred_range_{reference_case}",
        "feature__late_heading_circ_std_norm": "late_heading_circ_std_deg",
        "feature__late_range_std_norm": "late_range_std",
        "feature__full_heading_circ_std_norm": "full_heading_circ_std_deg",
        "feature__full_range_std_norm": "full_range_std",
        "feature__candidate_step_norm": "candidate_case",
        "feature__is_late_candidate": "phase103_e0_targets.is_late_candidate",
        "feature__candidate_delta_heading_norm": "phase103_e0_targets.candidate_delta_heading_norm",
        "feature__candidate_delta_range_norm": "phase103_e0_targets.candidate_delta_range_norm",
        "feature__candidate_movement_norm": "phase103_e0_targets.candidate_delta_heading_norm,candidate_delta_range_norm",
        "feature__candidate_delta_heading_abs": "phase103_e0_targets.candidate_delta_heading_norm",
        "feature__candidate_delta_range_abs": "phase103_e0_targets.candidate_delta_range_norm",
    }
    for name in values:
        sources.setdefault(name, "derived_prediction_only")
    return values, sources


def validate_feature_manifest(rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for row in rows:
        if row.get("feature_role") != "input":
            continue
        haystack = f"{row.get('feature_name', '')} {row.get('source_column', '')}".lower()
        for pattern in FORBIDDEN_INPUT_PATTERNS:
            if pattern in haystack:
                violations.append(f"{row.get('feature_name')} from {row.get('source_column')} matched {pattern}")
    return violations


def build_design_matrices(
    feature_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    x = [[float(row[name]) for name in feature_names] for row in feature_rows]
    y = [
        [
            float(row["true_correction_heading_norm"]),
            float(row["true_correction_range_norm"]),
        ]
        for row in feature_rows
    ]
    sample_indices = [int(float(row["sample_index"])) for row in feature_rows]
    return x, y, sample_indices


def standardize_train_eval(
    x_train: list[list[float]],
    x_eval: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    if not x_train:
        return [], [], [], []
    cols = len(x_train[0])
    mean = [sum(row[col] for row in x_train) / len(x_train) for col in range(cols)]
    scale: list[float] = []
    for col in range(cols):
        var = sum((row[col] - mean[col]) ** 2 for row in x_train) / len(x_train)
        std = math.sqrt(var)
        scale.append(std if std >= 1e-8 else 1.0)

    def transform(rows: list[list[float]]) -> list[list[float]]:
        return [[(row[col] - mean[col]) / scale[col] for col in range(cols)] for row in rows]

    return transform(x_train), transform(x_eval), mean, scale


def solve_linear_system(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    if n == 0:
        return []
    m = len(rhs[0]) if rhs else 0
    aug = [list(matrix[i]) + list(rhs[i]) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + m):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) <= 1e-18:
                continue
            for j in range(col, n + m):
                aug[row][j] -= factor * aug[col][j]
    return [aug[row][n:] for row in range(n)]


def ridge_fit_predict(
    x_train: list[list[float]],
    y_train: list[list[float]],
    x_eval: list[list[float]],
    alpha: float,
) -> tuple[list[list[float]], list[list[float]]]:
    if not x_train:
        return [], []
    x_aug = [[1.0, *row] for row in x_train]
    e_aug = [[1.0, *row] for row in x_eval]
    cols = len(x_aug[0])
    targets = len(y_train[0]) if y_train else 0
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [[0.0 for _ in range(targets)] for _ in range(cols)]
    for row, y_row in zip(x_aug, y_train):
        for i in range(cols):
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]
            for t in range(targets):
                xty[i][t] += row[i] * y_row[t]
    for i in range(1, cols):
        xtx[i][i] += alpha
    coef = solve_linear_system(xtx, xty)
    preds = [
        [sum(row[col] * coef[col][target] for col in range(cols)) for target in range(targets)]
        for row in e_aug
    ]
    return preds, coef


def mean_case_error(rows: list[dict[str, str]], indices: list[int], case_id: str, range_span: float, objective: str) -> float:
    values: list[float] = []
    for idx in indices:
        row = rows[idx]
        h, r, final = score_pred(
            f(row, f"pred_heading_{case_id}"),
            f(row, f"pred_range_{case_id}"),
            f(row, "true_heading_deg"),
            f(row, "true_range"),
            range_span,
        )
        values.append(h if objective == "heading" else r if objective == "range" else final)
    return safe_mean(values)


def prediction_row(
    policy_name: str,
    fold_id: str,
    row: dict[str, str],
    pred_heading: float,
    pred_range: float,
    selected_heading_case: str,
    selected_range_case: str,
    selected_joint_case: str,
    range_span: float,
    deployable: bool,
    fit_desc: str,
) -> dict[str, Any]:
    heading_rel, range_rel, final_error = score_pred(
        pred_heading,
        pred_range,
        f(row, "true_heading_deg"),
        f(row, "true_range"),
        range_span,
    )
    baseline_error = f(row, "baseline_final_error")
    return {
        "policy_name": policy_name,
        "deployable": int(deployable),
        "fold_id": fold_id,
        "fit_desc": fit_desc,
        "sample_index": row["sample_index"],
        "group_id": row.get("group_id", ""),
        "json_id": row.get("json_id", ""),
        "selected_heading_case": selected_heading_case,
        "selected_range_case": selected_range_case,
        "selected_joint_case": selected_joint_case,
        "pred_heading_deg": pred_heading,
        "pred_range": pred_range,
        "heading_rel_error": heading_rel,
        "range_rel_error": range_rel,
        "final_error": final_error,
        "baseline_final_error": baseline_error,
        "improvement_vs_baseline": baseline_error - final_error,
        "selected_non_final": int(selected_heading_case != "final" or selected_range_case != "final"),
    }


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [f(row, "improvement_vs_baseline") for row in rows]
    selected = [f(row, "selected_non_final", 0.0) for row in rows]
    return {
        "policy_name": s(rows[0], "policy_name") if rows else "",
        "deployable": int(f(rows[0], "deployable", 0.0)) if rows else 0,
        "count": len(rows),
        "mean_final_error": safe_mean([f(row, "final_error") for row in rows]),
        "mean_heading_rel_error": safe_mean([f(row, "heading_rel_error") for row in rows]),
        "mean_range_rel_error": safe_mean([f(row, "range_rel_error") for row in rows]),
        "mean_baseline_final_error": safe_mean([f(row, "baseline_final_error") for row in rows]),
        "mean_improvement_vs_baseline": safe_mean(improvements),
        "improve_rate": safe_mean([1.0 if value > 0.0 else 0.0 for value in improvements]),
        "selection_rate": safe_mean(selected),
        "fallback_rate": 1.0 - safe_mean(selected) if selected else math.nan,
    }


def run_tier0(
    sample_rows: list[dict[str, str]],
    candidate_cases: list[str],
    folds: list[str],
    range_span: float,
    reference_case: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fold in folds:
        train_idx = [idx for idx, row in enumerate(sample_rows) if s(row, "fold_id") != fold]
        eval_idx = [idx for idx, row in enumerate(sample_rows) if s(row, "fold_id") == fold]

        best_case = min(candidate_cases, key=lambda case: mean_case_error(sample_rows, train_idx, case, range_span, "final"))
        best_heading_case = min(candidate_cases, key=lambda case: mean_case_error(sample_rows, train_idx, case, range_span, "heading"))
        best_range_case = min(candidate_cases, key=lambda case: mean_case_error(sample_rows, train_idx, case, range_span, "range"))

        for idx in eval_idx:
            row = sample_rows[idx]
            out.append(
                prediction_row(
                    "baseline_final",
                    fold,
                    row,
                    f(row, f"pred_heading_{reference_case}"),
                    f(row, f"pred_range_{reference_case}"),
                    reference_case,
                    reference_case,
                    reference_case,
                    range_span,
                    True,
                    "fixed_final",
                )
            )
            out.append(
                prediction_row(
                    "fixed_best_checkpoint_late4_oof",
                    fold,
                    row,
                    f(row, f"pred_heading_{best_case}"),
                    f(row, f"pred_range_{best_case}"),
                    best_case,
                    best_case,
                    best_case,
                    range_span,
                    True,
                    f"train_best={best_case}",
                )
            )
            out.append(
                prediction_row(
                    "fixed_best_heading_range_axis_late4_oof",
                    fold,
                    row,
                    f(row, f"pred_heading_{best_heading_case}"),
                    f(row, f"pred_range_{best_range_case}"),
                    best_heading_case,
                    best_range_case,
                    "",
                    range_span,
                    True,
                    f"heading={best_heading_case};range={best_range_case}",
                )
            )
            oracle_case = min(candidate_cases, key=lambda case: f(row, f"final_error_{case}", math.inf))
            out.append(
                prediction_row(
                    "positive_benefit_oracle_late4_upper_bound",
                    fold,
                    row,
                    f(row, f"pred_heading_{oracle_case}"),
                    f(row, f"pred_range_{oracle_case}"),
                    oracle_case,
                    oracle_case,
                    oracle_case,
                    range_span,
                    False,
                    "label_oracle_do_not_deploy",
                )
            )
    return out


def choose_policy_from_scores(
    rows_by_sample: dict[str, dict[str, Any]],
    candidate_cases: list[str],
    score_by_key: dict[tuple[str, str], float],
    threshold: float,
    reference_case: str,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    non_final = [case for case in candidate_cases if case != reference_case]
    for sample_index in rows_by_sample:
        best_case = reference_case
        best_score = -math.inf
        for case in non_final:
            score = score_by_key.get((sample_index, case), -math.inf)
            if score > best_score:
                best_score = score
                best_case = case
        selected[sample_index] = best_case if best_score > threshold else reference_case
    return selected


def evaluate_selected_cases(
    sample_rows_by_index: dict[str, dict[str, str]],
    selected: dict[str, str],
    indices: list[str],
    range_span: float,
    policy_name: str,
    fold_id: str,
    fit_desc: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_index in indices:
        row = sample_rows_by_index[sample_index]
        case = selected[sample_index]
        out.append(
            prediction_row(
                policy_name,
                fold_id,
                row,
                f(row, f"pred_heading_{case}"),
                f(row, f"pred_range_{case}"),
                case,
                case,
                case,
                range_span,
                True,
                fit_desc,
            )
        )
    return out


def run_tier1(
    sample_rows: list[dict[str, str]],
    candidate_feature_rows: list[dict[str, Any]],
    feature_names: list[str],
    candidate_cases: list[str],
    folds: list[str],
    alphas: list[float],
    penalties: list[float],
    quantiles: list[float],
    range_span: float,
    reference_case: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows_by_index = {s(row, "sample_index"): row for row in sample_rows}
    sample_feature_rows = [row for row in candidate_feature_rows if s(row, "candidate_case") == reference_case]
    sample_feature_by_index = {s(row, "sample_index"): row for row in sample_feature_rows}
    x_all, y_all, sample_indices_np = build_design_matrices(sample_feature_rows, feature_names)
    row_pos_by_sample = {str(int(idx)): pos for pos, idx in enumerate(sample_indices_np)}
    candidate_rows_by_sample: dict[str, dict[str, Any]] = {s(row, "sample_index"): row for row in sample_feature_rows}
    fold_records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold in folds:
        train_samples = [s(row, "sample_index") for row in sample_rows if s(row, "fold_id") != fold]
        eval_samples = [s(row, "sample_index") for row in sample_rows if s(row, "fold_id") == fold]
        train_pos = [row_pos_by_sample[idx] for idx in train_samples]
        eval_pos = [row_pos_by_sample[idx] for idx in eval_samples]
        x_train_raw = [x_all[idx] for idx in train_pos]
        y_train = [y_all[idx] for idx in train_pos]
        x_eval_raw = [x_all[idx] for idx in eval_pos]

        best_train_error = math.inf
        best_payload: dict[str, Any] | None = None
        for alpha in alphas:
            x_train, x_eval, mean_vec, scale_vec = standardize_train_eval(x_train_raw, x_eval_raw)
            train_pred, coef = ridge_fit_predict(x_train, y_train, x_train, alpha)
            eval_pred, _ = ridge_fit_predict(x_train, y_train, x_eval, alpha)
            pred_by_sample = {
                sample_index: train_pred[pos]
                for pos, sample_index in enumerate(train_samples)
            }
            pred_by_sample.update({sample_index: eval_pred[pos] for pos, sample_index in enumerate(eval_samples)})

            for penalty in penalties:
                score_by_key: dict[tuple[str, str], float] = {}
                train_scores: list[float] = []
                for row in candidate_feature_rows:
                    case = s(row, "candidate_case")
                    if case == reference_case or case not in candidate_cases:
                        continue
                    sample_index = s(row, "sample_index")
                    pred = pred_by_sample.get(sample_index)
                    if pred is None:
                        continue
                    score = (
                        float(pred[0]) * f(row, "feature__candidate_delta_heading_norm", 0.0)
                        + float(pred[1]) * f(row, "feature__candidate_delta_range_norm", 0.0)
                        - penalty * f(row, "feature__candidate_movement_norm", 0.0)
                    )
                    score_by_key[(sample_index, case)] = score
                    if sample_index in set(train_samples):
                        train_scores.append(score)
                for q in quantiles:
                    threshold = safe_quantile(train_scores, q)
                    if not math.isfinite(threshold):
                        continue
                    selected_train = choose_policy_from_scores(
                        candidate_rows_by_sample,
                        candidate_cases,
                        score_by_key,
                        threshold,
                        reference_case,
                    )
                    train_pred_rows = evaluate_selected_cases(
                        sample_rows_by_index,
                        selected_train,
                        train_samples,
                        range_span,
                        "tier1_ridge_direction_late4_oof",
                        fold,
                        "train_grid_internal",
                    )
                    train_error = safe_mean([f(row, "final_error") for row in train_pred_rows])
                    if train_error < best_train_error:
                        selected_eval = choose_policy_from_scores(
                            candidate_rows_by_sample,
                            candidate_cases,
                            score_by_key,
                            threshold,
                            reference_case,
                        )
                        best_train_error = train_error
                        best_payload = {
                            "alpha": alpha,
                            "penalty": penalty,
                            "threshold_quantile": q,
                            "threshold": threshold,
                            "train_internal_error": train_error,
                            "score_by_key": score_by_key,
                            "selected_eval": selected_eval,
                            "coef_shape": [len(coef), len(coef[0]) if coef else 0],
                            "standardize_mean_first5": [float(v) for v in mean_vec[:5]],
                            "standardize_scale_first5": [float(v) for v in scale_vec[:5]],
                        }

        if best_payload is None:
            raise RuntimeError(f"No Tier1 policy selected for fold {fold}")
        fit_desc = (
            f"alpha={best_payload['alpha']};penalty={best_payload['penalty']};"
            f"q={best_payload['threshold_quantile']};threshold={best_payload['threshold']}"
        )
        fold_pred_rows = evaluate_selected_cases(
            sample_rows_by_index,
            best_payload["selected_eval"],
            eval_samples,
            range_span,
            "tier1_ridge_direction_late4_oof",
            fold,
            fit_desc,
        )
        predictions.extend(fold_pred_rows)
        fold_records.append(
            {
                "fold_id": fold,
                "alpha": best_payload["alpha"],
                "penalty": best_payload["penalty"],
                "threshold_quantile": best_payload["threshold_quantile"],
                "threshold": best_payload["threshold"],
                "train_internal_error": best_payload["train_internal_error"],
                "eval_rows": len(fold_pred_rows),
                "eval_mean_final_error": safe_mean([f(row, "final_error") for row in fold_pred_rows]),
                "eval_selection_rate": safe_mean([f(row, "selected_non_final", 0.0) for row in fold_pred_rows]),
            }
        )
    return predictions, fold_records


def add_headroom_flags(sample_rows: list[dict[str, str]]) -> None:
    positives = [f(row, "baseline_minus_axiswise_oracle") for row in sample_rows if f(row, "baseline_minus_axiswise_oracle") > 0.0]
    q75 = safe_quantile(positives, 0.75)
    for row in sample_rows:
        row["high_axiswise_headroom_q75"] = str(int(math.isfinite(q75) and f(row, "baseline_minus_axiswise_oracle") >= q75))


def build_bucket_summary(pred_rows: list[dict[str, Any]], sample_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_sample = {s(row, "sample_index"): row for row in sample_rows}
    group_keys = [
        "best_final_case",
        "best_heading_case",
        "best_range_case",
        "heading_range_best_case_mismatch",
        "pred_heading_bin_idx",
        "pred_range_abs_bucket",
        "pred_range_sign",
        "high_axiswise_headroom_q75",
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pred in pred_rows:
        sample = by_sample.get(s(pred, "sample_index"), {})
        for key in group_keys:
            grouped[(s(pred, "policy_name"), key, s(sample, key))].append(pred)
    out: list[dict[str, Any]] = []
    for (policy, key, value), rows in sorted(grouped.items()):
        selected = [f(row, "selected_non_final", 0.0) for row in rows]
        improvements = [f(row, "improvement_vs_baseline") for row in rows]
        out.append(
            {
                "policy_name": policy,
                "group_key": key,
                "group_value": value,
                "count": len(rows),
                "mean_final_error": safe_mean([f(row, "final_error") for row in rows]),
                "mean_baseline_final_error": safe_mean([f(row, "baseline_final_error") for row in rows]),
                "mean_improvement_vs_baseline": safe_mean(improvements),
                "improve_rate": safe_mean([1.0 if value > 0.0 else 0.0 for value in improvements]),
                "selection_rate": safe_mean(selected),
                "fallback_rate": 1.0 - safe_mean(selected) if selected else math.nan,
                "mean_heading_rel_error": safe_mean([f(row, "heading_rel_error") for row in rows]),
                "mean_range_rel_error": safe_mean([f(row, "range_rel_error") for row in rows]),
            }
        )
    return out


def write_markdown(path: Path, summary: dict[str, Any], policy_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase103-E2 Correction-Direction Estimator",
        "",
        "Train-only OOF tabular estimator over Phase103-E0 trajectory artifacts.",
        "",
        "## Run",
        "",
        f"- e0_dir: `{summary['e0_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- rows: `{summary['rows']}`",
        f"- candidate_cases: `{', '.join(summary['candidate_cases'])}`",
        f"- feature_count: `{summary['feature_count']}`",
        f"- forbidden_input_violations: `{summary['forbidden_input_violations']}`",
        "",
        "## Policies",
        "",
        "| rank | policy | deployable | final error | delta vs H8 | heading | range | selected |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = next((row for row in policy_rows if row["policy_name"] == "baseline_final"), None)
    baseline_error = float(baseline["mean_final_error"]) if baseline else math.nan
    for rank, row in enumerate(policy_rows, 1):
        delta = float(row["mean_final_error"]) - baseline_error
        lines.append(
            f"| {rank} | `{row['policy_name']}` | {int(row['deployable'])} | "
            f"{float(row['mean_final_error']):.10g} | {delta:.10g} | "
            f"{float(row['mean_heading_rel_error']):.10g} | {float(row['mean_range_rel_error']):.10g} | "
            f"{float(row['selection_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- best_deployable_policy: `{summary['best_deployable_policy']}`",
            f"- best_deployable_delta_vs_baseline: `{summary['best_deployable_delta_vs_baseline']}`",
            f"- promotion_gate_passed: `{summary['promotion_gate_passed']}`",
            f"- kill_condition_triggered: `{summary['kill_condition_triggered']}`",
            f"- recommended_next_step: `{summary['recommended_next_step']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    e0_dir = Path(args.e0_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_required_files(e0_dir)

    candidate_cases = parse_str_list(args.candidate_cases)
    if args.reference_case not in candidate_cases:
        raise ValueError(f"reference case {args.reference_case!r} missing from candidate cases")
    alphas = parse_float_list(args.ridge_alphas)
    penalties = parse_float_list(args.penalties)
    quantiles = parse_float_list(args.threshold_quantiles)

    sample_rows = read_csv(e0_dir / "phase103_e0_per_sample_features.csv")
    if args.max_rows and args.max_rows > 0:
        sample_rows = sample_rows[: args.max_rows]
    target_rows_all = read_csv(e0_dir / "phase103_e0_targets.csv")
    sample_set = {s(row, "sample_index") for row in sample_rows}
    target_rows = [row for row in target_rows_all if s(row, "sample_index") in sample_set and s(row, "candidate_case") in candidate_cases]
    target_index = index_targets(target_rows)
    add_headroom_flags(sample_rows)

    missing_cases = [
        case
        for case in candidate_cases
        for row in sample_rows[:1]
        if f"pred_heading_{case}" not in row or f"pred_range_{case}" not in row
    ]
    if missing_cases:
        raise KeyError(f"Missing prediction columns for cases: {missing_cases}")

    range_buckets = sorted({s(row, "pred_range_abs_bucket") for row in sample_rows if s(row, "pred_range_abs_bucket")})
    range_signs = sorted({s(row, "pred_range_sign") for row in sample_rows if s(row, "pred_range_sign")})
    feature_rows: list[dict[str, Any]] = []
    source_by_feature: dict[str, str] = {}
    for sample in sample_rows:
        sample_index = s(sample, "sample_index")
        for case in candidate_cases:
            target = target_index.get((sample_index, case))
            if target is None:
                raise KeyError(f"Missing target row for sample={sample_index}, case={case}")
            features, sources = build_sample_feature_record(
                sample,
                case,
                target,
                range_buckets,
                range_signs,
                args.range_span,
                args.reference_case,
            )
            source_by_feature.update(sources)
            out = {
                **features,
                "sample_index": sample_index,
                "group_id": s(sample, "group_id"),
                "json_id": s(sample, "json_id"),
                "fold_id": s(sample, "fold_id"),
                "candidate_case": case,
                "true_correction_heading_norm": f(target, "true_correction_heading_norm"),
                "true_correction_range_norm": f(target, "true_correction_range_norm"),
                "benefit_vs_baseline": f(target, "benefit_vs_baseline"),
                "positive_benefit": int(f(target, "positive_benefit", 0.0)),
                "heading_benefit_vs_baseline": f(target, "heading_benefit_vs_baseline"),
                "range_benefit_vs_baseline": f(target, "range_benefit_vs_baseline"),
                "candidate_correction_cosine_2d": f(target, "candidate_correction_cosine_2d"),
            }
            feature_rows.append(out)

    feature_names = sorted([key for key in feature_rows[0] if key.startswith("feature__")])
    manifest_rows = [
        {
            "feature_name": name,
            "feature_role": "input",
            "source_column": source_by_feature.get(name, "derived_prediction_only"),
            "deployable": 1,
            "notes": "prediction_only_or_candidate_delta",
        }
        for name in feature_names
    ]
    for target_name in [
        "true_correction_heading_norm",
        "true_correction_range_norm",
        "benefit_vs_baseline",
        "positive_benefit",
        "heading_benefit_vs_baseline",
        "range_benefit_vs_baseline",
        "candidate_correction_cosine_2d",
    ]:
        manifest_rows.append(
            {
                "feature_name": target_name,
                "feature_role": "target_or_diagnostic",
                "source_column": f"phase103_e0_targets.{target_name}",
                "deployable": 0,
                "notes": "not_used_as_input",
            }
        )
    violations = validate_feature_manifest(manifest_rows)
    write_csv(output_dir / "phase103_e2_feature_manifest.csv", manifest_rows)
    if violations:
        (output_dir / "phase103_e2_leakage_violations.json").write_text(
            json.dumps({"violations": violations}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"Forbidden input feature violations: {violations}")

    folds = sorted({s(row, "fold_id") for row in sample_rows})
    oof_rows = run_tier0(sample_rows, candidate_cases, folds, args.range_span, args.reference_case)
    tier1_rows, fold_records = run_tier1(
        sample_rows,
        feature_rows,
        feature_names,
        candidate_cases,
        folds,
        alphas,
        penalties,
        quantiles,
        args.range_span,
        args.reference_case,
    )
    oof_rows.extend(tier1_rows)

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        by_policy[s(row, "policy_name")].append(row)
    policy_summary = [summarize_policy(rows) for _policy, rows in sorted(by_policy.items())]
    policy_summary.sort(key=lambda row: float(row["mean_final_error"]))
    baseline = next(row for row in policy_summary if row["policy_name"] == "baseline_final")
    deployable_rows = [row for row in policy_summary if int(row["deployable"]) == 1]
    best_deployable = min(deployable_rows, key=lambda row: float(row["mean_final_error"]))
    baseline_error = float(baseline["mean_final_error"])
    best_delta = float(best_deployable["mean_final_error"]) - baseline_error
    best_selection = float(best_deployable["selection_rate"])
    selection_pathological = best_selection <= 0.01 or best_selection >= 0.99
    kill_condition_triggered = best_delta >= 0.0 or selection_pathological
    promotion_gate_passed = best_delta <= -1e-6 and not kill_condition_triggered

    bucket_rows = build_bucket_summary(oof_rows, sample_rows)
    model_manifest = {
        "phase": "phase103_e2_correction_direction_estimator",
        "policy_name": "tier1_ridge_direction_late4_oof",
        "folds": fold_records,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "candidate_cases": candidate_cases,
        "reference_case": args.reference_case,
        "ridge_alphas": alphas,
        "penalties": penalties,
        "threshold_quantiles": quantiles,
        "uses_hidden_test_labels": False,
        "uses_val811_for_fitting": False,
        "uses_label_direction_as_input": False,
        "forbidden_input_violations": violations,
    }
    summary = {
        "phase": "phase103_e2_correction_direction_estimator",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "e0_dir": str(e0_dir),
        "output_dir": str(output_dir),
        "rows": len(sample_rows),
        "candidate_feature_rows": len(feature_rows),
        "candidate_cases": candidate_cases,
        "fold_counts": dict(sorted(Counter(s(row, "fold_id") for row in sample_rows).items())),
        "feature_count": len(feature_names),
        "forbidden_input_violations": len(violations),
        "policy_count": len(policy_summary),
        "oof_prediction_rows": len(oof_rows),
        "bucket_summary_rows": len(bucket_rows),
        "baseline_final_error": baseline_error,
        "best_deployable_policy": best_deployable["policy_name"],
        "best_deployable_final_error": float(best_deployable["mean_final_error"]),
        "best_deployable_delta_vs_baseline": best_delta,
        "best_deployable_selection_rate": best_selection,
        "best_deployable_selection_pathological": selection_pathological,
        "promotion_gate_passed": promotion_gate_passed,
        "kill_condition_triggered": kill_condition_triggered,
        "recommended_next_step": (
            "freeze_policy_and_write_val811_audit_plan"
            if promotion_gate_passed
            else "stop_tier1_or_consider_tier2_only_if_bucket_signal_is_positive"
            if kill_condition_triggered
            else "inspect_bucket_signal_before_tier2_or_tier3"
        ),
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "uses_val811_for_fitting": False,
        "elapsed_sec": round(time.time() - started, 3),
    }

    write_csv(output_dir / "phase103_e2_oof_predictions.csv", oof_rows)
    write_csv(output_dir / "phase103_e2_policy_summary.csv", policy_summary)
    write_csv(output_dir / "phase103_e2_bucket_summary.csv", bucket_rows)
    (output_dir / "phase103_e2_model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "phase103_e2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_e2_summary.md", summary, policy_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
