#!/usr/bin/env python3
"""Phase103-E3 fixed-output ensemble panel.

This is an engineering-only panel over frozen prediction files. It fits only
global fixed weights or fixed per-axis weights on train-hash folds; it never
uses per-sample routing, selectors, latent features, hidden-test labels, or
leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-e0-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-e0-dir", default="")
    parser.add_argument("--range-span", type=float, default=264.0)
    parser.add_argument("--late3-denom", type=int, default=50)
    parser.add_argument("--late4-denom", type=int, default=50)
    parser.add_argument("--late5-denom", type=int, default=20)
    parser.add_argument("--weight-chunk", type=int, default=2048)
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


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def wrap_angle_np(values):
    import numpy as np

    return ((values + 180.0) % 360.0) - 180.0


def simplex_weights(n: int, denom: int):
    import numpy as np

    rows: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(tuple([*prefix, remaining]))
            return
        for value in range(remaining + 1):
            rec([*prefix, value], remaining - value, slots - 1)

    rec([], denom, n)
    return np.asarray(rows, dtype="float64") / float(denom)


def one_hot_weights(n: int):
    import numpy as np

    return np.eye(n, dtype="float64")


def load_e0_arrays(e0_dir: Path):
    import numpy as np

    rows = read_csv(e0_dir / "phase103_e0_per_sample_features.csv")
    manifest = read_csv(e0_dir / "phase103_e0_case_manifest.csv")
    case_ids = [row["case_id"] for row in manifest]
    headings = np.asarray([[f(row, f"pred_heading_{case}") for case in case_ids] for row in rows], dtype="float64")
    ranges = np.asarray([[f(row, f"pred_range_{case}") for case in case_ids] for row in rows], dtype="float64")
    true_heading = np.asarray([f(row, "true_heading_deg") for row in rows], dtype="float64")
    true_range = np.asarray([f(row, "true_range") for row in rows], dtype="float64")
    folds = np.asarray([str(row.get("fold_id", idx % 5)) for idx, row in enumerate(rows)])
    sample_ids = [row.get("sample_index", str(idx)) for idx, row in enumerate(rows)]
    return {
        "rows": rows,
        "case_ids": case_ids,
        "headings": headings,
        "ranges": ranges,
        "true_heading": true_heading,
        "true_range": true_range,
        "folds": folds,
        "sample_ids": sample_ids,
    }


def case_indices(case_ids: list[str], wanted: list[str]) -> list[int]:
    return [case_ids.index(case) for case in wanted if case in case_ids]


def available_groups(case_ids: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    specs = [
        ("late3", ["final", "step400000", "step450000"], "late3-denom"),
        ("late4", ["final", "step350000", "step400000", "step450000"], "late4-denom"),
        ("late5", ["final", "step300000", "step350000", "step400000", "step450000"], "late5-denom"),
    ]
    for name, wanted, denom_key in specs:
        idx = case_indices(case_ids, wanted)
        if len(idx) >= 2:
            groups.append({"group_name": name, "cases": [case_ids[item] for item in idx], "indices": idx, "denom_key": denom_key})
    return groups


def score_matrix(data: dict[str, Any], case_idx: list[int], heading_weights, range_weights, indices, range_span: float, chunk: int = 2048):
    import numpy as np

    h = data["headings"][np.asarray(indices)[:, None], np.asarray(case_idx)[None, :]]
    r = data["ranges"][np.asarray(indices)[:, None], np.asarray(case_idx)[None, :]]
    true_h = data["true_heading"][np.asarray(indices)]
    true_r = data["true_range"][np.asarray(indices)]
    sin_h = np.sin(np.deg2rad(h))
    cos_h = np.cos(np.deg2rad(h))
    scores: list[np.ndarray] = []
    heading_scores: list[np.ndarray] = []
    range_scores: list[np.ndarray] = []
    for start in range(0, len(heading_weights), chunk):
        hw = heading_weights[start : start + chunk]
        rw = range_weights[start : start + chunk]
        pred_sin = sin_h @ hw.T
        pred_cos = cos_h @ hw.T
        pred_h = np.rad2deg(np.arctan2(pred_sin, pred_cos))
        pred_r = r @ rw.T
        h_err = np.abs(wrap_angle_np(pred_h - true_h[:, None])) / 180.0
        r_err = np.abs(pred_r - true_r[:, None]) / max(range_span, 1e-12)
        heading_scores.append(h_err.mean(axis=0))
        range_scores.append(r_err.mean(axis=0))
        scores.append((0.5 * (h_err + r_err)).mean(axis=0))
    return np.concatenate(scores), np.concatenate(heading_scores), np.concatenate(range_scores)


def predict_rows(data: dict[str, Any], case_idx: list[int], heading_weight, range_weight, indices, range_span: float, policy: str, fold: str, fit_desc: str) -> list[dict[str, Any]]:
    import numpy as np

    idx = np.asarray(indices)
    ci = np.asarray(case_idx)
    h = data["headings"][idx[:, None], ci[None, :]]
    r = data["ranges"][idx[:, None], ci[None, :]]
    sin_h = np.sin(np.deg2rad(h))
    cos_h = np.cos(np.deg2rad(h))
    hw = np.asarray(heading_weight, dtype="float64")
    rw = np.asarray(range_weight, dtype="float64")
    pred_h = np.rad2deg(np.arctan2(sin_h @ hw, cos_h @ hw))
    pred_r = r @ rw
    h_err = np.abs(wrap_angle_np(pred_h - data["true_heading"][idx])) / 180.0
    r_err = np.abs(pred_r - data["true_range"][idx]) / max(range_span, 1e-12)
    final_error = 0.5 * (h_err + r_err)
    out: list[dict[str, Any]] = []
    for local_pos, sample_idx in enumerate(idx.tolist()):
        row = data["rows"][sample_idx]
        baseline = f(row, "baseline_final_error", math.nan)
        out.append(
            {
                "policy_name": policy,
                "fold_id": fold,
                "fit_desc": fit_desc,
                "sample_index": row.get("sample_index", str(sample_idx)),
                "group_id": row.get("group_id", ""),
                "json_id": row.get("json_id", ""),
                "pred_heading_deg": float(pred_h[local_pos]),
                "pred_range": float(pred_r[local_pos]),
                "heading_rel_error": float(h_err[local_pos]),
                "range_rel_error": float(r_err[local_pos]),
                "final_error": float(final_error[local_pos]),
                "baseline_final_error": baseline,
                "improvement_vs_baseline": baseline - float(final_error[local_pos]) if math.isfinite(baseline) else math.nan,
            }
        )
    return out


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [f(row, "improvement_vs_baseline") for row in rows]
    return {
        "policy_name": rows[0]["policy_name"] if rows else "",
        "count": len(rows),
        "mean_final_error": safe_mean([f(row, "final_error") for row in rows]),
        "mean_heading_rel_error": safe_mean([f(row, "heading_rel_error") for row in rows]),
        "mean_range_rel_error": safe_mean([f(row, "range_rel_error") for row in rows]),
        "mean_baseline_final_error": safe_mean([f(row, "baseline_final_error") for row in rows]),
        "mean_improvement_vs_baseline": safe_mean(improvements),
        "improve_rate": safe_mean([1.0 if value > 0.0 else 0.0 for value in improvements]),
    }


def choose_weight(data, train_idx, case_idx, weights, range_span: float, objective: str, chunk: int):
    scores, heading_scores, range_scores = score_matrix(data, case_idx, weights, weights, train_idx, range_span, chunk)
    if objective == "heading":
        selected = int(heading_scores.argmin())
    elif objective == "range":
        selected = int(range_scores.argmin())
    else:
        selected = int(scores.argmin())
    return weights[selected]


def choose_per_axis_weights(data, train_idx, case_idx, weights, range_span: float, chunk: int):
    _scores, heading_scores, range_scores = score_matrix(data, case_idx, weights, weights, train_idx, range_span, chunk)
    return weights[int(heading_scores.argmin())], weights[int(range_scores.argmin())]


def run_oof(data, groups: list[dict[str, Any]], denoms: dict[str, int], range_span: float, chunk: int):
    import numpy as np

    folds = sorted(set(data["folds"].tolist()))
    all_indices = list(range(len(data["rows"])))
    all_oof: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    weight_cache: dict[tuple[int, int], Any] = {}

    final_idx = case_indices(data["case_ids"], ["final"])
    for fold in folds:
        train_idx = [idx for idx in all_indices if data["folds"][idx] != fold]
        eval_idx = [idx for idx in all_indices if data["folds"][idx] == fold]
        all_oof.extend(predict_rows(data, final_idx, [1.0], [1.0], eval_idx, range_span, "baseline_final", fold, "fixed_final"))

        for group in groups:
            case_idx = group["indices"]
            cases = group["cases"]
            n = len(cases)
            uniform = np.full(n, 1.0 / n, dtype="float64")
            all_oof.extend(predict_rows(data, case_idx, uniform, uniform, eval_idx, range_span, f"uniform_{group['group_name']}", fold, "uniform_fixed"))
            weight_rows.append(
                {
                    "policy_name": f"uniform_{group['group_name']}",
                    "fold_id": fold,
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(uniform.tolist()),
                    "range_weights": json.dumps(uniform.tolist()),
                    "fit_scope": "none_uniform",
                }
            )

            eye = one_hot_weights(n)
            best_whole = choose_weight(data, train_idx, case_idx, eye, range_span, "final", chunk)
            best_h, best_r = choose_per_axis_weights(data, train_idx, case_idx, eye, range_span, chunk)
            all_oof.extend(
                predict_rows(data, case_idx, best_whole, best_whole, eval_idx, range_span, f"fixed_best_{group['group_name']}_oof", fold, "train_fixed_best_final")
            )
            all_oof.extend(
                predict_rows(data, case_idx, best_h, best_r, eval_idx, range_span, f"fixed_per_axis_best_{group['group_name']}_oof", fold, "train_fixed_best_axis")
            )
            weight_rows.extend(
                [
                    {
                        "policy_name": f"fixed_best_{group['group_name']}_oof",
                        "fold_id": fold,
                        "cases": ",".join(cases),
                        "heading_weights": json.dumps(best_whole.tolist()),
                        "range_weights": json.dumps(best_whole.tolist()),
                        "fit_scope": "train_hash_oof_fold_fixed",
                    },
                    {
                        "policy_name": f"fixed_per_axis_best_{group['group_name']}_oof",
                        "fold_id": fold,
                        "cases": ",".join(cases),
                        "heading_weights": json.dumps(best_h.tolist()),
                        "range_weights": json.dumps(best_r.tolist()),
                        "fit_scope": "train_hash_oof_fold_fixed_per_axis",
                    },
                ]
            )

            denom = denoms[group["denom_key"]]
            cache_key = (n, denom)
            if cache_key not in weight_cache:
                weight_cache[cache_key] = simplex_weights(n, denom)
            weights = weight_cache[cache_key]
            global_w = choose_weight(data, train_idx, case_idx, weights, range_span, "final", chunk)
            axis_h, axis_r = choose_per_axis_weights(data, train_idx, case_idx, weights, range_span, chunk)
            all_oof.extend(
                predict_rows(
                    data,
                    case_idx,
                    global_w,
                    global_w,
                    eval_idx,
                    range_span,
                    f"fixed_global_blend_{group['group_name']}_d{denom}_oof",
                    fold,
                    "train_hash_fold_min_final_error",
                )
            )
            all_oof.extend(
                predict_rows(
                    data,
                    case_idx,
                    axis_h,
                    axis_r,
                    eval_idx,
                    range_span,
                    f"fixed_per_axis_blend_{group['group_name']}_d{denom}_oof",
                    fold,
                    "train_hash_fold_min_axis_errors",
                )
            )
            weight_rows.extend(
                [
                    {
                        "policy_name": f"fixed_global_blend_{group['group_name']}_d{denom}_oof",
                        "fold_id": fold,
                        "cases": ",".join(cases),
                        "heading_weights": json.dumps(global_w.tolist()),
                        "range_weights": json.dumps(global_w.tolist()),
                        "fit_scope": "train_hash_oof_fold_global_fixed",
                    },
                    {
                        "policy_name": f"fixed_per_axis_blend_{group['group_name']}_d{denom}_oof",
                        "fold_id": fold,
                        "cases": ",".join(cases),
                        "heading_weights": json.dumps(axis_h.tolist()),
                        "range_weights": json.dumps(axis_r.tolist()),
                        "fit_scope": "train_hash_oof_fold_per_axis_fixed",
                    },
                ]
            )
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_oof:
        by_policy[str(row["policy_name"])].append(row)
    summary_rows = [summarize_policy(rows) for rows in by_policy.values()]
    summary_rows.sort(key=lambda row: float(row["mean_final_error"]))
    return summary_rows, all_oof, weight_rows


def frozen_weights(data, groups: list[dict[str, Any]], denoms: dict[str, int], range_span: float, chunk: int):
    import numpy as np

    all_idx = list(range(len(data["rows"])))
    rows: list[dict[str, Any]] = [
        {
            "policy_name": "baseline_final",
            "cases": "final",
            "heading_weights": "[1.0]",
            "range_weights": "[1.0]",
            "fit_scope": "fixed_anchor",
            "sample_level_routing": False,
        }
    ]
    weight_cache: dict[tuple[int, int], Any] = {}
    for group in groups:
        cases = group["cases"]
        case_idx = group["indices"]
        n = len(cases)
        uniform = np.full(n, 1.0 / n, dtype="float64")
        eye = one_hot_weights(n)
        best_whole = choose_weight(data, all_idx, case_idx, eye, range_span, "final", chunk)
        best_h, best_r = choose_per_axis_weights(data, all_idx, case_idx, eye, range_span, chunk)
        denom = denoms[group["denom_key"]]
        cache_key = (n, denom)
        if cache_key not in weight_cache:
            weight_cache[cache_key] = simplex_weights(n, denom)
        weights = weight_cache[cache_key]
        global_w = choose_weight(data, all_idx, case_idx, weights, range_span, "final", chunk)
        axis_h, axis_r = choose_per_axis_weights(data, all_idx, case_idx, weights, range_span, chunk)
        rows.extend(
            [
                {
                    "policy_name": f"uniform_{group['group_name']}",
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(uniform.tolist()),
                    "range_weights": json.dumps(uniform.tolist()),
                    "fit_scope": "none_uniform",
                    "sample_level_routing": False,
                },
                {
                    "policy_name": f"fixed_best_{group['group_name']}",
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(best_whole.tolist()),
                    "range_weights": json.dumps(best_whole.tolist()),
                    "fit_scope": "train_hash_all_fixed_best_final",
                    "sample_level_routing": False,
                },
                {
                    "policy_name": f"fixed_per_axis_best_{group['group_name']}",
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(best_h.tolist()),
                    "range_weights": json.dumps(best_r.tolist()),
                    "fit_scope": "train_hash_all_fixed_best_axis",
                    "sample_level_routing": False,
                },
                {
                    "policy_name": f"fixed_global_blend_{group['group_name']}_d{denom}",
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(global_w.tolist()),
                    "range_weights": json.dumps(global_w.tolist()),
                    "fit_scope": "train_hash_all_global_fixed",
                    "sample_level_routing": False,
                },
                {
                    "policy_name": f"fixed_per_axis_blend_{group['group_name']}_d{denom}",
                    "cases": ",".join(cases),
                    "heading_weights": json.dumps(axis_h.tolist()),
                    "range_weights": json.dumps(axis_r.tolist()),
                    "fit_scope": "train_hash_all_per_axis_fixed",
                    "sample_level_routing": False,
                },
            ]
        )
    return rows


def evaluate_frozen(data, frozen_rows: list[dict[str, Any]], range_span: float):
    out_rows: list[dict[str, Any]] = []
    case_ids = data["case_ids"]
    indices = list(range(len(data["rows"])))
    for row in frozen_rows:
        cases = [item for item in str(row["cases"]).split(",") if item]
        case_idx = case_indices(case_ids, cases)
        if len(case_idx) != len(cases):
            continue
        heading_weights = json.loads(str(row["heading_weights"]))
        range_weights = json.loads(str(row["range_weights"]))
        out_rows.extend(
            predict_rows(
                data,
                case_idx,
                heading_weights,
                range_weights,
                indices,
                range_span,
                str(row["policy_name"]),
                "frozen_all_train_hash",
                str(row["fit_scope"]),
            )
        )
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        by_policy[str(row["policy_name"])].append(row)
    summary_rows = [summarize_policy(rows) for rows in by_policy.values()]
    summary_rows.sort(key=lambda row: float(row["mean_final_error"]))
    return summary_rows, out_rows


def write_markdown(path: Path, summary: dict[str, Any], train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase103-E3 Fixed Ensemble Panel",
        "",
        "Hard constraint: fixed output ensemble only; no sample-level selector or routing.",
        "",
        "## Summary",
        "",
        f"- rows: `{summary['train_rows']}`",
        f"- val rows: `{summary['val_rows']}`",
        f"- best train OOF policy: `{summary['best_train_oof_policy']}`",
        f"- best train OOF delta: `{summary['best_train_oof_delta_vs_baseline']}`",
        f"- best val frozen policy: `{summary['best_val_frozen_policy']}`",
        f"- best val frozen delta: `{summary['best_val_frozen_delta_vs_baseline']}`",
        "",
        "## Train OOF",
        "",
        "| rank | policy | final error | delta vs H8 | heading | range | improve rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = next((row for row in train_rows if row["policy_name"] == "baseline_final"), None)
    baseline_error = float(baseline["mean_final_error"]) if baseline else math.nan
    for rank, row in enumerate(train_rows[:20], 1):
        delta = float(row["mean_final_error"]) - baseline_error
        lines.append(
            f"| {rank} | `{row['policy_name']}` | {float(row['mean_final_error']):.10g} | {delta:.10g} | "
            f"{float(row['mean_heading_rel_error']):.10g} | {float(row['mean_range_rel_error']):.10g} | {float(row['improve_rate']):.4f} |"
        )
    if val_rows:
        lines.extend(
            [
                "",
                "## Frozen Val Audit",
                "",
                "| rank | policy | final error | delta vs H8 | heading | range | improve rate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        val_baseline = next((row for row in val_rows if row["policy_name"] == "baseline_final"), None)
        val_baseline_error = float(val_baseline["mean_final_error"]) if val_baseline else math.nan
        for rank, row in enumerate(val_rows[:20], 1):
            delta = float(row["mean_final_error"]) - val_baseline_error
            lines.append(
                f"| {rank} | `{row['policy_name']}` | {float(row['mean_final_error']):.10g} | {delta:.10g} | "
                f"{float(row['mean_heading_rel_error']):.10g} | {float(row['mean_range_rel_error']):.10g} | {float(row['improve_rate']):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_e0_arrays(Path(args.train_e0_dir))
    groups = available_groups(train["case_ids"])
    denoms = {"late3-denom": args.late3_denom, "late4-denom": args.late4_denom, "late5-denom": args.late5_denom}
    train_summary, oof_rows, oof_weight_rows = run_oof(train, groups, denoms, args.range_span, args.weight_chunk)
    frozen_rows = frozen_weights(train, groups, denoms, args.range_span, args.weight_chunk)
    val_summary: list[dict[str, Any]] = []
    val_pred_rows: list[dict[str, Any]] = []
    if args.val_e0_dir:
        val = load_e0_arrays(Path(args.val_e0_dir))
        val_summary, val_pred_rows = evaluate_frozen(val, frozen_rows, args.range_span)

    baseline = next((row for row in train_summary if row["policy_name"] == "baseline_final"), None)
    best = train_summary[0] if train_summary else {}
    val_baseline = next((row for row in val_summary if row["policy_name"] == "baseline_final"), None)
    val_best = val_summary[0] if val_summary else {}
    summary = {
        "phase": "phase103_e3_fixed_ensemble_panel",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_e0_dir": str(Path(args.train_e0_dir)),
        "val_e0_dir": str(Path(args.val_e0_dir)) if args.val_e0_dir else "",
        "output_dir": str(output_dir),
        "train_rows": len(train["rows"]),
        "val_rows": len(val_pred_rows) // max(len(frozen_rows), 1) if val_pred_rows else 0,
        "case_ids": train["case_ids"],
        "groups": [{"group_name": row["group_name"], "cases": row["cases"], "denom": denoms[row["denom_key"]]} for row in groups],
        "fold_counts": dict(sorted(Counter(train["folds"].tolist()).items())),
        "policy_count": len(train_summary),
        "frozen_policy_count": len(frozen_rows),
        "baseline_train_oof_error": float(baseline["mean_final_error"]) if baseline else math.nan,
        "best_train_oof_policy": best.get("policy_name", ""),
        "best_train_oof_error": float(best.get("mean_final_error", math.nan)) if best else math.nan,
        "best_train_oof_delta_vs_baseline": (
            float(best.get("mean_final_error", math.nan)) - float(baseline["mean_final_error"]) if best and baseline else math.nan
        ),
        "baseline_val_frozen_error": float(val_baseline["mean_final_error"]) if val_baseline else math.nan,
        "best_val_frozen_policy": val_best.get("policy_name", ""),
        "best_val_frozen_error": float(val_best.get("mean_final_error", math.nan)) if val_best else math.nan,
        "best_val_frozen_delta_vs_baseline": (
            float(val_best.get("mean_final_error", math.nan)) - float(val_baseline["mean_final_error"]) if val_best and val_baseline else math.nan
        ),
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "trains_network": False,
        "fits_sample_level_selector": False,
        "uses_sample_level_routing": False,
        "allowed_weight_scope": "global_fixed_or_per_axis_fixed",
        "elapsed_sec": round(time.time() - started, 3),
    }
    constraint_manifest = {
        "hard_constraint": "direct_fixed_ensemble_only",
        "forbidden": [
            "sample_level_selector",
            "sample_level_routing",
            "latent_feature_policy",
            "val811_weight_tuning",
            "hidden_test_labels",
            "leaderboard_feedback",
            "network_training",
        ],
        "allowed": [
            "uniform_fixed_weights",
            "train_hash_oof_global_fixed_weights",
            "train_hash_oof_per_axis_fixed_weights",
            "frozen_train_hash_all_weights_for_val_audit",
        ],
    }
    write_csv(output_dir / "phase103_e3_train_oof_policy_summary.csv", train_summary)
    write_csv(output_dir / "phase103_e3_train_oof_predictions.csv", oof_rows)
    write_csv(output_dir / "phase103_e3_oof_fold_weights.csv", oof_weight_rows)
    write_csv(output_dir / "phase103_e3_frozen_weights.csv", frozen_rows)
    write_csv(output_dir / "phase103_e3_val_frozen_policy_summary.csv", val_summary)
    write_csv(output_dir / "phase103_e3_val_frozen_predictions.csv", val_pred_rows)
    (output_dir / "phase103_e3_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "phase103_e3_constraint_manifest.json").write_text(
        json.dumps(constraint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_e3_summary.md", summary, train_summary, val_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
