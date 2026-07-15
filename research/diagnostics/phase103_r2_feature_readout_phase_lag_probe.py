#!/usr/bin/env python3
"""Phase103-R2 feature/readout phase-lag mechanism probe.

This lab-side analysis consumes existing Phase102 R5/R6/R7 tabular artifacts.
It tests whether checkpoint-specific advantages look like feature/readout
co-adaptation rather than a pure feature-only, head-only, or simple global
feature-alignment effect.

This is not a hidden-test or leaderboard-feedback analysis. It uses val811
local labels already present in prior mechanism artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_ROOT = Path("/media/jgzn/SSD_lexar/RZ/UAVM")
DEFAULT_R5_ATTRIBUTION = DEFAULT_ROOT / (
    "runs/phase102_pasc_mechanism_v1/"
    "r5_repr_readout_attribution_phase89_h8_val811_20260622/"
    "phase102_r5_attribution_per_sample.csv"
)
DEFAULT_R6_PREDICTION = DEFAULT_ROOT / (
    "runs/phase102_pasc_mechanism_v1/"
    "r6_feature_head_coadapt_phase89_h8_val811_20260622/"
    "phase102_r6_prediction_per_sample.csv"
)
DEFAULT_R6_RECOVERY = DEFAULT_ROOT / (
    "runs/phase102_pasc_mechanism_v1/"
    "r6_feature_head_coadapt_phase89_h8_val811_20260622/"
    "phase102_r6_recovery_summary.csv"
)
DEFAULT_R7_RECOVERY = DEFAULT_ROOT / (
    "runs/phase102_pasc_mechanism_v1/"
    "r7_head_submodule_swap_phase89_h8_val811_20260622/"
    "phase102_r7_recovery_summary.csv"
)
DEFAULT_PHASE95_SUBSPACE = DEFAULT_ROOT / (
    "runs/phase95_pose_representation_geometry_v1/"
    "r2_h8_final_val811_subspace/"
    "phase95_r2_subspace_summary_rows.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / (
    "runs/phase103_trajectory_dynamics_v1/"
    "r2_feature_readout_phase_lag_val811_20260622"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-attribution", default=str(DEFAULT_R5_ATTRIBUTION))
    parser.add_argument("--r6-prediction", default=str(DEFAULT_R6_PREDICTION))
    parser.add_argument("--r6-recovery", default=str(DEFAULT_R6_RECOVERY))
    parser.add_argument("--r7-recovery", default=str(DEFAULT_R7_RECOVERY))
    parser.add_argument("--phase95-subspace-summary", default=str(DEFAULT_PHASE95_SUBSPACE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-group-count", type=int, default=10)
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


def s(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = f(row, key, math.nan)
    if math.isfinite(value):
        return int(value)
    text = s(row, key, "")
    if text.lower() in {"true", "yes"}:
        return 1
    if text.lower() in {"false", "no"}:
        return 0
    return default


def clean(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def safe_mean(values: list[float]) -> float:
    finite = clean(values)
    return sum(finite) / len(finite) if finite else math.nan


def safe_median(values: list[float]) -> float:
    finite = clean(values)
    return float(statistics.median(finite)) if finite else math.nan


def safe_min(values: list[float]) -> float:
    finite = clean(values)
    return min(finite) if finite else math.nan


def safe_max(values: list[float]) -> float:
    finite = clean(values)
    return max(finite) if finite else math.nan


def safe_frac(flags: list[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags) if flags else math.nan


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def enrich_r5_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        normal = f(row, "normal_improvement")
        feature = f(row, "feature_only_improvement")
        head = f(row, "head_only_improvement")
        best_single = max(feature, head)
        case_name = s(row, "case_name")
        best_final = s(row, "best_final_case")
        best_heading = s(row, "best_heading_case")
        best_range = s(row, "best_range_case")
        enriched: dict[str, Any] = dict(row)
        enriched.update(
            {
                "target_is_best_final": int(case_name == best_final),
                "target_is_best_heading": int(case_name == best_heading),
                "target_is_best_range": int(case_name == best_range),
                "axis_mismatch": int(best_heading != best_range),
                "best_single_path_improvement": best_single,
                "coupling_gap_vs_best_single": normal - best_single,
                "additive_coupling_residual": normal - feature - head,
                "normal_gt_feature": int(normal > feature),
                "normal_gt_head": int(normal > head),
                "normal_gt_best_single": int(normal > best_single),
                "coupled_only": int(normal > 0.0 and feature <= 0.0 and head <= 0.0),
                "any_single_helps": int(feature > 0.0 or head > 0.0),
            }
        )
        out.append(enriched)
    return out


def summarize_r5_selected(
    rows: list[dict[str, Any]],
    group_key: str,
    group_value: str,
    total_rows: int,
) -> dict[str, Any]:
    normal = [f(row, "normal_improvement") for row in rows]
    feature = [f(row, "feature_only_improvement") for row in rows]
    head = [f(row, "head_only_improvement") for row in rows]
    best_single = [f(row, "best_single_path_improvement") for row in rows]
    coupling_gap = [f(row, "coupling_gap_vs_best_single") for row in rows]
    additive = [f(row, "additive_coupling_residual") for row in rows]
    positive_rows = [row for row in rows if f(row, "normal_improvement") > 0.0]
    return {
        "group_key": group_key,
        "group_value": group_value,
        "count": len(rows),
        "frac_of_rows": len(rows) / max(total_rows, 1),
        "normal_improves_rate": safe_frac([value > 0.0 for value in normal if math.isfinite(value)]),
        "feature_only_improves_rate": safe_frac([value > 0.0 for value in feature if math.isfinite(value)]),
        "head_only_improves_rate": safe_frac([value > 0.0 for value in head if math.isfinite(value)]),
        "normal_gt_feature_rate": safe_frac([i(row, "normal_gt_feature") == 1 for row in rows]),
        "normal_gt_head_rate": safe_frac([i(row, "normal_gt_head") == 1 for row in rows]),
        "normal_gt_best_single_rate": safe_frac([i(row, "normal_gt_best_single") == 1 for row in rows]),
        "coupled_only_frac": safe_frac([i(row, "coupled_only") == 1 for row in rows]),
        "mean_normal_improvement": safe_mean(normal),
        "mean_feature_only_improvement": safe_mean(feature),
        "mean_head_only_improvement": safe_mean(head),
        "mean_best_single_path_improvement": safe_mean(best_single),
        "mean_coupling_gap_vs_best_single": safe_mean(coupling_gap),
        "median_coupling_gap_vs_best_single": safe_median(coupling_gap),
        "mean_additive_coupling_residual": safe_mean(additive),
        "positive_normal_count": len(positive_rows),
        "positive_normal_frac": len(positive_rows) / max(len(rows), 1),
        "mean_positive_normal_improvement": safe_mean([f(row, "normal_improvement") for row in positive_rows]),
        "mean_positive_coupling_gap": safe_mean([f(row, "coupling_gap_vs_best_single") for row in positive_rows]),
        "mean_baseline_minus_axiswise_oracle": safe_mean(
            [f(row, "baseline_minus_axiswise_oracle") for row in rows]
        ),
        "mean_baseline_minus_best_final_error": safe_mean(
            [f(row, "baseline_minus_best_final_error") for row in rows]
        ),
    }


def summarize_r5_coupling(rows: list[dict[str, Any]], min_group_count: int) -> list[dict[str, Any]]:
    groupers: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("ALL", lambda _row: "ALL"),
        ("case_name", lambda row: s(row, "case_name")),
        ("target_is_best_final", lambda row: str(i(row, "target_is_best_final"))),
        ("target_is_best_heading", lambda row: str(i(row, "target_is_best_heading"))),
        ("target_is_best_range", lambda row: str(i(row, "target_is_best_range"))),
        ("axis_mismatch", lambda row: str(i(row, "axis_mismatch"))),
        ("best_final_case", lambda row: s(row, "best_final_case")),
        ("best_heading_case", lambda row: s(row, "best_heading_case")),
        ("best_range_case", lambda row: s(row, "best_range_case")),
        ("case_name__target_is_best_final", lambda row: f"{s(row, 'case_name')}|{i(row, 'target_is_best_final')}"),
        ("case_name__target_is_best_heading", lambda row: f"{s(row, 'case_name')}|{i(row, 'target_is_best_heading')}"),
        ("case_name__target_is_best_range", lambda row: f"{s(row, 'case_name')}|{i(row, 'target_is_best_range')}"),
        ("case_name__axis_mismatch", lambda row: f"{s(row, 'case_name')}|{i(row, 'axis_mismatch')}"),
        (
            "phase100_axiswise_headroom_bucket",
            lambda row: s(row, "phase100_axiswise_headroom_bucket"),
        ),
        (
            "phase100_best_checkpoint_headroom_bucket",
            lambda row: s(row, "phase100_best_checkpoint_headroom_bucket"),
        ),
    ]
    out: list[dict[str, Any]] = []
    total_rows = len(rows)
    for group_key, group_fn in groupers:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[group_fn(row)].append(row)
        for group_value, selected in groups.items():
            if group_value == "" or len(selected) < min_group_count:
                continue
            out.append(summarize_r5_selected(selected, group_key, group_value, total_rows))
    return sorted(
        out,
        key=lambda row: (
            -float(row["mean_positive_coupling_gap"])
            if math.isfinite(float(row["mean_positive_coupling_gap"]))
            else 0.0,
            -float(row["positive_normal_frac"]),
            -int(row["count"]),
            str(row["group_key"]),
            str(row["group_value"]),
        ),
    )


def summarize_prediction_rows(
    rows: list[dict[str, str]],
    min_group_count: int,
) -> list[dict[str, Any]]:
    groupers: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("ALL", lambda _row: "ALL"),
        ("target_is_best_final", lambda row: str(int(s(row, "target_case") == s(row, "best_final_case")))),
        ("target_is_best_heading", lambda row: str(int(s(row, "target_case") == s(row, "best_heading_case")))),
        ("target_is_best_range", lambda row: str(int(s(row, "target_case") == s(row, "best_range_case")))),
        ("axis_mismatch", lambda row: str(int(s(row, "best_heading_case") != s(row, "best_range_case")))),
        ("phase100_axiswise_headroom_bucket", lambda row: s(row, "phase100_axiswise_headroom_bucket")),
        ("phase100_best_checkpoint_headroom_bucket", lambda row: s(row, "phase100_best_checkpoint_headroom_bucket")),
    ]
    out: list[dict[str, Any]] = []
    total_rows = len(rows)
    for group_key, group_fn in groupers:
        groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                s(row, "target_case"),
                s(row, "mode"),
                s(row, "variant"),
                s(row, "layer_spec"),
                group_fn(row),
            )
            groups[key].append(row)
        for (target_case, mode, variant, layer_spec, group_value), selected in groups.items():
            if len(selected) < min_group_count:
                continue
            improvements = [f(row, "improvement_vs_reference") for row in selected]
            out.append(
                {
                    "target_case": target_case,
                    "mode": mode,
                    "variant": variant,
                    "layer_spec": layer_spec,
                    "group_key": group_key,
                    "group_value": group_value,
                    "count": len(selected),
                    "frac_of_prediction_rows": len(selected) / max(total_rows, 1),
                    "mean_improvement_vs_reference": safe_mean(improvements),
                    "median_improvement_vs_reference": safe_median(improvements),
                    "improves_reference_rate": safe_frac(
                        [value > 0.0 for value in improvements if math.isfinite(value)]
                    ),
                    "mean_final_error": safe_mean([f(row, "final_error") for row in selected]),
                    "mean_heading_rel_error": safe_mean([f(row, "heading_rel_error") for row in selected]),
                    "mean_range_rel_error": safe_mean([f(row, "range_rel_error") for row in selected]),
                    "mean_heading_delta_to_reference": safe_mean(
                        [f(row, "heading_delta_to_reference") for row in selected]
                    ),
                    "mean_range_delta_to_reference": safe_mean(
                        [f(row, "range_delta_to_reference") for row in selected]
                    ),
                }
            )
    return sorted(
        out,
        key=lambda row: (
            str(row["target_case"]),
            str(row["group_key"]),
            str(row["group_value"]),
            -float(row["mean_improvement_vs_reference"])
            if math.isfinite(float(row["mean_improvement_vs_reference"]))
            else 0.0,
        ),
    )


def summarize_r6_recovery(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        normal = f(row, "normal_improvement")
        feature = f(row, "feature_only_improvement")
        adapted = f(row, "adapted_improvement")
        out.append(
            {
                "source": "r6_feature_alignment_recovery",
                "measurement_status": "feature_alignment_proxy_not_true_layer_similarity",
                "target_case": s(row, "target_case"),
                "variant": s(row, "variant"),
                "layer_spec": s(row, "layer_spec"),
                "normal_improvement": normal,
                "feature_only_improvement": feature,
                "adapted_improvement": adapted,
                "adapted_minus_feature_only": f(row, "adapted_minus_feature_only"),
                "normal_minus_feature_only": f(row, "normal_minus_feature_only"),
                "recovery_frac_of_normal_feature_gap": f(row, "recovery_frac_of_normal_feature_gap"),
                "adapted_minus_normal": adapted - normal,
                "adapted_mean_error": f(row, "adapted_mean_error"),
                "feature_only_mean_error": f(row, "feature_only_mean_error"),
                "normal_mean_error": f(row, "normal_mean_error"),
                "reference_mean_error": f(row, "reference_mean_error"),
            }
        )
    return out


def summarize_phase95_subspace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "source": "phase95_final_subspace_proxy",
                "measurement_status": "missing_optional_phase95_subspace_summary",
                "path": str(path),
            }
        ]
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        out.append(
            {
                "source": "phase95_final_subspace_proxy",
                "measurement_status": "final_checkpoint_subspace_proxy_not_cross_checkpoint_similarity",
                "control_kind": s(row, "control_kind"),
                "layer_id": s(row, "layer_id"),
                "regime_pair": s(row, "regime_pair"),
                "heading_test_acc_mean": f(row, "heading_test_acc_mean"),
                "range_test_acc_mean": f(row, "range_test_acc_mean"),
                "heading_self_mean_sq_cos": f(row, "heading_self_mean_sq_cos"),
                "range_self_mean_sq_cos": f(row, "range_self_mean_sq_cos"),
                "cross_mean_sq_cos": f(row, "cross_mean_sq_cos"),
                "cross_max_cos": f(row, "cross_max_cos"),
                "cross_to_self_ratio": f(row, "cross_to_self_ratio"),
                "non_overlap_score": f(row, "non_overlap_score"),
            }
        )
    return out


def enrich_r7_recovery(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        component = s(row, "component")
        if component.startswith("range"):
            axis_family = "range"
        elif component.startswith("heading"):
            axis_family = "heading"
        elif component == "all_head":
            axis_family = "all"
        else:
            axis_family = "other"
        recovery = f(row, "recovery_frac_of_target_feature_gap")
        out.append(
            {
                "target_case": s(row, "target_case"),
                "group_key": s(row, "group_key"),
                "group_value": s(row, "group_value"),
                "component": component,
                "axis_family": axis_family,
                "count": i(row, "count"),
                "feature_only_improvement": f(row, "feature_only_improvement"),
                "target_normal_improvement": f(row, "target_normal_improvement"),
                "mixed_improvement": f(row, "mixed_improvement"),
                "mixed_minus_feature_only": f(row, "mixed_minus_feature_only"),
                "target_normal_minus_feature_only": f(row, "target_normal_minus_feature_only"),
                "recovery_frac_of_target_feature_gap": recovery,
                "mixed_mean_error": f(row, "mixed_mean_error"),
                "feature_only_mean_error": f(row, "feature_only_mean_error"),
                "target_normal_mean_error": f(row, "target_normal_mean_error"),
                "component_rank_score": recovery,
            }
        )
    out.sort(
        key=lambda row: (
            str(row["target_case"]),
            str(row["group_key"]),
            str(row["group_value"]),
            -float(row["component_rank_score"])
            if math.isfinite(float(row["component_rank_score"]))
            else 0.0,
            str(row["component"]),
        )
    )
    current_key: tuple[str, str, str] | None = None
    rank = 0
    for row in out:
        key = (str(row["target_case"]), str(row["group_key"]), str(row["group_value"]))
        if key != current_key:
            current_key = key
            rank = 1
        else:
            rank += 1
        row["rank_within_target_group"] = rank
    return out


def summarize_neighbor_drift(r6_prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    proxy_cols = [
        "traj_heading_circ_std_deg",
        "traj_heading_max_drift_to_final_deg",
        "traj_heading_mean_drift_to_final_deg",
        "traj_range_std",
        "traj_range_span",
        "traj_range_max_drift_to_final",
        "traj_range_mean_drift_to_final",
    ]
    out: list[dict[str, Any]] = [
        {
            "source": "true_latent_neighbor_drift",
            "measurement_status": "missing_true_latent_neighbor_artifact",
            "interpretation": "R2 has no saved cross-checkpoint latent nearest-neighbor graph yet.",
        }
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in r6_prediction_rows:
        if s(row, "mode") == "reference_final":
            groups[(s(row, "target_case"), s(row, "mode"))].append(row)
    for (target_case, mode), selected in groups.items():
        summary: dict[str, Any] = {
            "source": "prediction_space_trajectory_drift_proxy",
            "measurement_status": "available_proxy_not_latent_neighbor_drift",
            "target_case": target_case,
            "mode": mode,
            "count": len(selected),
        }
        for col in proxy_cols:
            summary[f"mean_{col}"] = safe_mean([f(row, col) for row in selected])
            summary[f"max_{col}"] = safe_max([f(row, col) for row in selected])
        out.append(summary)
    return out


def find_summary_row(rows: list[dict[str, Any]], group_key: str, group_value: str) -> dict[str, Any] | None:
    for row in rows:
        if s(row, "group_key") == group_key and s(row, "group_value") == group_value:
            return row
    return None


def top_r7_all_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    selected = [row for row in rows if s(row, "group_key") == "ALL" and s(row, "group_value") == "ALL"]
    return sorted(
        selected,
        key=lambda row: (
            str(row["target_case"]),
            -float(row["component_rank_score"])
            if math.isfinite(float(row["component_rank_score"]))
            else 0.0,
        ),
    )[:limit]


def top_r7_support_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if s(row, "group_key") == "case_matches_best_final"
        and s(row, "group_value") == "1"
        and f(row, "target_normal_minus_feature_only") > 0.0
        and f(row, "target_normal_improvement") > 0.0
    ]
    return sorted(
        selected,
        key=lambda row: (
            str(row["target_case"]),
            -float(row["component_rank_score"])
            if math.isfinite(float(row["component_rank_score"]))
            else 0.0,
            str(row["component"]),
        ),
    )[:limit]


def build_verdict(
    coupling_rows: list[dict[str, Any]],
    feature_alignment_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_winner = find_summary_row(coupling_rows, "target_is_best_final", "1")
    axis_mismatch = find_summary_row(coupling_rows, "axis_mismatch", "1")
    all_coupling = find_summary_row(coupling_rows, "ALL", "ALL")
    winner_gap = f(target_winner or {}, "mean_positive_coupling_gap", math.nan)
    winner_coupled_only = f(target_winner or {}, "coupled_only_frac", math.nan)
    axis_gap = f(axis_mismatch or {}, "mean_positive_coupling_gap", math.nan)
    all_gap = f(all_coupling or {}, "mean_positive_coupling_gap", math.nan)

    adapted_rows = [
        row
        for row in feature_alignment_rows
        if "adapted" in s(row, "mode") and s(row, "group_key") == "ALL" and s(row, "group_value") == "ALL"
    ]
    best_adapted = max(
        adapted_rows,
        key=lambda row: f(row, "mean_improvement_vs_reference", -math.inf),
        default=None,
    )
    support_readout_rows = top_r7_support_rows(readout_rows, limit=len(readout_rows))
    top_readout_support = max(
        [row for row in support_readout_rows if s(row, "component") != "all_head"],
        key=lambda row: f(row, "component_rank_score", -math.inf),
        default=None,
    )
    support_flags = {
        "target_winner_positive_coupling_gap": math.isfinite(winner_gap) and winner_gap > 0.0,
        "axis_mismatch_positive_coupling_gap": math.isfinite(axis_gap) and axis_gap > 0.0,
        "all_positive_samples_positive_coupling_gap": math.isfinite(all_gap) and all_gap > 0.0,
        "r7_target_winner_has_strong_readout_recovery": f(top_readout_support or {}, "component_rank_score", math.nan)
        >= 0.5,
    }
    if (
        support_flags["target_winner_positive_coupling_gap"]
        and support_flags["r7_target_winner_has_strong_readout_recovery"]
    ):
        verdict = "weak_partial_support_target_winner_readout_side_coadaptation"
    elif support_flags["target_winner_positive_coupling_gap"]:
        verdict = "weak_support_feature_head_coupling_without_strong_readout_localization"
    else:
        verdict = "inconclusive_or_weak_for_feature_readout_phase_lag"
    return {
        "verdict": verdict,
        "scope": "val811_existing_phase102_artifacts_only",
        "not_proven": [
            "true_cross_checkpoint_layer_similarity",
            "true_latent_nearest_neighbor_drift",
            "cross_split_feature_head_mechanism",
        ],
        "support_flags": support_flags,
        "target_winner_mean_positive_coupling_gap": winner_gap,
        "target_winner_coupled_only_frac": winner_coupled_only,
        "axis_mismatch_mean_positive_coupling_gap": axis_gap,
        "all_positive_samples_mean_coupling_gap": all_gap,
        "best_r6_adapted_alignment_row": best_adapted,
        "top_r7_target_winner_non_all_readout_row": top_readout_support,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    verdict = summary["verdict"]
    top_r7 = verdict.get("top_r7_target_winner_non_all_readout_row") or {}
    best_r6 = verdict.get("best_r6_adapted_alignment_row") or {}
    lines = [
        "# Phase103-R2 Feature/Readout Phase-Lag Probe",
        "",
        "This is a val811 mechanism analysis over existing Phase102 R5/R6/R7 artifacts.",
        "It tests one candidate explanation: checkpoint-specific gains may require",
        "feature/readout co-adaptation rather than feature-only, head-only, or simple",
        "global feature alignment.",
        "",
        "## Scope",
        "",
        f"- output_dir: `{summary['output_dir']}`",
        f"- r5_attribution_rows: `{summary['r5_attribution_rows']}`",
        f"- r6_prediction_rows: `{summary['r6_prediction_rows']}`",
        f"- r7_recovery_rows: `{summary['r7_recovery_rows']}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "- hidden official test labels: `False`",
        "- leaderboard feedback: `False`",
        "",
        "## Verdict",
        "",
        f"- status: `{verdict['verdict']}`",
        f"- target-winner positive coupling gap: `{verdict['target_winner_mean_positive_coupling_gap']}`",
        f"- target-winner coupled-only fraction: `{verdict['target_winner_coupled_only_frac']}`",
        f"- axis-mismatch positive coupling gap: `{verdict['axis_mismatch_mean_positive_coupling_gap']}`",
        "",
        "## Best R6 Simple Alignment Row",
        "",
        f"- target_case: `{best_r6.get('target_case', '')}`",
        f"- mode: `{best_r6.get('mode', '')}`",
        f"- variant/layer: `{best_r6.get('variant', '')}/{best_r6.get('layer_spec', '')}`",
        f"- group: `{best_r6.get('group_key', '')}={best_r6.get('group_value', '')}`",
        f"- mean_improvement_vs_reference: `{best_r6.get('mean_improvement_vs_reference', '')}`",
        "",
        "## Top R7 Target-Winner Non-All Readout Row",
        "",
        f"- target_case: `{top_r7.get('target_case', '')}`",
        f"- component: `{top_r7.get('component', '')}`",
        f"- group: `{top_r7.get('group_key', '')}={top_r7.get('group_value', '')}`",
        f"- recovery_frac_of_target_feature_gap: `{top_r7.get('recovery_frac_of_target_feature_gap', '')}`",
        "",
        "## Limitations",
        "",
        "- R2 mechanism evidence is val811-only because R5/R6/R7 feature/readout artifacts are val811-only.",
        "- `phase103_r2_layer_similarity_summary.csv` includes R6 feature-alignment proxy rows and Phase95 final-only subspace proxy rows; it is not a true cross-checkpoint CKA result.",
        "- `phase103_r2_neighbor_drift_summary.csv` records that true latent nearest-neighbor drift has not been generated yet.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r5_path = Path(args.r5_attribution)
    r6_prediction_path = Path(args.r6_prediction)
    r6_recovery_path = Path(args.r6_recovery)
    r7_recovery_path = Path(args.r7_recovery)
    phase95_path = Path(args.phase95_subspace_summary)
    require_file(r5_path, "R5 attribution")
    require_file(r6_prediction_path, "R6 prediction")
    require_file(r6_recovery_path, "R6 recovery")
    require_file(r7_recovery_path, "R7 recovery")

    r5_rows = enrich_r5_rows(read_csv(r5_path))
    r6_prediction_rows = read_csv(r6_prediction_path)
    r6_recovery_rows = read_csv(r6_recovery_path)
    r7_recovery_rows = read_csv(r7_recovery_path)

    coupling_rows = summarize_r5_coupling(r5_rows, args.min_group_count)
    feature_alignment_rows = summarize_prediction_rows(r6_prediction_rows, args.min_group_count)
    r6_layer_rows = summarize_r6_recovery(r6_recovery_rows)
    phase95_layer_rows = summarize_phase95_subspace(phase95_path)
    layer_similarity_rows = r6_layer_rows + phase95_layer_rows
    readout_rows = enrich_r7_recovery(r7_recovery_rows)
    neighbor_rows = summarize_neighbor_drift(r6_prediction_rows)
    verdict = build_verdict(coupling_rows, feature_alignment_rows, readout_rows)

    write_csv(output_dir / "phase103_r2_feature_head_coupling_by_bucket.csv", coupling_rows)
    write_csv(output_dir / "phase103_r2_feature_alignment_summary.csv", feature_alignment_rows)
    write_csv(output_dir / "phase103_r2_layer_similarity_summary.csv", layer_similarity_rows)
    write_csv(output_dir / "phase103_r2_readout_submodule_summary.csv", readout_rows)
    write_csv(output_dir / "phase103_r2_neighbor_drift_summary.csv", neighbor_rows)

    summary = {
        "phase": "phase103_r2_feature_readout_phase_lag_probe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "inputs": {
            "r5_attribution": str(r5_path),
            "r6_prediction": str(r6_prediction_path),
            "r6_recovery": str(r6_recovery_path),
            "r7_recovery": str(r7_recovery_path),
            "phase95_subspace_summary": str(phase95_path),
        },
        "r5_attribution_rows": len(r5_rows),
        "r6_prediction_rows": len(r6_prediction_rows),
        "r6_recovery_rows": len(r6_recovery_rows),
        "r7_recovery_rows": len(r7_recovery_rows),
        "coupling_summary_rows": len(coupling_rows),
        "feature_alignment_summary_rows": len(feature_alignment_rows),
        "layer_similarity_summary_rows": len(layer_similarity_rows),
        "readout_submodule_summary_rows": len(readout_rows),
        "neighbor_drift_summary_rows": len(neighbor_rows),
        "top_coupling_rows": coupling_rows[:10],
        "top_readout_all_rows": top_r7_all_rows(readout_rows),
        "top_readout_target_winner_support_rows": top_r7_support_rows(readout_rows),
        "verdict": verdict,
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "diagnostic_only": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "phase103_r2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "phase103_r2_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
