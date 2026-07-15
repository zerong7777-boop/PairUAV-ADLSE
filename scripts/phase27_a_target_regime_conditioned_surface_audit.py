#!/usr/bin/env python3
"""Target-regime-conditioned evidence-state validation surface audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FORBIDDEN_ACTIONS_CONFIRMATION = (
    "No training, finetuning, sample weighting, threshold tuning, B/C gate, "
    "full eval, submission packaging, fuzzy join, silent deduplication, or "
    "leaderboard probing was run"
)


def _safe_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric value for {field}: {value!r}")
    return parsed


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float) -> float:
    return round(value, 6)


def _load_joined_rows(shared_surface: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    with shared_surface.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    joined_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("join_ok") != "1":
            continue
        evidence_state = row.get("state") or row.get("evidence_state")
        if not row.get("target_key"):
            raise ValueError("joined row missing target_key")
        if not evidence_state:
            raise ValueError("joined row missing state/evidence_state")
        joined_rows.append(
            {
                **row,
                "evidence_state": evidence_state,
                "joint_error_value": _safe_float(row.get("joint_error", ""), "joint_error"),
            }
        )
    return raw_rows, joined_rows


def _assign_target_regimes(target_means: dict[str, float]) -> dict[str, str]:
    ordered = sorted(target_means, key=lambda key: (target_means[key], key))
    target_count = len(ordered)
    if target_count == 1:
        return {ordered[0]: "blocked_single_target_regime"}
    if target_count == 2:
        return {
            ordered[0]: "easy_target_regime",
            ordered[1]: "hard_target_regime",
        }

    regimes: dict[str, str] = {}
    labels = ["easy_target_regime", "medium_target_regime", "hard_target_regime"]
    for idx, target_key in enumerate(ordered):
        regimes[target_key] = labels[min((idx * 3) // target_count, 2)]
    return regimes


def _stable_signed_state_count(
    cell_residuals: dict[tuple[str, str], list[float]],
    evidence_states: list[str],
    min_cell_count: int,
) -> int:
    stable_count = 0
    for state in evidence_states:
        signs = []
        for (regime, cell_state), residuals in cell_residuals.items():
            del regime
            if cell_state != state or len(residuals) < min_cell_count:
                continue
            residual_mean = _mean(residuals)
            if abs(residual_mean) < 5.0:
                continue
            signs.append(1 if residual_mean > 0 else -1)
        if len(signs) >= 2 and len(set(signs)) == 1:
            stable_count += 1
    return stable_count


def run_audit(
    shared_surface: Path,
    output_dir: Path,
    min_joined_fraction: float = 0.95,
    min_cell_count: int = 20,
) -> dict[str, Any]:
    """Run the target-regime-conditioned validation-only surface audit."""

    shared_surface = Path(shared_surface)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    metrics_dir = output_dir / "metrics"
    reports_dir = output_dir / "reports"
    for directory in (tables_dir, metrics_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_rows, joined_rows = _load_joined_rows(shared_surface)
    total_row_count = len(raw_rows)
    joined_row_count = len(joined_rows)
    joined_fraction = joined_row_count / total_row_count if total_row_count else 0.0

    by_target: dict[str, list[float]] = defaultdict(list)
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in joined_rows:
        by_target[row["target_key"]].append(row["joint_error_value"])
        by_state[row["evidence_state"]].append(row["joint_error_value"])

    target_means = {target: _mean(values) for target, values in by_target.items()}
    target_regimes = _assign_target_regimes(target_means) if target_means else {}
    for row in joined_rows:
        row["target_mean_error"] = target_means[row["target_key"]]
        row["target_regime"] = target_regimes[row["target_key"]]
        row["target_centered_residual"] = (
            row["joint_error_value"] - row["target_mean_error"]
        )

    regime_targets: dict[str, set[str]] = defaultdict(set)
    regime_values: dict[str, list[float]] = defaultdict(list)
    regime_residuals: dict[str, list[float]] = defaultdict(list)
    state_residuals: dict[str, list[float]] = defaultdict(list)
    cell_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_target_residuals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in joined_rows:
        regime = row["target_regime"]
        state = row["evidence_state"]
        regime_targets[regime].add(row["target_key"])
        regime_values[regime].append(row["joint_error_value"])
        cell_values[(regime, state)].append(row["joint_error_value"])
        cell_target_residuals[(regime, state)].append(row["target_centered_residual"])
        state_residuals[state].append(row["target_centered_residual"])

    regime_means = {regime: _mean(values) for regime, values in regime_values.items()}
    global_mean = _mean([row["joint_error_value"] for row in joined_rows])
    state_means = {state: _mean(values) for state, values in by_state.items()}

    for row in joined_rows:
        regime = row["target_regime"]
        row["regime_centered_residual"] = (
            row["joint_error_value"] - regime_means[regime]
        )
        regime_residuals[regime].append(row["regime_centered_residual"])

    target_report_rows = []
    for target_key in sorted(by_target, key=lambda key: (target_means[key], key)):
        values = by_target[target_key]
        target_report_rows.append(
            {
                "target_key": target_key,
                "target_regime": target_regimes[target_key],
                "count": len(values),
                "mean_joint_error": _round(target_means[target_key]),
            }
        )

    state_report_rows = []
    for state in sorted(by_state):
        residuals = state_residuals[state]
        state_report_rows.append(
            {
                "evidence_state": state,
                "count": len(by_state[state]),
                "mean_joint_error": _round(_mean(by_state[state])),
                "target_centered_residual_mean": _round(_mean(residuals)),
                "target_centered_residual_abs_mean": _round(
                    _mean([abs(value) for value in residuals])
                ),
            }
        )

    regime_report_rows = []
    for regime in sorted(regime_values):
        residuals = regime_residuals[regime]
        regime_report_rows.append(
            {
                "target_regime": regime,
                "target_count": len(regime_targets[regime]),
                "row_count": len(regime_values[regime]),
                "mean_joint_error": _round(regime_means[regime]),
                "regime_centered_residual_mean": _round(_mean(residuals)),
            }
        )

    surface_rows = []
    coverage_rows = []
    for regime, state in sorted(cell_values):
        values = cell_values[(regime, state)]
        target_residuals = cell_target_residuals[(regime, state)]
        cell_mean = _mean(values)
        additive_residual = cell_mean - regime_means[regime] - state_means[state] + global_mean
        meets_min_cell_count = len(values) >= min_cell_count
        surface_rows.append(
            {
                "target_regime": regime,
                "evidence_state": state,
                "count": len(values),
                "mean_joint_error": _round(cell_mean),
                "target_centered_residual_mean": _round(_mean(target_residuals)),
                "regime_centered_residual_mean": _round(
                    _mean([value - regime_means[regime] for value in values])
                ),
                "additive_residual": _round(additive_residual),
                "meets_min_cell_count": str(meets_min_cell_count).lower(),
            }
        )
        coverage_rows.append(
            {
                "target_regime": regime,
                "evidence_state": state,
                "count": len(values),
                "min_cell_count": min_cell_count,
                "meets_min_cell_count": str(meets_min_cell_count).lower(),
            }
        )

    residual_rows = []
    for regime, state in sorted(cell_target_residuals):
        residuals = cell_target_residuals[(regime, state)]
        residual_mean = _mean(residuals)
        residual_rows.append(
            {
                "target_regime": regime,
                "evidence_state": state,
                "count": len(residuals),
                "target_centered_residual_mean": _round(residual_mean),
                "target_centered_residual_abs_mean": _round(
                    _mean([abs(value) for value in residuals])
                ),
                "target_centered_residual_sign": (
                    "positive"
                    if residual_mean > 0
                    else "negative"
                    if residual_mean < 0
                    else "zero"
                ),
            }
        )

    _write_csv(
        tables_dir / "global_target_report.csv",
        ["target_key", "target_regime", "count", "mean_joint_error"],
        target_report_rows,
    )
    _write_csv(
        tables_dir / "global_evidence_state_report.csv",
        [
            "evidence_state",
            "count",
            "mean_joint_error",
            "target_centered_residual_mean",
            "target_centered_residual_abs_mean",
        ],
        state_report_rows,
    )
    _write_csv(
        tables_dir / "target_regime_report.csv",
        [
            "target_regime",
            "target_count",
            "row_count",
            "mean_joint_error",
            "regime_centered_residual_mean",
        ],
        regime_report_rows,
    )
    _write_csv(
        tables_dir / "target_regime_evidence_state_surface.csv",
        [
            "target_regime",
            "evidence_state",
            "count",
            "mean_joint_error",
            "target_centered_residual_mean",
            "regime_centered_residual_mean",
            "additive_residual",
            "meets_min_cell_count",
        ],
        surface_rows,
    )
    _write_csv(
        tables_dir / "target_centered_residual_report.csv",
        [
            "target_regime",
            "evidence_state",
            "count",
            "target_centered_residual_mean",
            "target_centered_residual_abs_mean",
            "target_centered_residual_sign",
        ],
        residual_rows,
    )
    _write_csv(
        tables_dir / "cell_coverage_report.csv",
        [
            "target_regime",
            "evidence_state",
            "count",
            "min_cell_count",
            "meets_min_cell_count",
        ],
        coverage_rows,
    )

    target_regime_mean_gap = (
        max(regime_means.values()) - min(regime_means.values()) if len(regime_means) >= 2 else 0.0
    )
    max_abs_state_residual_mean = (
        max(abs(_mean(values)) for values in state_residuals.values())
        if state_residuals
        else 0.0
    )
    cells_meeting_min_count = sum(
        1 for values in cell_values.values() if len(values) >= min_cell_count
    )
    critical_cell_coverage_failure = bool(cell_values) and any(
        len(values) < min_cell_count for values in cell_values.values()
    )
    target_regime_count = len(regime_values)
    evidence_state_count = len(by_state)
    stable_signed_state_count = _stable_signed_state_count(
        cell_target_residuals, sorted(by_state), min_cell_count
    )

    if joined_fraction < min_joined_fraction:
        verdict = "blocked-coverage"
        reason = "joined_fraction_below_threshold"
    elif target_regime_count < 2 or evidence_state_count < 2:
        verdict = "blocked-coverage"
        reason = "insufficient_target_regime_or_evidence_state_coverage"
    elif cells_meeting_min_count == 0:
        verdict = "blocked-coverage"
        reason = "insufficient_cell_coverage"
    elif (
        target_regime_mean_gap >= 10.0
        and max_abs_state_residual_mean >= 5.0
        and stable_signed_state_count >= 1
        and not critical_cell_coverage_failure
    ):
        verdict = "joint-surface-supported"
        reason = "target_regime_gap_and_stable_evidence_state_residuals"
    elif target_regime_mean_gap >= 10.0 and max_abs_state_residual_mean < 5.0:
        verdict = "target-only-dominant"
        reason = "target_regime_gap_without_material_evidence_state_residual"
    elif target_regime_mean_gap < 10.0 and max_abs_state_residual_mean < 5.0:
        verdict = "evidence-state-weak"
        reason = "weak_target_regime_gap_and_weak_evidence_state_residual"
    else:
        verdict = "evidence-state-conditional"
        reason = "evidence_state_residual_not_stable_across_regimes"

    metrics: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "shared_surface": str(shared_surface),
        "total_row_count": total_row_count,
        "joined_row_count": joined_row_count,
        "joined_fraction": _round(joined_fraction),
        "min_joined_fraction": min_joined_fraction,
        "target_count": len(by_target),
        "target_regime_count": target_regime_count,
        "evidence_state_count": evidence_state_count,
        "cell_count": len(cell_values),
        "cells_meeting_min_count": cells_meeting_min_count,
        "min_cell_count": min_cell_count,
        "critical_cell_coverage_failure": critical_cell_coverage_failure,
        "target_regime_mean_gap": _round(target_regime_mean_gap),
        "max_abs_state_residual_mean": _round(max_abs_state_residual_mean),
        "stable_signed_state_count": stable_signed_state_count,
        "target_regime_distribution": {
            regime: len(values) for regime, values in sorted(regime_values.items())
        },
        "evidence_state_distribution": {
            state: len(values) for state, values in sorted(by_state.items())
        },
        "target_regime_definition": (
            "validation_only: target_key mean joint_error sorted into easy/medium/hard "
            "tertiles; two targets use easy/hard; one target blocks coverage"
        ),
        "forbidden_actions_confirmation": FORBIDDEN_ACTIONS_CONFIRMATION,
    }

    leakage_deployability = {
        "target_regime": {
            "source": "validation_baseline_joint_error_by_target",
            "uses_gt_or_error": True,
            "deployability": "validation_only",
            "inference_time_allowed": False,
        },
        "evidence_state": {
            "source": "reacquired_state/state field from fixed manifest outcome surface",
            "uses_gt_or_error": False,
            "deployability": "candidate_deployable_subject_to_source_fields",
            "inference_time_allowed": "not_claimed_in_this_spec",
        },
    }

    (metrics_dir / "target_regime_conditioned_surface_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (metrics_dir / "leakage_deployability_audit.json").write_text(
        json.dumps(leakage_deployability, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Target-Regime-Conditioned Evidence-State Validation Surface",
        "",
        f"verdict: `{verdict}`",
        f"reason: `{reason}`",
        "",
        f"- joined_fraction: `{metrics['joined_fraction']}`",
        f"- target_regime_count: `{target_regime_count}`",
        f"- evidence_state_count: `{evidence_state_count}`",
        f"- cell_count: `{len(cell_values)}`",
        f"- target_regime_mean_gap: `{metrics['target_regime_mean_gap']}`",
        f"- max_abs_state_residual_mean: `{metrics['max_abs_state_residual_mean']}`",
        f"- stable_signed_state_count: `{stable_signed_state_count}`",
        "",
        "Target regimes are validation-only because they are derived from baseline joint_error.",
        "",
        FORBIDDEN_ACTIONS_CONFIRMATION + ".",
        "",
    ]
    (reports_dir / "target_regime_conditioned_surface_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    verdict_report = [
        f"verdict: `{verdict}`",
        f"reason: `{reason}`",
        "",
        FORBIDDEN_ACTIONS_CONFIRMATION + ".",
        "",
    ]
    (reports_dir / "go_no_go_verdict.md").write_text(
        "\n".join(verdict_report), encoding="utf-8"
    )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit target-regime-conditioned evidence-state validation surfaces."
    )
    parser.add_argument("shared_surface", nargs="?", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--shared-surface", dest="shared_surface_opt", type=Path)
    parser.add_argument("--output-dir", dest="output_dir_opt", type=Path)
    parser.add_argument("--min-joined-fraction", type=float, default=0.95)
    parser.add_argument("--min-cell-count", type=int, default=20)
    args = parser.parse_args()

    shared_surface = args.shared_surface_opt or args.shared_surface
    output_dir = args.output_dir_opt or args.output_dir
    if shared_surface is None or output_dir is None:
        parser.error("shared_surface and output_dir are required")

    metrics = run_audit(
        shared_surface=shared_surface,
        output_dir=output_dir,
        min_joined_fraction=args.min_joined_fraction,
        min_cell_count=args.min_cell_count,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
