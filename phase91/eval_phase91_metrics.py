from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

from .common import DEFAULT_RUN_ROOT, circular_abs_error_deg, ensure_run_root, write_json, write_text


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(_to_float_list(value))
        else:
            out.append(float(value))
    return out


def _check_same_length(*vectors: list[float]) -> int:
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise ValueError(f"Metric inputs must have equal lengths, got {sorted(lengths)}")
    sample_count = lengths.pop()
    if sample_count <= 0:
        raise ValueError("Metric inputs require at least one sample")
    return sample_count


def polar_xy(heading_deg: float, range_value: float) -> tuple[float, float]:
    rad = math.radians(float(heading_deg))
    radius = float(range_value)
    return math.cos(rad) * radius, math.sin(rad) * radius


def displacement_epe(
    heading_prediction_deg: Iterable[float],
    range_prediction: Iterable[float],
    heading_target_deg: Iterable[float],
    range_target: Iterable[float],
) -> float:
    pred_heading = _to_float_list(heading_prediction_deg)
    pred_range = _to_float_list(range_prediction)
    target_heading = _to_float_list(heading_target_deg)
    target_range = _to_float_list(range_target)
    _check_same_length(pred_heading, pred_range, target_heading, target_range)

    errors = []
    for pred_h, pred_r, target_h, target_r in zip(pred_heading, pred_range, target_heading, target_range):
        pred_x, pred_y = polar_xy(pred_h, pred_r)
        target_x, target_y = polar_xy(target_h, target_r)
        errors.append(math.hypot(pred_x - target_x, pred_y - target_y))
    return sum(errors) / len(errors)


def compute_phase91_metrics(
    heading_prediction_deg: Iterable[float],
    heading_target_deg: Iterable[float],
    range_prediction: Iterable[float],
    range_target: Iterable[float],
    range_min: float,
    range_max: float,
) -> dict[str, float | int | str]:
    pred_heading = _to_float_list(heading_prediction_deg)
    target_heading = _to_float_list(heading_target_deg)
    pred_range = _to_float_list(range_prediction)
    target_range = _to_float_list(range_target)
    sample_count = _check_same_length(pred_heading, target_heading, pred_range, target_range)

    distance_denominator = float(range_max) - float(range_min)
    if distance_denominator <= 0.0:
        raise ValueError("range_max must be greater than range_min")

    angle_errors = [circular_abs_error_deg(pred, target) for pred, target in zip(pred_heading, target_heading)]
    distance_errors = [abs(pred - target) for pred, target in zip(pred_range, target_range)]
    angle_mae_deg = sum(angle_errors) / sample_count
    distance_mae = sum(distance_errors) / sample_count
    angle_rel_error = angle_mae_deg / 180.0
    distance_rel_error = distance_mae / distance_denominator

    full_epe = displacement_epe(pred_heading, pred_range, target_heading, target_range)
    heading_only_epe = displacement_epe(pred_heading, target_range, target_heading, target_range)
    range_only_epe = displacement_epe(target_heading, pred_range, target_heading, target_range)

    return {
        "samples": sample_count,
        "angle_mae_deg": angle_mae_deg,
        "distance_mae": distance_mae,
        "angle_rel_error": angle_rel_error,
        "distance_rel_error": distance_rel_error,
        "final_score_proxy": (angle_rel_error + distance_rel_error) / 2.0,
        "distance_mode": "range_span",
        "range_min": float(range_min),
        "range_max": float(range_max),
        "displacement_epe": full_epe,
        "heading_induced_displacement_error": heading_only_epe,
        "range_induced_displacement_error": range_only_epe,
    }


def phase91_metric_contract() -> dict[str, Any]:
    return {
        "phase": "phase91_metric_contract",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": {
            "angle_mae_deg": "Circular MAE in degrees after 360-degree wrap.",
            "distance_mae": "Mean absolute error of PairUAV range values.",
            "angle_rel_error": "angle_mae_deg / 180.",
            "distance_rel_error": "distance_mae / (range_max - range_min).",
            "final_score_proxy": "Mean of angle_rel_error and distance_rel_error.",
            "displacement_epe": "Mean Euclidean error after polar recomposition.",
            "heading_induced_displacement_error": "EPE from predicted heading with target range.",
            "range_induced_displacement_error": "EPE from target heading with predicted range.",
            "reversal_violation": "Optional; disabled unless G1 coordinate convention verdict is pass.",
        },
        "defaults": {
            "distance_mode": "range_span",
            "reversal_loss_enabled_by_default": False,
        },
    }


def run_synthetic_tests() -> dict[str, Any]:
    wrap_payload = compute_phase91_metrics(
        heading_prediction_deg=[181.0, -179.0],
        heading_target_deg=[-179.0, 179.0],
        range_prediction=[10.0, 12.0],
        range_target=[10.0, 10.0],
        range_min=0.0,
        range_max=20.0,
    )
    recomposition = displacement_epe([0.0], [10.0], [90.0], [10.0])
    tests = {
        "wrap_angle_mae_expected_1deg": math.isclose(wrap_payload["angle_mae_deg"], 1.0),
        "range_mae_expected_1": math.isclose(wrap_payload["distance_mae"], 1.0),
        "proxy_expected": math.isclose(
            wrap_payload["final_score_proxy"],
            ((1.0 / 180.0) + (1.0 / 20.0)) / 2.0,
        ),
        "polar_recomposition_epe_expected": math.isclose(recomposition, math.sqrt(200.0)),
    }
    return {
        "phase": "phase91_metric_synthetic_tests",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pass": all(tests.values()),
        "tests": tests,
        "example_payload": wrap_payload,
        "recomposition_example_epe": recomposition,
    }


def write_metric_contract(run_root: Path) -> dict[str, Any]:
    ensure_run_root(run_root)
    contract = phase91_metric_contract()
    synthetic = run_synthetic_tests()
    write_json(run_root / "diagnostics" / "metric_synthetic_tests.json", synthetic)

    md = [
        "# Phase91 Shared Metric Contract",
        "",
        f"- created_at: `{contract['created_at']}`",
        f"- distance_mode: `{contract['defaults']['distance_mode']}`",
        f"- reversal_loss_enabled_by_default: {contract['defaults']['reversal_loss_enabled_by_default']}",
        f"- synthetic_tests_pass: {synthetic['pass']}",
        "",
        "## Metrics",
        "",
    ]
    for name, description in contract["metrics"].items():
        md.append(f"- `{name}`: {description}")
    md.extend(
        [
            "",
            "## Synthetic Test Payload",
            "",
            "```json",
            json.dumps(synthetic["example_payload"], ensure_ascii=False, indent=2),
            "```",
            "",
            "This contract is for local Phase91 mechanism comparisons only. It is not hidden CodaBench feedback.",
        ]
    )
    write_text(run_root / "diagnostics" / "metric_contract.md", "\n".join(md))
    return {"contract": contract, "synthetic": synthetic}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase91 shared metric contract.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_metric_contract(args.run_root.resolve())
    print(json.dumps({"ok": payload["synthetic"]["pass"], "run_root": str(args.run_root)}, ensure_ascii=False))
    return 0 if payload["synthetic"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
