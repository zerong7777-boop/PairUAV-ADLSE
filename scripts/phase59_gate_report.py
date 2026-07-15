#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_ANGLE_MAE = 0.131944
BASELINE_DISTANCE_MAE = 0.042472
G1_ANGLE_MAE = 0.125347
G2_ANGLE_MAE = 0.118750
G3_DISTANCE_MAE = 0.043750


def load_eval_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("angle_mae_deg", "distance_mae", "samples")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing required metric fields {missing} in {path}")
    return {
        "angle_mae_deg": float(payload["angle_mae_deg"]),
        "distance_mae": float(payload["distance_mae"]),
        "samples": int(payload["samples"]),
        "final_score_proxy": float(payload.get("final_score_proxy", 0.0)),
        "distance_mode": str(payload.get("distance_mode", "")),
    }


def _pct_delta(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def _gate(name: str, actual: float, threshold: float, direction: str) -> dict[str, Any]:
    if direction != "lower_or_equal":
        raise ValueError(f"unsupported gate direction: {direction}")
    return {
        "name": name,
        "actual": float(actual),
        "threshold": float(threshold),
        "direction": direction,
        "pass": float(actual) <= float(threshold),
    }


def build_gate_report(
    *,
    run_name: str,
    metrics: dict[str, Any],
    base_checkpoint: str,
    run_dir: str,
    eval_dir: str,
) -> dict[str, Any]:
    angle = float(metrics["angle_mae_deg"])
    distance = float(metrics["distance_mae"])
    gates = {
        "g1_direct_angle": _gate("G1 direct angle", angle, G1_ANGLE_MAE, "lower_or_equal"),
        "g2_strong_angle": _gate("G2 strong angle", angle, G2_ANGLE_MAE, "lower_or_equal"),
        "g3_distance_protection": _gate("G3 distance protection", distance, G3_DISTANCE_MAE, "lower_or_equal"),
    }
    if gates["g2_strong_angle"]["pass"] and gates["g3_distance_protection"]["pass"]:
        decision = "promote_to_scale_review"
    elif gates["g1_direct_angle"]["pass"] and gates["g3_distance_protection"]["pass"]:
        decision = "hold_for_repeat_or_low_risk_packaging_review"
    elif gates["g1_direct_angle"]["pass"] and not gates["g3_distance_protection"]["pass"]:
        decision = "hold_or_kill_distance_regression"
    else:
        decision = "kill_or_redesign"
    return {
        "phase": "Phase59",
        "run_name": run_name,
        "base_checkpoint": base_checkpoint,
        "run_dir": run_dir,
        "eval_dir": eval_dir,
        "metrics": metrics,
        "baseline": {
            "angle_mae_deg": BASELINE_ANGLE_MAE,
            "distance_mae": BASELINE_DISTANCE_MAE,
        },
        "delta_vs_rank1_pct": {
            "angle_mae_deg": _pct_delta(angle, BASELINE_ANGLE_MAE),
            "distance_mae": _pct_delta(distance, BASELINE_DISTANCE_MAE),
        },
        "gates": gates,
        "decision": decision,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    deltas = report["delta_vs_rank1_pct"]
    gates = report["gates"]
    return "\n".join(
        [
            f"# Phase59 Gate Report: {report['run_name']}",
            "",
            f"- base_checkpoint: `{report['base_checkpoint']}`",
            f"- run_dir: `{report['run_dir']}`",
            f"- eval_dir: `{report['eval_dir']}`",
            f"- decision: `{report['decision']}`",
            "",
            "| metric | value | delta vs rank1 |",
            "|---|---:|---:|",
            f"| angle_mae_deg | {metrics['angle_mae_deg']:.9f} | {deltas['angle_mae_deg']:.3f}% |",
            f"| distance_mae | {metrics['distance_mae']:.9f} | {deltas['distance_mae']:.3f}% |",
            "",
            "| gate | actual | threshold | pass |",
            "|---|---:|---:|---|",
            f"| G1 direct angle | {gates['g1_direct_angle']['actual']:.9f} | {gates['g1_direct_angle']['threshold']:.9f} | {gates['g1_direct_angle']['pass']} |",
            f"| G2 strong angle | {gates['g2_strong_angle']['actual']:.9f} | {gates['g2_strong_angle']['threshold']:.9f} | {gates['g2_strong_angle']['pass']} |",
            f"| G3 distance protection | {gates['g3_distance_protection']['actual']:.9f} | {gates['g3_distance_protection']['threshold']:.9f} | {gates['g3_distance_protection']['pass']} |",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metrics = load_eval_metrics(Path(args.metrics_json))
    report = build_gate_report(
        run_name=args.run_name,
        metrics=metrics,
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
