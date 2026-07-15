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


def pct_delta(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def gate(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "actual": float(actual),
        "threshold": float(threshold),
        "direction": "lower_or_equal",
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
        "g1_direct_angle": gate("G1 direct angle", angle, G1_ANGLE_MAE),
        "g2_strong_angle": gate("G2 strong angle", angle, G2_ANGLE_MAE),
        "g3_distance_protection": gate("G3 distance protection", distance, G3_DISTANCE_MAE),
        "g0_angle_non_regression": gate("G0 angle non-regression", angle, BASELINE_ANGLE_MAE),
        "g0_distance_non_regression": gate("G0 distance non-regression", distance, BASELINE_DISTANCE_MAE),
    }
    if gates["g2_strong_angle"]["pass"] and gates["g3_distance_protection"]["pass"]:
        decision = "promote_to_scale_review"
    elif gates["g1_direct_angle"]["pass"] and gates["g3_distance_protection"]["pass"]:
        decision = "hold_for_repeat_or_low_risk_packaging_review"
    elif gates["g0_angle_non_regression"]["pass"] and gates["g3_distance_protection"]["pass"]:
        decision = "hold_for_repeat_checkpoint_policy"
    else:
        decision = "kill_or_redesign"
    return {
        "phase": "Phase60",
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
            "angle_mae_deg": pct_delta(angle, BASELINE_ANGLE_MAE),
            "distance_mae": pct_delta(distance, BASELINE_DISTANCE_MAE),
        },
        "gates": gates,
        "decision": decision,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    deltas = report["delta_vs_rank1_pct"]
    gates = report["gates"]
    lines = [
        f"# Phase60 Gate Report: {report['run_name']}",
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
    ]
    for key in ("g0_angle_non_regression", "g0_distance_non_regression", "g1_direct_angle", "g2_strong_angle", "g3_distance_protection"):
        item = gates[key]
        lines.append(f"| {item['name']} | {item['actual']:.9f} | {item['threshold']:.9f} | {item['pass']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    report = build_gate_report(
        run_name=args.run_name,
        metrics=load_eval_metrics(Path(args.metrics_json)),
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "out_dir": str(out_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
