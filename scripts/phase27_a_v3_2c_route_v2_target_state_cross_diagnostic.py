"""Target x A-state cross diagnostic for route-v2 multitarget baseline audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def safe_float(value: Any):
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def fmt(value):
    return "" if value is None else f"{value:.12g}"


def mean(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-surface", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_csv(args.shared_surface) if r.get("join_ok") == "1"]
    by_target = defaultdict(list)
    by_target_state = defaultdict(list)
    by_state = defaultdict(list)
    for row in rows:
        target = row.get("target_key", "")
        state = row.get("state", "")
        error = safe_float(row.get("joint_error"))
        by_target[target].append(error)
        by_state[state].append(error)
        by_target_state[(target, state)].append(error)

    global_mean = mean([safe_float(row.get("joint_error")) for row in rows])
    target_mean = {target: mean(values) for target, values in by_target.items()}
    state_mean = {state: mean(values) for state, values in by_state.items()}
    cross_rows = []
    residuals_by_state = defaultdict(list)
    for (target, state), values in sorted(by_target_state.items()):
        cell_mean = mean(values)
        target_centered = None if cell_mean is None or target_mean.get(target) is None else cell_mean - target_mean[target]
        additive_residual = None
        if cell_mean is not None and target_mean.get(target) is not None and state_mean.get(state) is not None and global_mean is not None:
            additive_residual = cell_mean - target_mean[target] - state_mean[state] + global_mean
        residuals_by_state[state].append(target_centered)
        cross_rows.append(
            {
                "target_key": target,
                "state": state,
                "count": len(values),
                "mean_error": fmt(cell_mean),
                "target_mean_error": fmt(target_mean.get(target)),
                "state_mean_error": fmt(state_mean.get(state)),
                "target_centered_state_delta": fmt(target_centered),
                "additive_residual": fmt(additive_residual),
            }
        )
    residual_summary = []
    for state, values in sorted(residuals_by_state.items()):
        residual_summary.append(
            {
                "state": state,
                "target_centered_mean": fmt(mean(values)),
                "target_centered_abs_mean": fmt(mean([abs(v) for v in values if v is not None])),
                "target_cell_count": len([v for v in values if v is not None]),
            }
        )
    metrics = {
        "row_count": len(rows),
        "target_count": len(by_target),
        "state_count": len(by_state),
        "global_mean_error": global_mean,
        "target_mean_error": {k: target_mean[k] for k in sorted(target_mean)},
        "state_mean_error": {k: state_mean[k] for k in sorted(state_mean)},
        "interpretation": "target_centered_state_delta isolates state effect after subtracting target mean; large target means indicate target heterogeneity dominates global means.",
    }
    out = args.output_dir
    write_csv(out / "tables" / "route_v2_target_state_cross_report.csv", cross_rows, [
        "target_key",
        "state",
        "count",
        "mean_error",
        "target_mean_error",
        "state_mean_error",
        "target_centered_state_delta",
        "additive_residual",
    ])
    write_csv(out / "tables" / "route_v2_target_centered_state_summary.csv", residual_summary, [
        "state",
        "target_centered_mean",
        "target_centered_abs_mean",
        "target_cell_count",
    ])
    (out / "metrics" / "route_v2_target_state_cross_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# A-v3.2c Route-v2 Target x State Cross Diagnostic",
        "",
        f"- row_count: `{len(rows)}`",
        f"- target_count: `{len(by_target)}`",
        f"- state_count: `{len(by_state)}`",
        f"- global_mean_error: `{fmt(global_mean)}`",
        "",
        "## Target-Centered State Summary",
    ]
    for row in residual_summary:
        lines.append(
            f"- {row['state']}: target_centered_mean={row['target_centered_mean']}, "
            f"target_centered_abs_mean={row['target_centered_abs_mean']}, cells={row['target_cell_count']}"
        )
    lines.extend([
        "",
        "This diagnostic is analysis-only. No training, threshold tuning, B/C gate, full eval, or submission was run.",
    ])
    (out / "reports" / "route_v2_target_state_cross_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
