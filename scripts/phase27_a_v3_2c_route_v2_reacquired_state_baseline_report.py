"""State-wise baseline report for route-v2 fixed-manifest outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def quantile(values, q):
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def mean(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["canonical_pair_id"]: row for row in read_csv(args.manifest)}
    baseline = read_csv(args.baseline)
    by_state = defaultdict(list)
    missing_state = 0
    for row in baseline:
        m = manifest.get(row.get("canonical_pair_id", ""), {})
        state = m.get("reacquired_state") or m.get("base_regime") or m.get("candidate_state") or "unknown"
        if state == "unknown":
            missing_state += 1
        by_state[state].append(row)

    report = []
    for state, rows in sorted(by_state.items()):
        joint = [safe_float(r.get("joint_error")) for r in rows if r.get("row_status") == "ok"]
        heading = [safe_float(r.get("heading_abs_error")) for r in rows if r.get("row_status") == "ok"]
        rng = [safe_float(r.get("range_abs_error")) for r in rows if r.get("row_status") == "ok"]
        report.append(
            {
                "state": state,
                "count": str(len(rows)),
                "ok_count": str(sum(1 for r in rows if r.get("row_status") == "ok")),
                "mean_error": fmt(mean(joint)),
                "median_error": fmt(quantile(joint, 0.5)),
                "p90_error": fmt(quantile(joint, 0.9)),
                "p95_error": fmt(quantile(joint, 0.95)),
                "heading_error_mean": fmt(mean(heading)),
                "range_error_mean": fmt(mean(rng)),
            }
        )

    overall_joint = [safe_float(r.get("joint_error")) for r in baseline if r.get("row_status") == "ok"]
    metrics = {
        "row_count": len(baseline),
        "ok_count": sum(1 for r in baseline if r.get("row_status") == "ok"),
        "missing_state_count": missing_state,
        "state_count": len(report),
        "state_distribution": dict(Counter((manifest.get(r.get("canonical_pair_id", ""), {}).get("reacquired_state") or "unknown") for r in baseline)),
        "overall_mean_error": mean(overall_joint),
        "overall_median_error": quantile(overall_joint, 0.5),
        "verdict": "route-v2-baseline-state-report-ready" if len(report) >= 2 and missing_state == 0 else "route-v2-baseline-state-report-weak",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tables" / "route_v2_state_wise_baseline_report.csv", report, [
        "state",
        "count",
        "ok_count",
        "mean_error",
        "median_error",
        "p90_error",
        "p95_error",
        "heading_error_mean",
        "range_error_mean",
    ])
    write_json(args.output_dir / "metrics" / "route_v2_state_wise_baseline_metrics.json", metrics)
    lines = [
        "# A-v3.2c Route-v2 Reacquired-State Baseline Report",
        "",
        f"verdict: `{metrics['verdict']}`",
        f"row_count: `{metrics['row_count']}`",
        f"ok_count: `{metrics['ok_count']}`",
        f"state_count: `{metrics['state_count']}`",
        "",
        "## State Summary",
    ]
    for row in report:
        lines.append(
            f"- {row['state']}: count={row['count']}, mean_error={row['mean_error']}, "
            f"heading={row['heading_error_mean']}, range={row['range_error_mean']}"
        )
    lines.append("")
    lines.append("No training, finetuning, stress protocol, threshold tuning, B/C gate, full eval, or submission was run.")
    (args.output_dir / "reports" / "route_v2_state_wise_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "reports" / "route_v2_state_wise_baseline_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
