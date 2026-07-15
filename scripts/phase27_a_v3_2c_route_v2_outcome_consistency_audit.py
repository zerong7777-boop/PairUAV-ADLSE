"""Route-v2 repaired baseline outcome-consistency audit for A-v3.2c.

This audit is intentionally baseline-only: after the checkpoint/model-contract
repair, no valid route-v2 stress/intervention runner has been established yet.
The script therefore measures whether reacquired A states explain baseline
outcomes, and separately marks intervention consistency as blocked rather than
pretending repeated no-op forwards are stress variants.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATE_COLUMNS = ["reacquired_state", "evidence_base_regime", "candidate_state", "base_regime", "state"]


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


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.12g}"


def quantile(values: list[float], q: float):
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


def mean(values: list[float]):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def choose_state(row: dict[str, str]) -> str:
    for col in STATE_COLUMNS:
        value = (row.get(col) or "").strip()
        if value:
            return value
    return "unknown_unlabeled"


def unique_index(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], int, int]:
    counts = Counter(row.get("canonical_pair_id", "") for row in rows)
    duplicate_ids = {cid for cid, count in counts.items() if cid and count > 1}
    indexed = {
        row.get("canonical_pair_id", ""): row
        for row in rows
        if row.get("canonical_pair_id", "") and row.get("canonical_pair_id", "") not in duplicate_ids
    }
    return indexed, len(duplicate_ids), counts.get("", 0)


def cliff_delta(a: list[float], b: list[float]):
    aa = [x for x in a if x is not None]
    bb = [x for x in b if x is not None]
    if not aa or not bb:
        return None
    greater = 0
    lower = 0
    for x in aa:
        for y in bb:
            if x > y:
                greater += 1
            elif x < y:
                lower += 1
    return (greater - lower) / (len(aa) * len(bb))


def build_shared(manifest: list[dict[str, str]], baseline: list[dict[str, str]]):
    baseline_by_id, duplicate_baseline_ids, missing_baseline_ids = unique_index(baseline)
    manifest_by_id, duplicate_manifest_ids, missing_manifest_ids = unique_index(manifest)
    shared = []
    for m in manifest:
        cid = m.get("canonical_pair_id", "")
        b = baseline_by_id.get(cid)
        state = choose_state(m)
        row = {
            "canonical_pair_id": cid,
            "target_key": m.get("target_key") or m.get("group_id") or "",
            "state": state,
            "baseline_status": b.get("row_status", "missing") if b else "missing",
            "joint_error": b.get("joint_error", "") if b else "",
            "heading_abs_error": b.get("heading_abs_error", "") if b else "",
            "range_abs_error": b.get("range_abs_error", "") if b else "",
            "heading_rel_error": b.get("heading_rel_error", "") if b else "",
            "range_rel_error": b.get("range_rel_error", "") if b else "",
            "join_ok": "1" if cid in manifest_by_id and b and b.get("row_status") == "ok" else "0",
        }
        shared.append(row)
    issues = {
        "manifest_rows": len(manifest),
        "baseline_rows": len(baseline),
        "duplicate_manifest_ids": duplicate_manifest_ids,
        "duplicate_baseline_ids": duplicate_baseline_ids,
        "missing_manifest_ids": missing_manifest_ids,
        "missing_baseline_ids": missing_baseline_ids,
        "joined_ok_count": sum(1 for row in shared if row["join_ok"] == "1"),
    }
    return shared, issues


def summarize_state(shared: list[dict[str, str]]):
    ok_rows = [row for row in shared if row["join_ok"] == "1"]
    global_errors = [safe_float(row["joint_error"]) for row in ok_rows]
    global_median = quantile(global_errors, 0.5)
    global_p75 = quantile(global_errors, 0.75)
    by_state = defaultdict(list)
    for row in shared:
        by_state[row["state"]].append(row)
    rows = []
    raw_errors_by_state: dict[str, list[float]] = {}
    for state, state_rows in sorted(by_state.items()):
        ok = [row for row in state_rows if row["join_ok"] == "1"]
        joint = [safe_float(row["joint_error"]) for row in ok]
        heading = [safe_float(row["heading_abs_error"]) for row in ok]
        range_error = [safe_float(row["range_abs_error"]) for row in ok]
        rel_heading = [safe_float(row["heading_rel_error"]) for row in ok]
        rel_range = [safe_float(row["range_rel_error"]) for row in ok]
        raw_errors_by_state[state] = [x for x in joint if x is not None]
        difficult_count = sum(1 for value in joint if global_p75 is not None and value is not None and value >= global_p75)
        median_hard_count = sum(1 for value in joint if global_median is not None and value is not None and value >= global_median)
        state_mean = mean(joint)
        rows.append(
            {
                "state": state,
                "count": len(state_rows),
                "joined_ok_count": len(ok),
                "mean_error": fmt(state_mean),
                "median_error": fmt(quantile(joint, 0.5)),
                "p90_error": fmt(quantile(joint, 0.9)),
                "p95_error": fmt(quantile(joint, 0.95)),
                "heading_error_mean": fmt(mean(heading)),
                "range_error_mean": fmt(mean(range_error)),
                "heading_rel_error_mean": fmt(mean(rel_heading)),
                "range_rel_error_mean": fmt(mean(rel_range)),
                "mean_delta_vs_global": fmt(None if state_mean is None or mean(global_errors) is None else state_mean - mean(global_errors)),
                "above_global_median_fraction": fmt(median_hard_count / len(ok) if ok else None),
                "top_quartile_error_fraction": fmt(difficult_count / len(ok) if ok else None),
            }
        )
    return rows, raw_errors_by_state


def pairwise_report(errors_by_state: dict[str, list[float]]):
    states = sorted(errors_by_state)
    rows = []
    for i, a in enumerate(states):
        for b in states[i + 1 :]:
            rows.append(
                {
                    "state_a": a,
                    "state_b": b,
                    "count_a": len(errors_by_state[a]),
                    "count_b": len(errors_by_state[b]),
                    "mean_a": fmt(mean(errors_by_state[a])),
                    "mean_b": fmt(mean(errors_by_state[b])),
                    "mean_a_minus_b": fmt(None if mean(errors_by_state[a]) is None or mean(errors_by_state[b]) is None else mean(errors_by_state[a]) - mean(errors_by_state[b])),
                    "cliff_delta_a_harder_than_b": fmt(cliff_delta(errors_by_state[a], errors_by_state[b])),
                }
            )
    return rows


def target_bias(shared: list[dict[str, str]]):
    by_target = defaultdict(list)
    for row in shared:
        by_target[row["target_key"]].append(row)
    rows = []
    for target, target_rows in sorted(by_target.items()):
        ok = [row for row in target_rows if row["join_ok"] == "1"]
        values = [safe_float(row["joint_error"]) for row in ok]
        rows.append(
            {
                "target_key": target,
                "count": len(target_rows),
                "joined_ok_count": len(ok),
                "mean_error": fmt(mean(values)),
                "state_distribution": json.dumps(dict(Counter(row["state"] for row in target_rows)), sort_keys=True),
            }
        )
    return rows


def decide(metrics: dict[str, Any], state_rows: list[dict[str, Any]], pairwise_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if metrics["duplicate_manifest_ids"] or metrics["duplicate_baseline_ids"] or metrics["missing_manifest_ids"]:
        return "route-v2-outcome-consistency-blocked-identity", "duplicate_or_missing_identity"
    if metrics["joined_fraction"] < 0.95:
        return "route-v2-outcome-consistency-blocked-coverage", "joined_fraction_below_0_95"
    if metrics["state_count"] < 2:
        return "route-v2-outcome-consistency-fail", "state_surface_collapsed"
    gaps = [abs(safe_float(row["mean_a_minus_b"]) or 0.0) for row in pairwise_rows]
    cliffs = [abs(safe_float(row["cliff_delta_a_harder_than_b"]) or 0.0) for row in pairwise_rows]
    if max(gaps or [0.0]) >= 10.0 and max(cliffs or [0.0]) >= 0.10:
        return "route-v2-baseline-outcome-consistency-informative", "state_membership_separates_baseline_error_but_intervention_blocked"
    return "route-v2-baseline-outcome-consistency-weak", "baseline_error_separation_below_informative_threshold"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repeat-metrics", type=Path, required=True)
    parser.add_argument("--identity-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    baseline = read_csv(args.baseline)
    shared, issues = build_shared(manifest, baseline)
    state_rows, errors_by_state = summarize_state(shared)
    pairwise_rows = pairwise_report(errors_by_state)
    target_rows = target_bias(shared)
    repeat = json.loads(args.repeat_metrics.read_text(encoding="utf-8"))
    identity = json.loads(args.identity_metrics.read_text(encoding="utf-8"))

    joined = issues["joined_ok_count"]
    metrics: dict[str, Any] = {
        **issues,
        "joined_fraction": joined / len(manifest) if manifest else 0.0,
        "state_count": len(state_rows),
        "state_distribution": dict(Counter(row["state"] for row in shared)),
        "repeatability_verdict": repeat.get("verdict"),
        "identity_verdict": identity.get("verdict"),
        "intervention_consistency_status": "blocked_no_valid_route_v2_intervention_runner",
        "max_state_mean_gap": max((abs(safe_float(row["mean_a_minus_b"]) or 0.0) for row in pairwise_rows), default=0.0),
        "max_abs_cliff_delta": max((abs(safe_float(row["cliff_delta_a_harder_than_b"]) or 0.0) for row in pairwise_rows), default=0.0),
    }
    verdict, reason = decide(metrics, state_rows, pairwise_rows)
    metrics["verdict"] = verdict
    metrics["reason"] = reason

    out = args.output_dir
    write_csv(out / "tables" / "route_v2_shared_baseline_outcome_surface.csv", shared, [
        "canonical_pair_id",
        "target_key",
        "state",
        "baseline_status",
        "joint_error",
        "heading_abs_error",
        "range_abs_error",
        "heading_rel_error",
        "range_rel_error",
        "join_ok",
    ])
    write_csv(out / "tables" / "route_v2_state_wise_outcome_consistency.csv", state_rows, [
        "state",
        "count",
        "joined_ok_count",
        "mean_error",
        "median_error",
        "p90_error",
        "p95_error",
        "heading_error_mean",
        "range_error_mean",
        "heading_rel_error_mean",
        "range_rel_error_mean",
        "mean_delta_vs_global",
        "above_global_median_fraction",
        "top_quartile_error_fraction",
    ])
    write_csv(out / "tables" / "route_v2_pairwise_state_separation.csv", pairwise_rows, [
        "state_a",
        "state_b",
        "count_a",
        "count_b",
        "mean_a",
        "mean_b",
        "mean_a_minus_b",
        "cliff_delta_a_harder_than_b",
    ])
    write_csv(out / "tables" / "route_v2_target_bias_report.csv", target_rows, [
        "target_key",
        "count",
        "joined_ok_count",
        "mean_error",
        "state_distribution",
    ])
    write_json(out / "metrics" / "route_v2_outcome_consistency_metrics.json", metrics)

    lines = [
        "# A-v3.2c Route-v2 Outcome-Consistency Audit",
        "",
        f"verdict: `{verdict}`",
        f"reason: `{reason}`",
        "",
        f"- manifest_rows: `{len(manifest)}`",
        f"- joined_ok_count: `{joined}`",
        f"- joined_fraction: `{metrics['joined_fraction']:.6f}`",
        f"- state_count: `{metrics['state_count']}`",
        f"- repeatability_verdict: `{metrics['repeatability_verdict']}`",
        f"- identity_verdict: `{metrics['identity_verdict']}`",
        f"- intervention_consistency_status: `{metrics['intervention_consistency_status']}`",
        f"- max_state_mean_gap: `{metrics['max_state_mean_gap']:.12g}`",
        f"- max_abs_cliff_delta: `{metrics['max_abs_cliff_delta']:.12g}`",
        "",
        "## State Summary",
    ]
    for row in state_rows:
        lines.append(
            f"- {row['state']}: count={row['count']}, mean={row['mean_error']}, "
            f"median={row['median_error']}, heading={row['heading_error_mean']}, range={row['range_error_mean']}, "
            f"top_quartile_fraction={row['top_quartile_error_fraction']}"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a repaired route-v2 baseline outcome audit. It can support claims about whether A states separate baseline difficulty. It cannot support intervention-response claims because no valid route-v2 stress/intervention runner has been established in this branch.",
        "",
        "No training, finetuning, sample weighting, threshold tuning, B/C gate, full eval, submission packaging, fuzzy join, silent deduplication, or leaderboard probing was run.",
    ])
    (out / "reports" / "route_v2_outcome_consistency_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "reports" / "route_v2_outcome_consistency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
