"""Compute Phase27 A taxonomy redesign-v2 manifest metrics.

Stdlib-only implementation for remote Python environments without pandas/numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


VERDICT_LABELS = {
    "taxonomy-redesign-v2-ready-for-knowledge-review",
    "taxonomy-redesign-v2-needs-redesign",
    "taxonomy-redesign-v2-blocked-by-coverage",
    "taxonomy-redesign-v2-smoke-only",
}
SCORE_COLUMNS = [
    "baseline_error_score",
    "heading_error_score",
    "range_error_score",
    "stress_sensitivity_score",
    "checkpoint_disagreement_score",
]


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_table(out_dir: Path, stem: str, rows: list[dict]) -> None:
    _write_csv(out_dir / f"{stem}.csv", rows)
    _write_json(out_dir / f"{stem}.json", rows)


def _float(value, default=0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _prepare(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        out = dict(row)
        out.setdefault("derived_state", "unknown")
        out.setdefault("old_base_regime", "unknown")
        for column in SCORE_COLUMNS:
            out[column] = max(0.0, min(1.0, _float(out.get(column))))
        out["tail_outlier_flag"] = _bool(out.get("tail_outlier_flag"))
        out["composite_error_score"] = statistics.fmean(out[column] for column in SCORE_COLUMNS)
        prepared.append(out)
    return prepared


def _state_counts(rows: list[dict]) -> list[dict]:
    total = len(rows)
    counts = {}
    for row in rows:
        counts[row["derived_state"]] = counts.get(row["derived_state"], 0) + 1
    return [
        {"derived_state": state, "count": count, "fraction": count / total if total else 0.0}
        for state, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _old_vs_new(rows: list[dict]) -> list[dict]:
    counts = {}
    totals = {}
    for row in rows:
        old = row["old_base_regime"]
        state = row["derived_state"]
        counts[(old, state)] = counts.get((old, state), 0) + 1
        totals[old] = totals.get(old, 0) + 1
    return [
        {
            "old_base_regime": old,
            "derived_state": state,
            "count": count,
            "fraction_within_old_regime": count / totals[old] if totals.get(old) else 0.0,
        }
        for (old, state), count in sorted(counts.items())
    ]


def _taxonomy_comparison_summary(rows: list[dict]) -> dict:
    old_hard = [row for row in rows if row["old_base_regime"] == "hard_trainable"]
    old_control = [row for row in rows if row["old_base_regime"] == "ordinary_control_anchor"]
    return {
        "old_hard_trainable_count": len(old_hard),
        "old_ordinary_control_anchor_count": len(old_control),
        "old_hard_trainable_mean_composite_score": statistics.fmean(row["composite_error_score"] for row in old_hard) if old_hard else 0.0,
        "old_ordinary_control_anchor_mean_composite_score": statistics.fmean(row["composite_error_score"] for row in old_control) if old_control else 0.0,
        "new_state_hard_count": sum(1 for row in rows if row["derived_state"] == "evidence_sufficient_hard"),
        "new_state_control_count": sum(1 for row in rows if row["derived_state"] == "stable_control_anchor"),
        "new_state_hard_control_delta": _hard_minus_control_delta(rows),
    }


def _per_state(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["derived_state"], []).append(row)
    output = []
    for state in sorted(grouped):
        group = grouped[state]
        scores = [row["composite_error_score"] for row in group]
        output.append(
            {
                "derived_state": state,
                "count": len(group),
                "mean_composite_score": statistics.fmean(scores) if scores else 0.0,
                "median_composite_score": statistics.median(scores) if scores else 0.0,
                "old_hard_trainable_fraction": sum(1 for row in group if row["old_base_regime"] == "hard_trainable") / len(group),
                "old_ordinary_control_anchor_fraction": sum(1 for row in group if row["old_base_regime"] == "ordinary_control_anchor") / len(group),
                "tail_outlier_rate": sum(1 for row in group if row["tail_outlier_flag"]) / len(group),
            }
        )
    return output


def _target_scene_bias(rows: list[dict]) -> list[dict]:
    counts = {}
    totals = {}
    for row in rows:
        target = row.get("target_key") or row.get("target_label") or row.get("target") or "unknown"
        scene = row.get("scene_key") or row.get("scene") or row.get("scene_id") or "unknown"
        key = (target, scene, row["derived_state"])
        counts[key] = counts.get(key, 0) + 1
        totals[(target, scene)] = totals.get((target, scene), 0) + 1
    return [
        {
            "target_key": target,
            "scene_key": scene,
            "derived_state": state,
            "count": count,
            "fraction_within_target_scene": count / totals[(target, scene)],
        }
        for (target, scene, state), count in sorted(counts.items())
    ]


def _hard_minus_control_delta(rows: list[dict]) -> float:
    hard = [row["composite_error_score"] for row in rows if row["derived_state"] == "evidence_sufficient_hard"]
    control = [row["composite_error_score"] for row in rows if row["derived_state"] == "stable_control_anchor"]
    if not hard or not control:
        return 0.0
    return statistics.fmean(hard) - statistics.fmean(control)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[idx]


def _bootstrap_ci(rows: list[dict], iters: int, seed: int) -> dict:
    rng = random.Random(seed)
    samples = []
    if rows and iters > 0:
        for _ in range(iters):
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            samples.append(_hard_minus_control_delta(sample))
    else:
        samples = [0.0]
    return {
        "bootstrap_iters": max(0, iters),
        "seed": seed,
        "new_state_hard_minus_control_delta": _hard_minus_control_delta(rows),
        "ci_low": _quantile(samples, 0.025),
        "ci_high": _quantile(samples, 0.975),
    }


def _join_coverage(rows: list[dict]) -> dict:
    total = len(rows)
    payload = {"total_rows": total}
    for column in ["full_dev_joined", "stress_joined"]:
        count = sum(1 for row in rows if _bool(row.get(column)))
        payload[f"{column}_count"] = count
        payload[f"{column}_fraction"] = count / total if total else 0.0
    return payload


def _tail_risk(rows: list[dict]) -> dict:
    states = {}
    tail_rows = [row for row in rows if row["tail_outlier_flag"]]
    for row in tail_rows:
        states[row["derived_state"]] = states.get(row["derived_state"], 0) + 1
    return {
        "total_rows": len(rows),
        "tail_outlier_count": len(tail_rows),
        "tail_outlier_rate": len(tail_rows) / len(rows) if rows else 0.0,
        "tail_state_counts": states,
    }


def _final_verdict(rows: list[dict], coverage: dict, tail: dict) -> dict:
    min_coverage = min(coverage.get("full_dev_joined_fraction", 0.0), coverage.get("stress_joined_fraction", 0.0))
    joined_any = max(coverage.get("full_dev_joined_count", 0), coverage.get("stress_joined_count", 0))
    hard_count = sum(1 for row in rows if row["derived_state"] == "evidence_sufficient_hard")
    control_count = sum(1 for row in rows if row["derived_state"] == "stable_control_anchor")
    if len(rows) < 10:
        label = "taxonomy-redesign-v2-smoke-only"
    elif joined_any == 0:
        label = "taxonomy-redesign-v2-blocked-by-coverage"
    elif hard_count == 0 or control_count == 0:
        label = "taxonomy-redesign-v2-needs-redesign"
    elif tail.get("tail_outlier_rate", 0.0) > 0.25:
        label = "taxonomy-redesign-v2-needs-redesign"
    else:
        label = "taxonomy-redesign-v2-ready-for-knowledge-review"
    if label not in VERDICT_LABELS:
        raise AssertionError("invalid verdict label")
    return {
        "label": label,
        "rows": len(rows),
        "min_join_coverage": min_coverage,
        "joined_any_count": joined_any,
        "new_state_hard_count": hard_count,
        "new_state_control_count": control_count,
        "tail_outlier_rate": tail.get("tail_outlier_rate", 0.0),
        "forbidden_training_policy_spec_emitted": False,
    }


def compute_metrics(manifest: str | Path, out_dir: str | Path, bootstrap_iters: int, seed: int) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _prepare(_read_csv(manifest))

    state_counts = _state_counts(rows)
    old_vs_new = _old_vs_new(rows)
    old_vs_new_summary = _taxonomy_comparison_summary(rows)
    per_state = _per_state(rows)
    bias = _target_scene_bias(rows)
    ci = _bootstrap_ci(rows, bootstrap_iters, seed)
    coverage = _join_coverage(rows)
    tail = _tail_risk(rows)
    verdict = _final_verdict(rows, coverage, tail)

    _write_table(out_dir, "state_counts", state_counts)
    _write_table(out_dir, "old_vs_new_taxonomy_comparison", old_vs_new)
    _write_json(out_dir / "old_vs_new_taxonomy_comparison.json", old_vs_new_summary)
    _write_table(out_dir, "per_state_surface_metrics", per_state)
    _write_json(out_dir / "bootstrap_ci.json", ci)
    _write_table(out_dir, "target_scene_bias", bias)
    _write_json(out_dir / "join_coverage.json", coverage)
    _write_json(out_dir / "tail_risk_metrics.json", tail)
    _write_json(out_dir / "final_verdict.json", verdict)

    return {
        "state_counts": state_counts,
        "old_vs_new_taxonomy_comparison": old_vs_new,
        "old_vs_new_taxonomy_comparison_summary": old_vs_new_summary,
        "per_state_surface_metrics": per_state,
        "bootstrap_ci": ci,
        "target_scene_bias": bias,
        "join_coverage": coverage,
        "tail_risk_metrics": tail,
        "final_verdict": verdict,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    compute_metrics(args.manifest, args.out_dir, args.bootstrap_iters, args.seed)


if __name__ == "__main__":
    main()
