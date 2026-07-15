#!/usr/bin/env python3
"""Validate Phase27 A v3 feature calibration outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_REGIMES = [
    "ordinary_control_anchor",
    "high_evidence_anchor",
    "hard_trainable",
    "low_observable",
    "ambiguous_unreliable",
    "unknown_insufficient_features",
]
AXIS_NAMES = [
    "observability_axis",
    "pair_similarity_axis",
    "scale_compatibility_axis",
    "layout_risk_axis",
]
RISK_TAG_COLUMN = "risk_tags"
FORBIDDEN_PATTERNS = [
    "heading_num",
    "range_num",
    "gt_angle",
    "gt_distance",
    "final_score",
    "angle_err",
    "range_err",
    "combined_error",
    "residual",
    "official",
    "leaderboard",
    "phase11",
    "phase13",
    "phase14",
    "slice_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--axes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-v3-manifest", required=True)
    parser.add_argument("--raw-v3-metrics", required=True)
    parser.add_argument("--v2-manifest", required=True)
    parser.add_argument("--v2-metrics", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--metrics-csv", required=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def fractions(counts: dict[str, int], total: int) -> dict[str, float]:
    return {key: (value / total if total else 0.0) for key, value in counts.items()}


def count_base_regimes(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {regime: 0 for regime in BASE_REGIMES}
    for row in rows:
        regime = row.get("base_regime", "")
        if regime in counts:
            counts[regime] += 1
    return counts


def exactly_one_base(row: dict[str, str]) -> bool:
    total = 0
    for regime in BASE_REGIMES:
        try:
            total += int(str(row.get(f"base_{regime}", "0")).strip() or "0")
        except ValueError:
            return False
    return total == 1


def leakage_audit(columns: list[str]) -> dict[str, Any]:
    forbidden = [
        column
        for column in columns
        if any(pattern in column.lower() for pattern in FORBIDDEN_PATTERNS)
    ]
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "column_count": len(columns),
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def axis_quantiles(rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for axis in AXIS_NAMES:
        values = [value for value in (as_float(row.get(axis)) for row in rows) if value is not None]
        out[axis] = {
            "q00": quantile(values, 0.0),
            "q25": quantile(values, 0.25),
            "q50": quantile(values, 0.50),
            "q75": quantile(values, 0.75),
            "q100": quantile(values, 1.0),
        }
    return out


def band_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for axis in AXIS_NAMES:
        column = f"{axis}_band"
        out[axis] = dict(Counter(row.get(column, "missing") for row in rows))
    return out


def risk_tag_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in str(row.get(RISK_TAG_COLUMN, "")).split("|"):
            tag = tag.strip()
            if tag:
                counts[tag] += 1
    return dict(counts)


def by_split(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("source_split", "unknown") or "unknown"].append(row)
    return dict(groups)


def split_base_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, split_rows in by_split(rows).items():
        counts = count_base_regimes(split_rows)
        out[split] = {
            "row_count": len(split_rows),
            "base_regime_counts": counts,
            "base_regime_fractions": fractions(counts, len(split_rows)),
            "axis_band_counts": band_counts(split_rows),
        }
    return out


def max_split_fraction_shift(split_metrics: dict[str, Any]) -> float:
    if "train" not in split_metrics or "dev" not in split_metrics:
        return 0.0
    train = split_metrics["train"]["base_regime_fractions"]
    dev = split_metrics["dev"]["base_regime_fractions"]
    return max(abs(train.get(regime, 0.0) - dev.get(regime, 0.0)) for regime in BASE_REGIMES)


def raw_v3_comparison(raw_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]], raw_metrics: dict[str, Any]) -> dict[str, Any]:
    raw_counts = count_base_regimes(raw_rows)
    cal_counts = count_base_regimes(calibrated_rows)
    raw_total = len(raw_rows)
    cal_total = len(calibrated_rows)
    raw_max = max(fractions(raw_counts, raw_total).values()) if raw_total else 0.0
    cal_max = max(fractions(cal_counts, cal_total).values()) if cal_total else 0.0
    return {
        "raw_high_evidence_anchor_count": raw_counts.get("high_evidence_anchor", 0),
        "raw_high_evidence_anchor_fraction": raw_counts.get("high_evidence_anchor", 0) / raw_total if raw_total else 0.0,
        "calibrated_high_evidence_anchor_count": cal_counts.get("high_evidence_anchor", 0),
        "calibrated_high_evidence_anchor_fraction": cal_counts.get("high_evidence_anchor", 0) / cal_total if cal_total else 0.0,
        "raw_ordinary_control_anchor_count": raw_counts.get("ordinary_control_anchor", 0),
        "raw_ordinary_control_anchor_fraction": raw_counts.get("ordinary_control_anchor", 0) / raw_total if raw_total else 0.0,
        "calibrated_ordinary_control_anchor_count": cal_counts.get("ordinary_control_anchor", 0),
        "calibrated_ordinary_control_anchor_fraction": cal_counts.get("ordinary_control_anchor", 0) / cal_total if cal_total else 0.0,
        "raw_max_base_regime_fraction": raw_metrics.get("max_base_regime_fraction", raw_max),
        "calibrated_max_base_regime_fraction": cal_max,
        "collapse_improved": cal_max < float(raw_metrics.get("max_base_regime_fraction", raw_max)),
        "raw_reference_numbers": {
            "high_evidence_anchor": "57122/60000",
            "ordinary_control_anchor": "21/60000",
            "max_base_regime_fraction": "0.9520",
        },
    }


def v2_overlap_comparison(v2_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]]) -> dict[str, Any]:
    cal_by_pair = {row.get("pair_id", ""): row for row in calibrated_rows if row.get("pair_id")}
    overlap = [row for row in v2_rows if row.get("pair_id") in cal_by_pair]
    v2_ordinary = [row for row in overlap if as_bool(row.get("base_ordinary_control_anchor")) or row.get("base_regime") == "ordinary_control_anchor"]
    calibrated_ordinary_on_v2_ordinary = [
        cal_by_pair[row["pair_id"]]
        for row in v2_ordinary
        if cal_by_pair[row["pair_id"]].get("base_regime") == "ordinary_control_anchor"
    ]
    calibrated_ordinary_on_overlap = [
        row for row in overlap if cal_by_pair[row["pair_id"]].get("base_regime") == "ordinary_control_anchor"
    ]
    lost = len(v2_ordinary) - len(calibrated_ordinary_on_v2_ordinary)
    status = "reliable" if overlap else "unreliable"
    return {
        "status": status,
        "overlap_count": len(overlap),
        "v2_ordinary_control_count": len(v2_ordinary),
        "calibrated_ordinary_on_v2_ordinary_count": len(calibrated_ordinary_on_v2_ordinary),
        "calibrated_ordinary_on_overlap_count": len(calibrated_ordinary_on_overlap),
        "v2_ordinary_control_lost_count": lost,
        "regressed_below_v2": status != "reliable" or len(calibrated_ordinary_on_v2_ordinary) < len(v2_ordinary),
    }


def write_flat_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    rows.append(("row_count", metrics["row_count"]))
    rows.append(("verdict", metrics["verdict"]))
    for regime, count in metrics["base_regime_counts"].items():
        rows.append((f"base_count.{regime}", count))
    for regime, frac in metrics["base_regime_fractions"].items():
        rows.append((f"base_fraction.{regime}", frac))
    for key, value in metrics["raw_v3_comparison"].items():
        if not isinstance(value, dict):
            rows.append((f"raw_v3_comparison.{key}", value))
    for key, value in metrics["v2_overlap_comparison"].items():
        rows.append((f"v2_overlap_comparison.{key}", value))
    for reason in metrics["verdict_reasons"]:
        rows.append(("verdict_reason", reason))
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    axes_rows, axes_columns = read_csv(args.axes)
    manifest_rows, manifest_columns = read_csv(args.manifest)
    raw_rows, _ = read_csv(args.raw_v3_manifest)
    v2_rows, _ = read_csv(args.v2_manifest)
    raw_metrics = read_json(args.raw_v3_metrics)
    v2_metrics = read_json(args.v2_metrics)

    row_count = len(manifest_rows)
    base_counts = count_base_regimes(manifest_rows)
    base_fracs = fractions(base_counts, row_count)
    max_base_fraction = max(base_fracs.values()) if base_fracs else 0.0
    leakage = leakage_audit(axes_columns + manifest_columns)
    exactly_one = all(exactly_one_base(row) for row in manifest_rows)
    missing_required_axes = {
        axis: sum(1 for row in manifest_rows if row.get(axis) in {"", None} or row.get(f"{axis}_band") in {"", None})
        for axis in AXIS_NAMES
    }
    missing_base_regime = sum(1 for row in manifest_rows if not row.get("base_regime"))
    unknown_fraction = base_fracs.get("unknown_insufficient_features", 0.0)
    ordinary_fraction = base_fracs.get("ordinary_control_anchor", 0.0)
    high_fraction = base_fracs.get("high_evidence_anchor", 0.0)
    split_metrics = split_base_metrics(manifest_rows)
    max_shift = max_split_fraction_shift(split_metrics)
    raw_compare = raw_v3_comparison(raw_rows, manifest_rows, raw_metrics)
    v2_compare = v2_overlap_comparison(v2_rows, manifest_rows)

    reasons: list[str] = []
    if not leakage["passed"]:
        reasons.append("leakage audit failed")
    if row_count < 60000:
        reasons.append(f"coverage below 60000: {row_count}")
    if not exactly_one:
        reasons.append("exactly-one-base-regime failed")
    if unknown_fraction >= 0.05:
        reasons.append(f"unknown_insufficient_features >= 5%: {unknown_fraction:.4f}")
    if ordinary_fraction < 0.15:
        reasons.append(f"ordinary_control_anchor below 15%: {ordinary_fraction:.4f}")
    if high_fraction >= 0.70:
        reasons.append(f"high_evidence_anchor >= 70%: {high_fraction:.4f}")
    if max_base_fraction > 0.85:
        reasons.append(f"base regime exceeds 85%: {max_base_fraction:.4f}")
    if v2_compare["regressed_below_v2"]:
        reasons.append("v2-overlap ordinary/control anchor coverage regressed below v2")
    if max_shift > 0.30:
        reasons.append(f"train/dev base-regime fraction shift above 0.30: {max_shift:.4f}")
    if not raw_compare["collapse_improved"]:
        reasons.append("calibrated mapping did not improve raw v3 collapse metric")

    if not reasons:
        verdict = "calibration-ready-for-knowledge-review"
    elif (
        not leakage["passed"]
        or ordinary_fraction < 0.15
        or v2_compare["regressed_below_v2"]
        or max_base_fraction > 0.85
    ):
        verdict = "calibration-rejected"
    else:
        verdict = "calibration-weak-inconclusive"

    metrics = {
        "schema_version": "phase27_a_feature_calibration_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": args.project_root,
        "axes": args.axes,
        "manifest": args.manifest,
        "raw_v3_manifest": args.raw_v3_manifest,
        "raw_v3_metrics": args.raw_v3_metrics,
        "v2_manifest": args.v2_manifest,
        "v2_metrics": args.v2_metrics,
        "row_count": row_count,
        "axes_row_count": len(axes_rows),
        "leakage_audit": leakage,
        "exactly_one_base_regime_passed": exactly_one,
        "missing_base_regime_count": missing_base_regime,
        "missing_required_axes": missing_required_axes,
        "base_regime_counts": base_counts,
        "base_regime_fractions": base_fracs,
        "risk_tag_counts": risk_tag_counts(manifest_rows),
        "risk_tag_fractions": fractions(risk_tag_counts(manifest_rows), row_count),
        "axis_quantiles": axis_quantiles(axes_rows),
        "axis_band_counts": band_counts(manifest_rows),
        "split_metrics": split_metrics,
        "max_train_dev_base_fraction_shift": max_shift,
        "raw_v3_comparison": raw_compare,
        "v2_overlap_comparison": v2_compare,
        "v2_reference_verdict": v2_metrics.get("verdict"),
        "unknown_insufficient_features_fraction": unknown_fraction,
        "ordinary_control_anchor_fraction": ordinary_fraction,
        "high_evidence_anchor_fraction": high_fraction,
        "max_base_regime_fraction": max_base_fraction,
        "training_started": False,
        "submission_created": False,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }

    metrics_json = Path(args.metrics_json)
    ensure_parent(metrics_json)
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_flat_csv(Path(args.metrics_csv), metrics)
    print(json.dumps({"verdict": verdict, "row_count": row_count, "reasons": reasons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
