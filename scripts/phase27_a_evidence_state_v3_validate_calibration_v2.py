#!/usr/bin/env python3
"""Validate Phase27 A evidence-state calibration-v2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_v3_calibration_v2 import (
    BASE_REGIMES,
    FORBIDDEN_CONSTRUCTION_PATTERNS,
    canonical_pair_id,
)


AXIS_NAMES = [
    "observability_axis",
    "pair_similarity_axis",
    "scale_risk_axis",
    "layout_risk_axis",
    "conflict_risk_axis",
    "control_centrality_score",
]
ADEQUACY_FIELDS = [
    "feature_complete",
    "observable_adequate",
    "image_quality_adequate",
    "pair_identity_valid",
    "adequacy_passed",
    "low_observable_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--axes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-v3-manifest", required=True)
    parser.add_argument("--raw-v3-metrics", required=True)
    parser.add_argument("--v1-manifest", required=True)
    parser.add_argument("--v1-metrics", required=True)
    parser.add_argument("--v2-reference-manifest", required=True)
    parser.add_argument("--v2-reference-metrics", required=True)
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
        if any(pattern in column.lower() for pattern in FORBIDDEN_CONSTRUCTION_PATTERNS)
    ]
    return {"passed": not forbidden, "forbidden_columns": forbidden, "column_count": len(columns)}


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


def quantile_summary(rows: list[dict[str, str]], column: str) -> dict[str, float | None]:
    values = [value for value in (as_float(row.get(column)) for row in rows) if value is not None]
    return {
        "q00": quantile(values, 0.0),
        "q25": quantile(values, 0.25),
        "q50": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "q100": quantile(values, 1.0),
    }


def split_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("source_split", "unknown") or "unknown"].append(row)
    out: dict[str, Any] = {}
    for split, split_rows in groups.items():
        counts = count_base_regimes(split_rows)
        out[split] = {
            "row_count": len(split_rows),
            "base_regime_counts": counts,
            "base_regime_fractions": fractions(counts, len(split_rows)),
        }
    return out


def max_train_dev_shift(metrics: dict[str, Any]) -> float:
    if "train" not in metrics or "dev" not in metrics:
        return 0.0
    train = metrics["train"]["base_regime_fractions"]
    dev = metrics["dev"]["base_regime_fractions"]
    return max(abs(train.get(regime, 0.0) - dev.get(regime, 0.0)) for regime in BASE_REGIMES)


def low_observable_reason_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("low_observable_reason", "") or "").split("|"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return dict(counter)


def centrality_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    out = {"all": quantile_summary(rows, "control_centrality_score")}
    for regime in BASE_REGIMES:
        regime_rows = [row for row in rows if row.get("base_regime") == regime]
        out[regime] = quantile_summary(regime_rows, "control_centrality_score")
    return out


def adequacy_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if as_bool(row.get(field)))
        for field in ADEQUACY_FIELDS
        if field != "low_observable_reason"
    }


def canonicalize_reference_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = canonical_pair_id(row)
        if key:
            out[key] = row
    return out


def v1_vs_v2(v1_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> dict[str, Any]:
    v1_counts = count_base_regimes(v1_rows)
    v2_counts = count_base_regimes(v2_rows)
    v1_total = len(v1_rows)
    v2_total = len(v2_rows)
    v1_fracs = fractions(v1_counts, v1_total)
    v2_fracs = fractions(v2_counts, v2_total)
    return {
        "v1_base_regime_counts": v1_counts,
        "v2_base_regime_counts": v2_counts,
        "v1_base_regime_fractions": v1_fracs,
        "v2_base_regime_fractions": v2_fracs,
        "v1_ordinary_control_anchor_fraction": v1_fracs.get("ordinary_control_anchor", 0.0),
        "v2_ordinary_control_anchor_fraction": v2_fracs.get("ordinary_control_anchor", 0.0),
        "v1_high_evidence_anchor_fraction": v1_fracs.get("high_evidence_anchor", 0.0),
        "v2_high_evidence_anchor_fraction": v2_fracs.get("high_evidence_anchor", 0.0),
        "v1_max_base_regime_fraction": max(v1_fracs.values()) if v1_fracs else 0.0,
        "v2_max_base_regime_fraction": max(v2_fracs.values()) if v2_fracs else 0.0,
    }


def raw_v3_comparison(raw_rows: list[dict[str, str]], v2_rows: list[dict[str, str]], raw_metrics: dict[str, Any]) -> dict[str, Any]:
    raw_counts = count_base_regimes(raw_rows)
    v2_counts = count_base_regimes(v2_rows)
    raw_fracs = fractions(raw_counts, len(raw_rows))
    v2_fracs = fractions(v2_counts, len(v2_rows))
    raw_max = float(raw_metrics.get("max_base_regime_fraction", max(raw_fracs.values()) if raw_fracs else 0.0))
    v2_max = max(v2_fracs.values()) if v2_fracs else 0.0
    return {
        "raw_base_regime_counts": raw_counts,
        "v2_base_regime_counts": v2_counts,
        "raw_high_evidence_anchor_fraction": raw_fracs.get("high_evidence_anchor", 0.0),
        "v2_high_evidence_anchor_fraction": v2_fracs.get("high_evidence_anchor", 0.0),
        "raw_max_base_regime_fraction": raw_max,
        "v2_max_base_regime_fraction": v2_max,
        "collapse_remains_fixed": v2_max < 0.85 and v2_max < raw_max,
    }


def canonical_overlap(reference_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> dict[str, Any]:
    ref_by_key = canonicalize_reference_rows(reference_rows)
    v2_by_key = {row.get("canonical_pair_id", ""): row for row in v2_rows if row.get("canonical_pair_id")}
    overlap_keys = sorted(set(ref_by_key) & set(v2_by_key))
    ref_ordinary_keys = [
        key for key in overlap_keys
        if ref_by_key[key].get("base_regime") == "ordinary_control_anchor"
        or as_bool(ref_by_key[key].get("base_ordinary_control_anchor"))
    ]
    v2_ordinary_on_ref_ordinary = [
        key for key in ref_ordinary_keys
        if v2_by_key[key].get("base_regime") == "ordinary_control_anchor"
    ]
    v2_ordinary_on_overlap = [
        key for key in overlap_keys
        if v2_by_key[key].get("base_regime") == "ordinary_control_anchor"
    ]
    status = "reliable" if overlap_keys else "unreliable"
    regressed = status == "reliable" and len(v2_ordinary_on_ref_ordinary) < len(ref_ordinary_keys)
    return {
        "status": status,
        "reference_key_count": len(ref_by_key),
        "v2_key_count": len(v2_by_key),
        "overlap_count": len(overlap_keys),
        "reference_ordinary_control_count": len(ref_ordinary_keys),
        "v2_ordinary_on_reference_ordinary_count": len(v2_ordinary_on_ref_ordinary),
        "v2_ordinary_on_overlap_count": len(v2_ordinary_on_overlap),
        "reference_ordinary_lost_count": len(ref_ordinary_keys) - len(v2_ordinary_on_ref_ordinary),
        "regressed_below_reference": regressed,
    }


def quota_only_suspected(rows: list[dict[str, str]], ordinary_fraction: float) -> bool:
    if ordinary_fraction < 0.15:
        return False
    ordinary_rows = [row for row in rows if row.get("base_regime") == "ordinary_control_anchor"]
    if not ordinary_rows:
        return True
    eligible = []
    for row in ordinary_rows:
        scale = as_float(row.get("scale_risk_axis"))
        layout = as_float(row.get("layout_risk_axis"))
        conflict = as_float(row.get("conflict_risk_axis"))
        centrality = as_float(row.get("control_centrality_score"))
        if (
            as_bool(row.get("adequacy_passed"))
            and scale is not None and scale <= 0.30
            and layout is not None and layout <= 0.35
            and conflict is not None and conflict <= 0.25
            and centrality is not None and centrality >= 0.82
        ):
            eligible.append(row)
    return len(eligible) != len(ordinary_rows)


def write_flat_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows: list[tuple[str, Any]] = [
        ("row_count", metrics["row_count"]),
        ("verdict", metrics["verdict"]),
        ("ordinary_control_anchor_fraction", metrics["ordinary_control_anchor_fraction"]),
        ("high_evidence_anchor_fraction", metrics["high_evidence_anchor_fraction"]),
        ("max_base_regime_fraction", metrics["max_base_regime_fraction"]),
        ("quota_only_suspected", metrics["quota_only_suspected"]),
    ]
    for regime, count in metrics["base_regime_counts"].items():
        rows.append((f"base_count.{regime}", count))
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
    v1_rows, _ = read_csv(args.v1_manifest)
    reference_rows, _ = read_csv(args.v2_reference_manifest)
    raw_metrics = read_json(args.raw_v3_metrics)
    v1_metrics = read_json(args.v1_metrics)
    reference_metrics = read_json(args.v2_reference_metrics)

    row_count = len(manifest_rows)
    base_counts = count_base_regimes(manifest_rows)
    base_fracs = fractions(base_counts, row_count)
    max_base = max(base_fracs.values()) if base_fracs else 0.0
    leakage = leakage_audit(axes_columns + manifest_columns)
    exactly_one = all(exactly_one_base(row) for row in manifest_rows)
    missing_canonical = sum(1 for row in manifest_rows if not row.get("canonical_pair_id"))
    missing_adequacy = {
        field: sum(1 for row in manifest_rows if field not in row or row.get(field) == "")
        for field in ADEQUACY_FIELDS
    }
    split = split_metrics(manifest_rows)
    max_shift = max_train_dev_shift(split)
    v1_compare = v1_vs_v2(v1_rows, manifest_rows)
    raw_compare = raw_v3_comparison(raw_rows, manifest_rows, raw_metrics)
    overlap = canonical_overlap(reference_rows, manifest_rows)
    ordinary_fraction = base_fracs.get("ordinary_control_anchor", 0.0)
    high_fraction = base_fracs.get("high_evidence_anchor", 0.0)
    unknown_fraction = base_fracs.get("unknown_insufficient_features", 0.0)
    quota_suspected = quota_only_suspected(manifest_rows, ordinary_fraction)
    low_reasons = low_observable_reason_counts(manifest_rows)

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
    if max_base > 0.85:
        reasons.append(f"base regime exceeds 85%: {max_base:.4f}")
    if not raw_compare["collapse_remains_fixed"]:
        reasons.append("raw v3 collapse not fixed")
    if overlap["status"] != "reliable":
        reasons.append("canonical v2-overlap status unreliable")
    if overlap["regressed_below_reference"]:
        reasons.append("v2-overlap ordinary/control anchor coverage regressed below v2")
    if max_shift > 0.30:
        reasons.append(f"train/dev base-regime fraction shift above 0.30: {max_shift:.4f}")
    if quota_suspected:
        reasons.append("quota-only recovery suspected")

    if not reasons:
        verdict = "calibration-v2-ready-for-knowledge-review"
    elif (
        not leakage["passed"]
        or ordinary_fraction < 0.15
        or not raw_compare["collapse_remains_fixed"]
        or overlap["status"] != "reliable"
        or quota_suspected
    ):
        verdict = "calibration-v2-rejected"
    else:
        verdict = "calibration-v2-weak-inconclusive"

    metrics = {
        "schema_version": "phase27_a_feature_calibration_v2_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": args.project_root,
        "axes": args.axes,
        "manifest": args.manifest,
        "raw_v3_manifest": args.raw_v3_manifest,
        "raw_v3_metrics": args.raw_v3_metrics,
        "v1_manifest": args.v1_manifest,
        "v1_metrics": args.v1_metrics,
        "v2_reference_manifest": args.v2_reference_manifest,
        "v2_reference_metrics": args.v2_reference_metrics,
        "v1_reference_verdict": v1_metrics.get("verdict"),
        "v2_reference_verdict": reference_metrics.get("verdict"),
        "row_count": row_count,
        "axes_row_count": len(axes_rows),
        "leakage_audit": leakage,
        "exactly_one_base_regime_passed": exactly_one,
        "missing_canonical_pair_id_count": missing_canonical,
        "missing_adequacy_fields": missing_adequacy,
        "adequacy_counts": adequacy_counts(manifest_rows),
        "adequacy_fractions": fractions(adequacy_counts(manifest_rows), row_count),
        "low_observable_reason_counts": low_reasons,
        "base_regime_counts": base_counts,
        "base_regime_fractions": base_fracs,
        "ordinary_control_anchor_fraction": ordinary_fraction,
        "high_evidence_anchor_fraction": high_fraction,
        "unknown_insufficient_features_fraction": unknown_fraction,
        "max_base_regime_fraction": max_base,
        "centrality_quantiles": centrality_diagnostics(manifest_rows),
        "split_metrics": split,
        "max_train_dev_base_fraction_shift": max_shift,
        "v1_vs_v2_comparison": v1_compare,
        "raw_v3_comparison": raw_compare,
        "canonical_v2_overlap_comparison": overlap,
        "quota_only_suspected": quota_suspected,
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
