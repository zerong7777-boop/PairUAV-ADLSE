#!/usr/bin/env python3
"""Validate Phase27 v3 coverage manifest."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_schema_v2 import BASE_FLAG_COLUMNS, BASE_REGIMES, RISK_TAGS
from scripts.phase27_a_evidence_state_v3_feature_schema import audit_feature_columns_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--v2-metrics", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_one(value: str | None) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _count(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if _is_one(row.get(column)))


def _base_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {regime: sum(1 for row in rows if row.get("base_regime") == regime) for regime in BASE_REGIMES}


def _tag_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {tag: _count(rows, tag) for tag in RISK_TAGS}


def _invariant_violations(rows: list[dict[str, str]]) -> list[str]:
    violations: list[str] = []
    for idx, row in enumerate(rows):
        active = [col for col in BASE_FLAG_COLUMNS if _is_one(row.get(col))]
        if len(active) != 1:
            violations.append(f"row {idx} active_base_count={len(active)}")
        elif row.get("base_regime") != active[0].replace("base_", "", 1):
            violations.append(f"row {idx} base_regime mismatch")
    return violations[:20]


def _feature_metrics(feature_rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(feature_rows)
    source_counts: dict[str, int] = {}
    for row in feature_rows:
        split = row.get("source_split", "")
        source_counts[split] = source_counts.get(split, 0) + 1
    metrics = {
        "total_rows": total,
        "source_split_counts": source_counts,
        "identity_rows": _count(feature_rows, "has_identity_features"),
        "cheap_image_rows": _count(feature_rows, "has_cheap_image_features"),
        "cached_matcher_rows": _count(feature_rows, "has_cached_matcher_features"),
        "missing_image_rows": sum(
            1
            for row in feature_rows
            if not _is_one(row.get("image_a_exists")) or not _is_one(row.get("image_b_exists"))
        ),
    }
    for key in ["identity_rows", "cheap_image_rows", "cached_matcher_rows", "missing_image_rows"]:
        metrics[f"{key}_fraction"] = metrics[key] / total if total else 0.0
    return metrics


def _v2_overlap(manifest_rows: list[dict[str, str]], v2: dict[str, Any]) -> dict[str, Any]:
    # v2 metrics do not carry per-row anchors; compare the aggregate anchor floor.
    v2_fraction = v2.get("ordinary_control_anchor_fraction")
    if v2_fraction is None:
        v2_fraction = v2.get("v1_comparison", {}).get("v2_ordinary_control_anchor_fraction")
    if v2_fraction is None:
        v2_fraction = 0.3939393939393939
    row_count = len(manifest_rows)
    v3_fraction = sum(1 for row in manifest_rows if row.get("base_regime") == "ordinary_control_anchor") / row_count if row_count else 0.0
    return {
        "v2_ordinary_control_anchor_fraction": v2_fraction,
        "v3_ordinary_control_anchor_fraction": v3_fraction,
        "regressed_below_v2": v3_fraction < float(v2_fraction),
        "comparison_level": "aggregate_anchor_fraction",
    }


def _verdict(
    feature_metrics: dict[str, Any],
    base_counts: dict[str, int],
    invariant_violations: list[str],
    leakage: dict[str, Any],
    overlap: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    total = feature_metrics["total_rows"]
    train_rows = feature_metrics["source_split_counts"].get("train", 0)
    dev_rows = feature_metrics["source_split_counts"].get("dev", 0)
    unknown_fraction = base_counts.get("unknown_insufficient_features", 0) / total if total else 1.0
    ordinary_fraction = base_counts.get("ordinary_control_anchor", 0) / total if total else 0.0
    max_fraction = max((count / total for count in base_counts.values()), default=1.0) if total else 1.0
    assigned_fraction = 1.0 - unknown_fraction
    if not leakage["passed"]:
        reasons.append("leakage audit failed")
    if invariant_violations:
        reasons.append("base-regime invariant failed")
    if train_rows < 50000 and train_rows < 545 * 100:
        reasons.append(f"train coverage below 50k and below 100x v2: {train_rows}")
    if dev_rows < 10000:
        reasons.append(f"dev coverage below 10k: {dev_rows}")
    if assigned_fraction < 0.95:
        reasons.append(f"base_regime assigned fraction below 95%: {assigned_fraction:.4f}")
    if unknown_fraction > 0.05:
        reasons.append(f"unknown_insufficient_features above 5%: {unknown_fraction:.4f}")
    if ordinary_fraction < 0.15:
        reasons.append(f"ordinary_control_anchor below 15%: {ordinary_fraction:.4f}")
    if max_fraction > 0.90:
        reasons.append(f"base regime collapse above 90%: {max_fraction:.4f}")
    if overlap["regressed_below_v2"]:
        reasons.append("v3 aggregate ordinary/control anchor coverage regressed below v2")
    if reasons:
        return "coverage-rejected", reasons
    return "coverage-ready-for-knowledge-review", ["all v3 coverage gates passed"]


def _write_metrics_csv(path: Path, base_counts: dict[str, int], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_regime", "count", "fraction"])
        writer.writeheader()
        for regime, count in base_counts.items():
            writer.writerow({"base_regime": regime, "count": count, "fraction": count / total if total else 0.0})


def main() -> int:
    args = parse_args()
    feature_rows = _read_csv(args.feature_manifest)
    manifest_rows = _read_csv(args.manifest)
    if len(feature_rows) != len(manifest_rows):
        raise SystemExit(f"feature/manifest row mismatch: {len(feature_rows)} != {len(manifest_rows)}")
    leakage = audit_feature_columns_v3(feature_rows[0].keys() if feature_rows else [])
    invariants = _invariant_violations(manifest_rows)
    feature = _feature_metrics(feature_rows)
    bases = _base_counts(manifest_rows)
    tags = _tag_counts(manifest_rows)
    total = len(manifest_rows)
    overlap = _v2_overlap(manifest_rows, _read_json(args.v2_metrics))
    verdict, reasons = _verdict(feature, bases, invariants, leakage, overlap)
    payload = {
        "schema_version": "phase27_a_manifest_v3_coverage_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_manifest": str(args.feature_manifest),
        "manifest": str(args.manifest),
        "feature_coverage": feature,
        "base_regime_counts": bases,
        "base_regime_fractions": {key: (value / total if total else 0.0) for key, value in bases.items()},
        "risk_tag_counts": tags,
        "risk_tag_fractions": {key: (value / total if total else 0.0) for key, value in tags.items()},
        "unknown_insufficient_features_fraction": bases.get("unknown_insufficient_features", 0) / total if total else 1.0,
        "ordinary_control_anchor_fraction": bases.get("ordinary_control_anchor", 0) / total if total else 0.0,
        "max_base_regime_fraction": max((value / total for value in bases.values()), default=0.0) if total else 0.0,
        "v2_overlap_comparison": overlap,
        "leakage_audit": leakage,
        "invariant_violations": invariants,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "claim_level": "bounded_coverage",
        "training_started": False,
        "submission_created": False,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics_csv(args.metrics_csv, bases, total)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "rows": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
