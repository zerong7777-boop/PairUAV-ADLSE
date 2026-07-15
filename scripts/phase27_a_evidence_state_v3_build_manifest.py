#!/usr/bin/env python3
"""Assign v3 evidence-state regimes from v3 feature manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_schema_v2 import (
    BASE_FLAG_COLUMNS,
    BASE_REGIMES,
    RISK_TAGS,
    assign_evidence_state_v2,
)


OUTPUT_EXTRA = [
    "base_regime",
    *BASE_FLAG_COLUMNS,
    *RISK_TAGS,
    "ordinary_with_risk_tag",
    "evidence_input_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    return parser.parse_args()


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except ValueError:
        return 0


def _evidence_inputs(row: dict[str, str]) -> tuple[dict[str, Any], str]:
    if _int(row, "has_cached_matcher_features"):
        return (
            {
                "match_count": _float(row, "cached_match_count"),
                "valid_ratio": 1.0,
                "confidence_sum": (_float(row, "cached_match_count") or 0.0) * 0.4,
                "mean_confidence": 0.4,
                "spatial_entropy": _float(row, "cached_spatial_entropy"),
                "occupied_cells": _float(row, "cached_occupied_cells"),
                "anchor_spread": _float(row, "cached_anchor_spread"),
                "scale_balance": _float(row, "cached_scale_balance"),
                "fallback_used": _int(row, "cached_fallback_used"),
            },
            "cached_matcher",
        )
    if _int(row, "has_cheap_image_features"):
        hist = _float(row, "grayscale_hist_similarity")
        aspect_gap = _float(row, "aspect_ratio_gap_abs")
        brightness_gap = _float(row, "brightness_gap_abs")
        contrast_gap = _float(row, "contrast_gap_abs")
        gap = _float(row, "image_index_gap_abs")
        hist = hist if hist is not None else 0.5
        aspect_gap = aspect_gap if aspect_gap is not None else 0.0
        brightness_gap = brightness_gap if brightness_gap is not None else 0.0
        contrast_gap = contrast_gap if contrast_gap is not None else 0.0
        gap = gap if gap is not None else 0.0
        match_proxy = max(5.0, min(80.0, 10.0 + hist * 55.0 + max(0.0, 15.0 - gap) * 0.5))
        occupied_proxy = max(2.0, min(16.0, 4.0 + hist * 8.0 + max(0.0, 1.0 - aspect_gap) * 4.0))
        scale_balance = max(0.0, min(1.0, 1.0 - aspect_gap))
        spatial_entropy = max(0.0, min(1.0, hist - min(0.3, brightness_gap / 255.0)))
        anchor_spread = max(0.01, min(0.5, gap / 100.0 + contrast_gap / 255.0))
        return (
            {
                "match_count": match_proxy,
                "valid_ratio": 1.0,
                "confidence_sum": match_proxy * max(0.2, hist),
                "mean_confidence": max(0.2, min(0.8, hist)),
                "spatial_entropy": spatial_entropy,
                "occupied_cells": occupied_proxy,
                "anchor_spread": anchor_spread,
                "scale_balance": scale_balance,
                "fallback_used": 0,
            },
            "cheap_proxy",
        )
    if _int(row, "has_identity_features"):
        gap = _float(row, "image_index_gap_abs")
        gap = gap if gap is not None else 50.0
        return (
            {
                "match_count": max(5.0, min(40.0, 30.0 - gap * 0.2)),
                "valid_ratio": 1.0,
                "confidence_sum": max(2.0, min(15.0, 12.0 - gap * 0.05)),
                "mean_confidence": 0.35,
                "spatial_entropy": 0.5,
                "occupied_cells": max(4.0, min(12.0, 10.0 - gap * 0.05)),
                "anchor_spread": max(0.03, min(0.4, gap / 100.0)),
                "scale_balance": 0.5,
                "fallback_used": 0,
            },
            "identity_only",
        )
    return ({}, "insufficient")


def main() -> int:
    args = parse_args()
    rows = []
    with args.feature_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = list(reader.fieldnames or [])
        for row in reader:
            evidence, source = _evidence_inputs(row)
            assignment = assign_evidence_state_v2(evidence)
            out = dict(row)
            out.update(assignment)
            out["evidence_input_source"] = source
            rows.append(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = input_columns + OUTPUT_EXTRA
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})
    base_counts = {regime: sum(1 for row in rows if row.get("base_regime") == regime) for regime in BASE_REGIMES}
    tag_counts = {tag: sum(int(row.get(tag, 0)) for row in rows) for tag in RISK_TAGS}
    metrics = {
        "row_count": len(rows),
        "base_regime_counts": base_counts,
        "base_regime_fractions": {key: (value / len(rows) if rows else 0.0) for key, value in base_counts.items()},
        "risk_tag_counts": tag_counts,
        "unknown_insufficient_features_fraction": base_counts.get("unknown_insufficient_features", 0) / len(rows) if rows else 0.0,
        "ordinary_control_anchor_fraction": base_counts.get("ordinary_control_anchor", 0) / len(rows) if rows else 0.0,
        "input_source_counts": {
            source: sum(1 for row in rows if row.get("evidence_input_source") == source)
            for source in ["cached_matcher", "cheap_proxy", "identity_only", "insufficient"]
        },
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(args.out), "base_counts": base_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
