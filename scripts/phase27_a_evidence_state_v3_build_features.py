#!/usr/bin/env python3
"""Build Phase27 v3 feature manifests with broad non-leaky coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_v3_feature_schema import (
    CACHED_MATCHER_COLUMNS,
    CHEAP_IMAGE_COLUMNS,
    IDENTITY_COLUMNS,
    LAYER_FLAG_COLUMNS,
    SCHEMA_VERSION,
    audit_feature_columns_v3,
    compute_cheap_image_features,
    extract_identity_layout_features,
    make_feature_layer_flags,
    resolve_image_path,
)


IDENTITY_MANIFESTS = {
    "train": "experiments/paper_pillars/15_train_test_distribution_gap/manifests/train_labeled_manifest.csv",
    "dev": "experiments/paper_pillars/15_train_test_distribution_gap/manifests/dev_labeled_manifest.csv",
}

OUTPUT_COLUMNS = (
    IDENTITY_COLUMNS
    + CHEAP_IMAGE_COLUMNS
    + CACHED_MATCHER_COLUMNS
    + LAYER_FLAG_COLUMNS
    + ["schema_version"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--split", choices=sorted(IDENTITY_MANIFESTS), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--enable-cheap-image-features", choices=["true", "false"], default="true")
    parser.add_argument("--cached-matcher-jsonl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _occupied_cells(spatial_bins: Any) -> int | str:
    if not isinstance(spatial_bins, list):
        return ""
    count = 0
    for row in spatial_bins:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, list) and cell:
                first = _safe_float(cell[0])
                if first is not None and first > 0:
                    count += 1
    return count


def _anchor_stats(topk_anchors: Any) -> tuple[float | str, float | str]:
    if not isinstance(topk_anchors, list) or not topk_anchors:
        return "", ""
    disps: list[float] = []
    confs: list[float] = []
    for anchor in topk_anchors:
        if not isinstance(anchor, list) or len(anchor) < 5:
            continue
        x1, y1, x2, y2, conf = (_safe_float(value) for value in anchor[:5])
        if None in (x1, y1, x2, y2):
            continue
        disps.append(math.hypot(float(x2) - float(x1), float(y2) - float(y1)))
        if conf is not None:
            confs.append(float(conf))
    if not disps:
        return "", ""
    mean_disp = sum(disps) / len(disps)
    variance = sum((value - mean_disp) ** 2 for value in disps) / len(disps)
    spread = math.sqrt(variance)
    balance = 1.0 / (1.0 + spread + (max(disps) - min(disps) if len(disps) > 1 else 0.0))
    if confs:
        balance *= max(0.0, min(1.0, sum(confs) / len(confs)))
    return spread, balance


def _load_cached(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            sample_id = str(record.get("sample_id") or "")
            raw = record.get("raw_global_stats") if isinstance(record.get("raw_global_stats"), dict) else {}
            log_match = _safe_float(raw.get("log1p_match_count"))
            match_count = math.expm1(log_match) if log_match is not None else ""
            spread, balance = _anchor_stats(record.get("topk_anchors"))
            cache[sample_id] = {
                "has_cached_matcher_features": 1,
                "cached_match_count": match_count,
                "cached_spatial_entropy": raw.get("spatial_entropy", ""),
                "cached_occupied_cells": _occupied_cells(record.get("spatial_bins")),
                "cached_anchor_spread": spread,
                "cached_scale_balance": balance,
                "cached_fallback_used": int(bool(record.get("fallback_used") or raw.get("fallback_used"))),
            }
    return cache


def _empty_cheap() -> dict[str, Any]:
    return {column: "" for column in CHEAP_IMAGE_COLUMNS} | {"has_cheap_image_features": 0}


def _empty_cached() -> dict[str, Any]:
    return {column: "" for column in CACHED_MATCHER_COLUMNS} | {"has_cached_matcher_features": 0}


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def main() -> int:
    args = parse_args()
    start = time.time()
    identity_path = args.project_root / IDENTITY_MANIFESTS[args.split]
    cached = _load_cached(args.cached_matcher_jsonl)
    rows_written = 0
    selected_seen = 0
    image_failures = 0
    cheap_rows = 0
    cached_rows = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with identity_path.open("r", encoding="utf-8", newline="") as handle, args.out.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.DictReader(handle)
        audit = audit_feature_columns_v3(reader.fieldnames or [])
        # Labeled manifests contain forbidden label columns; do not write them.
        writer = csv.DictWriter(out_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(reader):
            if args.stride > 1 and idx % args.stride != 0:
                continue
            if args.limit and selected_seen >= args.limit:
                break
            selected_seen += 1
            identity = extract_identity_layout_features(row)
            cheap = _empty_cheap()
            if args.enable_cheap_image_features == "true":
                image_a_path = resolve_image_path(args.project_root, str(row.get("image_a", "")))
                image_b_path = resolve_image_path(args.project_root, str(row.get("image_b", "")))
                cheap = compute_cheap_image_features(image_a_path, image_b_path)
                if cheap.get("has_cheap_image_features"):
                    cheap_rows += 1
                else:
                    image_failures += 1
            pair_id = str(identity.get("pair_id", ""))
            cached_features = dict(cached.get(pair_id, _empty_cached()))
            if cached_features.get("has_cached_matcher_features"):
                cached_rows += 1
            combined = {}
            combined.update(identity)
            combined.update(cheap)
            combined.update(cached_features)
            combined.update(make_feature_layer_flags(combined))
            combined["schema_version"] = SCHEMA_VERSION
            writer.writerow({column: _fmt(combined.get(column, "")) for column in OUTPUT_COLUMNS})
            rows_written += 1
    elapsed = time.time() - start
    projected_train_seconds = None
    if rows_written:
        projected_train_seconds = elapsed / rows_written * 1839996
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "selected_rows": rows_written,
        "rows_with_identity_features": rows_written,
        "rows_with_cheap_image_features": cheap_rows,
        "rows_with_cached_matcher_features": cached_rows,
        "image_path_failure_count": image_failures,
        "cheap_feature_extraction_seconds": elapsed,
        "estimated_full_train_seconds": projected_train_seconds,
        "source_column_audit": audit,
        "construction_column_audit": audit_feature_columns_v3(OUTPUT_COLUMNS),
        "claim_level": "smoke" if rows_written <= 128 else "bounded_feature",
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows_written, "cheap_rows": cheap_rows, "cached_rows": cached_rows, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
