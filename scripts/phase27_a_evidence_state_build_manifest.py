#!/usr/bin/env python3
"""Build Phase27 A non-leaky evidence-state manifests from observable packets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from scripts.phase27_a_evidence_state_schema import (
    SCHEMA_VERSION,
    STATE_COLUMNS,
    audit_construction_columns,
    assign_evidence_states,
)


FEATURE_PATHS = {
    "eval": "experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl",
    "train": "experiments/phase26_b_selective_correspondence_reasoning/features/train_bscr_features.jsonl",
}

CONSTRUCTION_COLUMNS = [
    "pair_id",
    "query_name",
    "reference_name",
    "group_id",
    "match_count",
    "valid_ratio",
    "confidence_sum",
    "mean_confidence",
    "max_confidence",
    "spatial_entropy",
    "occupied_cells",
    "anchor_spread",
    "scale_balance",
    "semantic_proxy",
    "geometry_proxy",
    "target_shift_proxy",
    "fallback_used",
    "construction_source",
    "schema_version",
    "leakage_audit_passed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--split", choices=sorted(FEATURE_PATHS), default="eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--schema-out", required=True, type=Path)
    parser.add_argument("--metrics-out", type=Path)
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


def _sample_names(sample_id: str, match_path: str) -> tuple[str, str, str]:
    group_id = sample_id.split("/", 1)[0] if "/" in sample_id else ""
    if "/" in sample_id and "_" in sample_id.rsplit("/", 1)[-1]:
        left, right = sample_id.rsplit("/", 1)[-1].split("_", 1)
        return group_id, f"image-{left}", f"image-{right}"
    name = Path(match_path).name
    if "_matches" in name and "_" in name:
        parts = name.replace("_matches.npz", "").split("_")
        if len(parts) >= 2:
            return group_id, parts[0], parts[1]
    return group_id, sample_id, ""


def _occupied_cells(spatial_bins: Any) -> int | None:
    if not isinstance(spatial_bins, list):
        return None
    count = 0
    for row in spatial_bins:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, list) and cell and _safe_float(cell[0]) and _safe_float(cell[0]) > 0:
                count += 1
    return count


def _anchor_stats(topk_anchors: Any) -> tuple[float | None, float | None]:
    if not isinstance(topk_anchors, list) or not topk_anchors:
        return None, None
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
        return None, None
    disp_mean = sum(disps) / len(disps)
    variance = sum((value - disp_mean) ** 2 for value in disps) / len(disps)
    anchor_spread = math.sqrt(variance)
    if len(disps) < 2:
        scale_balance = 1.0
    else:
        scale_balance = 1.0 / (1.0 + anchor_spread + abs(max(disps) - min(disps)))
    if confs:
        scale_balance *= max(0.0, min(1.0, sum(confs) / len(confs)))
    return anchor_spread, scale_balance


def _geometry_proxy(match_count: float | None, occupied_cells: int | None, entropy: float | None) -> float | None:
    parts: list[float] = []
    if match_count is not None:
        parts.append(max(0.0, min(1.0, math.log1p(match_count) / math.log1p(100.0))))
    if occupied_cells is not None:
        parts.append(max(0.0, min(1.0, occupied_cells / 16.0)))
    if entropy is not None:
        parts.append(max(0.0, min(1.0, entropy)))
    if not parts:
        return None
    return sum(parts) / len(parts)


def _record_to_manifest_row(record: dict[str, Any], split: str) -> dict[str, Any]:
    raw = record.get("raw_global_stats") if isinstance(record.get("raw_global_stats"), dict) else {}
    sample_id = str(record.get("sample_id") or "")
    match_path = str(record.get("match_path") or "")
    group_id, query_name, reference_name = _sample_names(sample_id, match_path)
    log_match = _safe_float(raw.get("log1p_match_count"))
    match_count = math.expm1(log_match) if log_match is not None else None
    mean_confidence = _safe_float(raw.get("mean_confidence"))
    max_confidence = _safe_float(raw.get("max_confidence"))
    spatial_entropy = _safe_float(raw.get("spatial_entropy"))
    occupied = _occupied_cells(record.get("spatial_bins"))
    anchor_spread, scale_balance = _anchor_stats(record.get("topk_anchors"))
    confidence_sum = None
    if match_count is not None and mean_confidence is not None:
        confidence_sum = match_count * mean_confidence
    valid_ratio = None
    quality_mask = record.get("quality_mask")
    if isinstance(quality_mask, list) and quality_mask:
        values = [_safe_float(value) for value in quality_mask]
        valid_values = [value for value in values if value is not None]
        if valid_values:
            valid_ratio = sum(valid_values) / len(valid_values)
    fallback_used = bool(record.get("fallback_used") or raw.get("fallback_used"))
    geometry_proxy = _geometry_proxy(match_count, occupied, spatial_entropy)
    return {
        "pair_id": sample_id,
        "query_name": query_name,
        "reference_name": reference_name,
        "group_id": group_id,
        "match_count": match_count,
        "valid_ratio": valid_ratio,
        "confidence_sum": confidence_sum,
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence,
        "spatial_entropy": spatial_entropy,
        "occupied_cells": occupied,
        "anchor_spread": anchor_spread,
        "scale_balance": scale_balance,
        "semantic_proxy": "",
        "geometry_proxy": geometry_proxy,
        "target_shift_proxy": "",
        "fallback_used": int(fallback_used),
        "construction_source": f"{split}_bscr_features",
        "schema_version": SCHEMA_VERSION,
        "leakage_audit_passed": True,
    }


def _read_features(path: Path, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(_record_to_manifest_row(json.loads(stripped), split))
            if limit and len(rows) >= limit:
                break
    return rows


def _quantile(values: list[float], q: float, default: float) -> float:
    clean = sorted(value for value in values if value is not None and not math.isnan(value))
    if not clean:
        return default
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[int(pos)]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def _compute_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if isinstance(row.get(name), (int, float))]

    return {
        "low_match_count": _quantile(values("match_count"), 0.20, 10.0),
        "low_valid_ratio": _quantile(values("valid_ratio"), 0.20, 0.20),
        "low_occupied_cells": _quantile(values("occupied_cells"), 0.20, 4.0),
        "high_match_count": _quantile(values("match_count"), 0.80, 50.0),
        "high_valid_ratio": _quantile(values("valid_ratio"), 0.70, 0.50),
        "high_confidence_sum": _quantile(values("confidence_sum"), 0.80, 5.0),
        "high_occupied_cells": _quantile(values("occupied_cells"), 0.70, 8.0),
        "low_anchor_spread": _quantile(values("anchor_spread"), 0.20, 0.15),
        "high_anchor_spread": _quantile(values("anchor_spread"), 0.80, 0.60),
        "low_spatial_entropy": _quantile(values("spatial_entropy"), 0.20, 0.25),
        "low_scale_balance": _quantile(values("scale_balance"), 0.20, 0.35),
        "high_scale_balance": _quantile(values("scale_balance"), 0.80, 0.75),
        "semantic_geometry_gap": 0.35,
        "target_shift_proxy": 0.70,
    }


def _write_schema(path: Path, thresholds: dict[str, float], audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "construction_columns": CONSTRUCTION_COLUMNS,
        "state_columns": STATE_COLUMNS,
        "thresholds": thresholds,
        "construction_column_audit": audit,
        "forbidden_feature_policy": "Forbidden fields may be used only after schema freeze for validation.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _write_manifest(path: Path, rows: list[dict[str, Any]], thresholds: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CONSTRUCTION_COLUMNS + STATE_COLUMNS + ["state_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            states = assign_evidence_states(row, thresholds)
            out = dict(row)
            out.update(states)
            out["state_count"] = sum(states.values())
            writer.writerow({key: _format_value(out.get(key, "")) for key in fieldnames})


def _state_counts(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, int]:
    counts = {state: 0 for state in STATE_COLUMNS}
    for row in rows:
        states = assign_evidence_states(row, thresholds)
        for state, value in states.items():
            counts[state] += int(value)
    return counts


def _write_metrics(path: Path, rows: list[dict[str, Any]], thresholds: dict[str, float], audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = _state_counts(rows, thresholds)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "leakage_audit": audit,
        "state_counts": counts,
        "state_fractions": {state: (count / len(rows) if rows else 0.0) for state, count in counts.items()},
        "missing_feature_counts": {
            column: sum(1 for row in rows if row.get(column) in (None, ""))
            for column in [
                "match_count",
                "valid_ratio",
                "confidence_sum",
                "spatial_entropy",
                "occupied_cells",
                "anchor_spread",
                "scale_balance",
                "semantic_proxy",
                "target_shift_proxy",
            ]
        },
        "thresholds": thresholds,
        "claim_level": "smoke" if len(rows) <= 128 else "bounded_manifest",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    feature_path = args.project_root / FEATURE_PATHS[args.split]
    rows = _read_features(feature_path, args.split, args.limit)
    audit = audit_construction_columns(CONSTRUCTION_COLUMNS)
    if not audit["passed"]:
        raise SystemExit(f"construction leakage audit failed: {audit['forbidden_columns']}")
    thresholds = _compute_thresholds(rows)
    _write_schema(args.schema_out, thresholds, audit)
    _write_manifest(args.out, rows, thresholds)
    if args.metrics_out:
        _write_metrics(args.metrics_out, rows, thresholds, audit)
    print(json.dumps({"rows": len(rows), "out": str(args.out), "schema_out": str(args.schema_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
