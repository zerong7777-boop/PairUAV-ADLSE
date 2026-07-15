#!/usr/bin/env python3
"""Feature schema helpers for Phase27 A evidence-state manifest v3."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "phase27_a_feature_v3"

FEATURE_LAYERS = [
    "identity_layout",
    "cheap_image_observability",
    "cached_matcher",
]

FORBIDDEN_CONSTRUCTION_PATTERNS = [
    "heading_num",
    "range_num",
    "abs_range",
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

IDENTITY_COLUMNS = [
    "source_split",
    "json_path",
    "group_id",
    "pair_id",
    "pair_key",
    "image_a",
    "image_b",
    "image_a_name",
    "image_b_name",
    "image_a_index",
    "image_b_index",
    "image_index_gap_abs",
    "query_reference_order",
    "has_identity_features",
]

CHEAP_IMAGE_COLUMNS = [
    "image_a_exists",
    "image_b_exists",
    "image_a_width",
    "image_a_height",
    "image_b_width",
    "image_b_height",
    "aspect_ratio_gap_abs",
    "brightness_gap_abs",
    "contrast_gap_abs",
    "grayscale_hist_similarity",
    "cheap_feature_error",
    "has_cheap_image_features",
]

CACHED_MATCHER_COLUMNS = [
    "has_cached_matcher_features",
    "cached_match_count",
    "cached_spatial_entropy",
    "cached_occupied_cells",
    "cached_anchor_spread",
    "cached_scale_balance",
    "cached_fallback_used",
]

LAYER_FLAG_COLUMNS = [
    "feature_layer_mask",
    "feature_confidence_level",
]


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = _lower(column)
    return any(pattern in lowered for pattern in patterns)


def audit_feature_columns_v3(columns: Iterable[str]) -> dict[str, Any]:
    unique_columns = list(dict.fromkeys(str(column) for column in columns))
    forbidden_columns = [
        column
        for column in unique_columns
        if _matches_any(column, FORBIDDEN_CONSTRUCTION_PATTERNS)
    ]
    return {
        "passed": not forbidden_columns,
        "forbidden_columns": forbidden_columns,
        "column_count": len(unique_columns),
    }


def extract_image_index(name: str | None) -> int | None:
    if not name:
        return None
    match = re.search(r"image-(\d+)", str(name))
    if not match:
        return None
    return int(match.group(1))


def extract_identity_layout_features(row: dict[str, Any]) -> dict[str, Any]:
    image_a_name = row.get("image_a_name") or Path(str(row.get("image_a", ""))).name
    image_b_name = row.get("image_b_name") or Path(str(row.get("image_b", ""))).name
    image_a_index = extract_image_index(str(image_a_name))
    image_b_index = extract_image_index(str(image_b_name))
    gap = None
    if image_a_index is not None and image_b_index is not None:
        gap = abs(image_a_index - image_b_index)
    pair_id = str(row.get("json_id") or "")
    if row.get("group_id") and pair_id and "/" not in pair_id:
        pair_id = f"{row.get('group_id')}/{pair_id}"
    return {
        "source_split": row.get("split", ""),
        "json_path": row.get("json_path", ""),
        "group_id": row.get("group_id", ""),
        "pair_id": pair_id,
        "pair_key": row.get("pair_key", ""),
        "image_a": row.get("image_a", ""),
        "image_b": row.get("image_b", ""),
        "image_a_name": image_a_name,
        "image_b_name": image_b_name,
        "image_a_index": image_a_index if image_a_index is not None else "",
        "image_b_index": image_b_index if image_b_index is not None else "",
        "image_index_gap_abs": gap if gap is not None else "",
        "query_reference_order": "same" if image_a_name == image_b_name else "ordered",
        "has_identity_features": 1,
    }


def _image_stats(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
    except Exception as exc:  # pragma: no cover - depends on environment.
        return {"exists": int(path.exists()), "error": f"PIL unavailable: {exc}"}
    if not path.exists():
        return {"exists": 0, "error": "missing"}
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((64, 64))
            stat = ImageStat.Stat(gray)
            hist = gray.histogram()
            total = float(sum(hist)) or 1.0
            norm_hist = [value / total for value in hist]
            return {
                "exists": 1,
                "width": img.width,
                "height": img.height,
                "aspect": img.width / img.height if img.height else 0.0,
                "brightness": float(stat.mean[0]),
                "contrast": float(stat.stddev[0]),
                "hist": norm_hist,
                "error": "",
            }
    except Exception as exc:
        return {"exists": 0, "error": f"{type(exc).__name__}: {exc}"}


def _hist_similarity(hist_a: list[float] | None, hist_b: list[float] | None) -> float | str:
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return ""
    intersection = sum(min(a, b) for a, b in zip(hist_a, hist_b))
    return max(0.0, min(1.0, intersection))


def compute_cheap_image_features(image_a_path: str | Path, image_b_path: str | Path) -> dict[str, Any]:
    path_a = Path(image_a_path)
    path_b = Path(image_b_path)
    a = _image_stats(path_a)
    b = _image_stats(path_b)
    both = bool(a.get("exists")) and bool(b.get("exists"))
    aspect_gap = ""
    brightness_gap = ""
    contrast_gap = ""
    if both:
        aspect_gap = abs(float(a["aspect"]) - float(b["aspect"]))
        brightness_gap = abs(float(a["brightness"]) - float(b["brightness"]))
        contrast_gap = abs(float(a["contrast"]) - float(b["contrast"]))
    errors = [err for err in [a.get("error"), b.get("error")] if err]
    return {
        "image_a_exists": int(bool(a.get("exists"))),
        "image_b_exists": int(bool(b.get("exists"))),
        "image_a_width": a.get("width", ""),
        "image_a_height": a.get("height", ""),
        "image_b_width": b.get("width", ""),
        "image_b_height": b.get("height", ""),
        "aspect_ratio_gap_abs": aspect_gap,
        "brightness_gap_abs": brightness_gap,
        "contrast_gap_abs": contrast_gap,
        "grayscale_hist_similarity": _hist_similarity(a.get("hist"), b.get("hist")),
        "cheap_feature_error": "; ".join(errors),
        "has_cheap_image_features": int(both),
    }


def make_feature_layer_flags(row: dict[str, Any]) -> dict[str, Any]:
    has_identity = int(str(row.get("has_identity_features", "0")) in {"1", "1.0", "true", "True"})
    has_cheap = int(str(row.get("has_cheap_image_features", "0")) in {"1", "1.0", "true", "True"})
    has_cached = int(str(row.get("has_cached_matcher_features", "0")) in {"1", "1.0", "true", "True"})
    layers = []
    if has_identity:
        layers.append("identity")
    if has_cheap:
        layers.append("cheap")
    if has_cached:
        layers.append("cached")
    confidence = "none"
    if has_identity:
        confidence = "identity"
    if has_identity and has_cheap:
        confidence = "cheap"
    if has_cached:
        confidence = "cached"
    return {
        "has_identity_features": has_identity,
        "has_cheap_image_features": has_cheap,
        "has_cached_matcher_features": has_cached,
        "feature_layer_mask": "+".join(layers),
        "feature_confidence_level": confidence,
    }


def resolve_image_path(project_root: Path, image_rel: str) -> Path:
    rel = Path(str(image_rel))
    candidates = [
        project_root / rel,
        project_root / "official" / "UAVM_2026" / "pairUAV" / "train_tour" / rel,
        project_root / "official" / "UAVM_2026" / "pairUAV" / "test_query" / rel,
        project_root / "official" / "UAVM_2026" / "pairUAV" / "test_gallery" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]
