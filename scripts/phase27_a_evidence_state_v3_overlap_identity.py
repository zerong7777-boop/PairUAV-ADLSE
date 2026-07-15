#!/usr/bin/env python3
"""Phase27 A evidence-state calibration-v3 overlap identity helpers.

This module is intentionally limited to identity normalization. It must not
construct labels, errors, scores, predictions, checkpoints, or training inputs.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


FORBIDDEN_IDENTITY_PATTERNS = [
    "final_score",
    "distance_rel_error",
    "angle_rel_error",
    "range_err",
    "angle_err",
    "combined_error",
    "residual",
    "gt_angle",
    "gt_distance",
    "official",
    "leaderboard",
]

IDENTITY_COLUMNS = [
    "group_id",
    "pair_id",
    "pair_key",
    "json_path",
    "image_a",
    "image_b",
    "image_a_name",
    "image_b_name",
    "query_image",
    "reference_image",
    "satellite_image",
    "drone_image",
]

_IMAGE_EXT_RE = re.compile(r"\.(jpeg|jpg|png|bmp|tif|tiff)$", re.IGNORECASE)
_IMAGE_PREFIX_RE = re.compile(r"^(image-|image_|img-|img_)", re.IGNORECASE)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    lowered = column.lower()
    return any(pattern in lowered for pattern in patterns)


def audit_forbidden_columns(columns: Iterable[str]) -> dict[str, Any]:
    unique_columns = list(dict.fromkeys(_as_text(column) for column in columns))
    forbidden_columns = [
        column
        for column in unique_columns
        if _matches_any(column, FORBIDDEN_IDENTITY_PATTERNS)
    ]
    return {
        "passed": not forbidden_columns,
        "forbidden_columns": forbidden_columns,
        "column_count": len(unique_columns),
        "identity_columns": [
            column
            for column in unique_columns
            if column in IDENTITY_COLUMNS and column not in forbidden_columns
        ],
    }


def canonical_image_token(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""

    basename = re.split(r"[\\/]", text)[-1].strip().lower()
    basename = _IMAGE_EXT_RE.sub("", basename)
    basename = _IMAGE_PREFIX_RE.sub("", basename)

    integer_tokens = re.findall(r"\d+", basename)
    if not integer_tokens:
        return ""

    number = int(integer_tokens[-1])
    return f"{number:02d}" if number < 100 else str(number)


def _canonical_group_token(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""

    normalized = text.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    search_text = "/".join(parts)

    group_match = re.search(
        r"(?:group|target)[_-]?(\d+)", search_text, flags=re.IGNORECASE
    )
    if group_match:
        return group_match.group(1)

    if re.fullmatch(r"\d+", text):
        return text

    integer_tokens = re.findall(r"\d+", search_text)
    for token in integer_tokens:
        if len(token) >= 3:
            return token
    return integer_tokens[0] if integer_tokens else ""


def _split_pair_id(value: Any, fallback_group_id: str) -> tuple[str, str, str] | None:
    text = _as_text(value)
    if not text:
        return None

    group_id = fallback_group_id
    rest = text.replace("\\", "/")
    if "/" in rest:
        prefix, rest = rest.split("/", 1)
        group_id = _canonical_group_token(prefix) or group_id

    integer_tokens = re.findall(r"\d+", rest)
    if len(integer_tokens) < 2:
        return None

    if not group_id and len(integer_tokens) >= 3:
        group_id = integer_tokens[0]
        integer_tokens = integer_tokens[1:]

    query_id = canonical_image_token(integer_tokens[0])
    reference_id = canonical_image_token(integer_tokens[1])
    if not group_id or not query_id or not reference_id:
        return None
    return group_id, query_id, reference_id


def _derive_group_id(row: dict[str, Any]) -> str:
    for column in ("group_id", "group", "target"):
        group_id = _canonical_group_token(row.get(column))
        if group_id:
            return group_id

    for column in ("json_path", "pair_id", "pair_key"):
        group_id = _canonical_group_token(row.get(column))
        if group_id:
            return group_id

    return ""


def _derive_pair_from_columns(
    row: dict[str, Any], group_id: str
) -> tuple[str, str, str, str] | None:
    column_pairs = [
        ("image_a_name", "image_b_name"),
        ("image_a", "image_b"),
        ("query_image", "reference_image"),
        ("satellite_image", "drone_image"),
    ]

    for left_column, right_column in column_pairs:
        left = canonical_image_token(row.get(left_column))
        right = canonical_image_token(row.get(right_column))
        if group_id and left and right:
            return group_id, left, right, f"{left_column}/{right_column}"

    return None


def _make_pair_keys(
    group_id: str, query_id: str, reference_id: str, identity_source: str
) -> dict[str, str]:
    canonical_pair_id = f"{group_id}/{query_id}_{reference_id}"
    canonical_pair_id_flipped = f"{group_id}/{reference_id}_{query_id}"
    ordered = sorted([query_id, reference_id], key=lambda token: (int(token), token))
    canonical_pair_id_orderless = f"{group_id}/{ordered[0]}_{ordered[1]}"

    return {
        "canonical_pair_id": canonical_pair_id,
        "canonical_pair_id_flipped": canonical_pair_id_flipped,
        "canonical_pair_id_orderless": canonical_pair_id_orderless,
        "canonical_group_id": group_id,
        "canonical_query_id": query_id,
        "canonical_reference_id": reference_id,
        "identity_source": identity_source,
    }


def canonical_pair_keys(row: dict[str, Any]) -> dict[str, str]:
    group_id = _derive_group_id(row)

    for column in ("pair_id", "pair_key"):
        parsed = _split_pair_id(row.get(column), group_id)
        if parsed is not None:
            parsed_group, query_id, reference_id = parsed
            return _make_pair_keys(parsed_group, query_id, reference_id, column)

    parsed_columns = _derive_pair_from_columns(row, group_id)
    if parsed_columns is not None:
        parsed_group, query_id, reference_id, source = parsed_columns
        return _make_pair_keys(parsed_group, query_id, reference_id, source)

    return {
        "canonical_pair_id": "",
        "canonical_pair_id_flipped": "",
        "canonical_pair_id_orderless": "",
        "canonical_group_id": group_id,
        "canonical_query_id": "",
        "canonical_reference_id": "",
        "identity_source": "missing",
    }


def _parse_pair_key(key: Any) -> tuple[str, str, str] | None:
    text = _as_text(key)
    match = re.fullmatch(r"([^/]+)/([^_]+)_([^_]+)", text)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def classify_order(left_key: Any, right_key: Any) -> str:
    left = _parse_pair_key(left_key)
    right = _parse_pair_key(right_key)
    if left is None or right is None:
        return "missing_key"

    if left == right:
        return "same_order"

    left_group, left_query, left_reference = left
    right_group, right_query, right_reference = right
    if (
        left_group == right_group
        and left_query == right_reference
        and left_reference == right_query
    ):
        return "flipped_order"

    return "different_pair"


def count_key_duplicates(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [_as_text(row.get(key)) for row in rows]
    nonempty_values = [value for value in values if value]
    counts = Counter(nonempty_values)
    duplicate_keys = {
        value: count for value, count in sorted(counts.items()) if count > 1
    }

    return {
        "key": key,
        "row_count": len(values),
        "nonempty_row_count": len(nonempty_values),
        "unique_key_count": len(counts),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_row_count": sum(duplicate_keys.values()),
        "duplicate_keys": duplicate_keys,
    }


def key_coverage(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    materialized = list(rows)
    row_count = len(materialized)
    nonempty_row_count = sum(1 for row in materialized if _as_text(row.get(key)))
    return {
        "key": key,
        "row_count": row_count,
        "nonempty_row_count": nonempty_row_count,
        "missing_row_count": row_count - nonempty_row_count,
        "nonempty_fraction": nonempty_row_count / row_count if row_count else 0.0,
    }


def attach_canonical_keys(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.update(canonical_pair_keys(row))
    return out
