"""Canonical identity helpers for Phase27 A validation spine."""

from __future__ import annotations

import os
import re
from collections import Counter

KEY_SCHEMA_VERSION = "pair_key_v1"

FORBIDDEN_COLUMN_TOKENS = (
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
)

_IMAGE_EXT_RE = re.compile(r"\.(jpeg|jpg|png|bmp|tif|tiff)$", re.IGNORECASE)
_IMAGE_PREFIX_RE = re.compile(r"^(image-|image_|img-|img_)", re.IGNORECASE)


def _text(value):
    return "" if value is None else str(value).strip()


def _forbidden(column):
    lowered = column.lower()
    return any(token in lowered for token in FORBIDDEN_COLUMN_TOKENS)


def canonical_image_key(value):
    """Return a stable image token from a path or display name."""
    text = _text(value)
    if not text:
        return ""
    basename = os.path.basename(text.replace("\\", "/")).lower()
    basename = _IMAGE_EXT_RE.sub("", basename)
    basename = _IMAGE_PREFIX_RE.sub("", basename)
    numbers = re.findall(r"\d+", basename)
    if not numbers:
        return ""
    number = int(numbers[-1])
    return f"{number:02d}" if number < 100 else str(number)


def _canonical_group(value):
    text = _text(value)
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    match = re.search(r"(?:group|target)[_-]?(\d+)", normalized, re.IGNORECASE)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", text):
        return text
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return text
    numbers = re.findall(r"\d+", normalized)
    for token in numbers:
        if len(token) >= 3:
            return token
    return numbers[0] if numbers else ""


def _namespace(row, default_namespace):
    return _text(row.get("namespace")) or _text(row.get("split")) or default_namespace


def _pair_from_pair_id(value, fallback_group):
    text = _text(value).replace("\\", "/")
    if not text:
        return None
    group = fallback_group
    rest = text
    if "/" in text:
        prefix, rest = text.split("/", 1)
        group = _canonical_group(prefix) or group
    tokens = re.findall(r"\d+", rest)
    if len(tokens) < 2:
        return None
    if not group and len(tokens) >= 3:
        group = tokens[0]
        tokens = tokens[1:]
    source = canonical_image_key(tokens[0])
    target = canonical_image_key(tokens[1])
    if not group or not source or not target:
        return None
    return group, source, target


def _pair_from_columns(row, group):
    candidates = (
        ("image_a_name", "image_b_name"),
        ("image_a", "image_b"),
        ("query_image", "reference_image"),
        ("source_image", "target_image"),
        ("satellite_image", "drone_image"),
    )
    for left, right in candidates:
        source = canonical_image_key(row.get(left))
        target = canonical_image_key(row.get(right))
        if group and source and target:
            return group, source, target, f"{left}/{right}"
    return None


def canonical_pair_keys(row, default_namespace="unknown"):
    """Return order-sensitive, flipped, and orderless pair keys."""
    namespace = _namespace(row, default_namespace)
    group = ""
    for key in ("group_id", "group", "target"):
        group = _canonical_group(row.get(key))
        if group:
            break
    if not group:
        for key in ("pair_id", "pair_key", "json_path"):
            group = _canonical_group(row.get(key))
            if group:
                break

    for key in ("pair_id", "pair_key"):
        parsed = _pair_from_pair_id(row.get(key), group)
        if parsed:
            group, source, target = parsed
            identity_source = key
            break
    else:
        parsed_columns = _pair_from_columns(row, group)
        if parsed_columns:
            group, source, target, identity_source = parsed_columns
        else:
            source = target = ""
            identity_source = "missing"

    ordered = sorted([source, target], key=lambda item: (int(item) if item.isdigit() else 10**9, item))
    prefix = f"{namespace}/{group}" if namespace else group
    canonical = f"{prefix}/{source}_{target}" if group and source and target else ""
    flipped = f"{prefix}/{target}_{source}" if group and source and target else ""
    orderless = f"{prefix}/{ordered[0]}_{ordered[1]}" if group and source and target else ""
    return {
        "canonical_pair_id": canonical,
        "flipped_pair_id": flipped,
        "orderless_pair_id": orderless,
        "canonical_group_id": group,
        "source_image_key": source,
        "target_image_key": target,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "identity_source": identity_source,
    }


def attach_canonical_keys(row, default_namespace="unknown"):
    out = dict(row)
    out.update(canonical_pair_keys(row, default_namespace=default_namespace))
    return out


def audit_forbidden_columns(columns):
    unique = list(dict.fromkeys(_text(column) for column in columns))
    forbidden = [column for column in unique if _forbidden(column)]
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "column_count": len(unique),
    }


def count_duplicate_keys(rows, key):
    values = [_text(row.get(key)) for row in rows if _text(row.get(key))]
    counts = Counter(values)
    duplicate_keys = {value: count for value, count in counts.items() if count > 1}
    return {
        "key": key,
        "row_count": len(rows),
        "nonempty_row_count": len(values),
        "unique_key_count": len(counts),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_row_count": sum(duplicate_keys.values()),
        "duplicate_keys": duplicate_keys,
    }


def classify_pair_overlap(left_row, right_row):
    left = _text(left_row.get("canonical_pair_id"))
    right = _text(right_row.get("canonical_pair_id"))
    if left and right and left == right:
        return "same_order"
    if left and left == _text(right_row.get("flipped_pair_id")):
        return "flipped_order"
    if _text(left_row.get("orderless_pair_id")) and _text(left_row.get("orderless_pair_id")) == _text(right_row.get("orderless_pair_id")):
        return "orderless_only"
    return "different_pair"


def _set(rows, key):
    return {_text(row.get(key)) for row in rows if _text(row.get(key))}


def _examples(rows, matched, key):
    examples = []
    for row in rows:
        value = _text(row.get(key))
        if value and value in matched:
            continue
        examples.append({k: row.get(k, "") for k in ("canonical_pair_id", "flipped_pair_id", "orderless_pair_id", "canonical_group_id", "source_image_key", "target_image_key")})
        if len(examples) >= 10:
            break
    return examples


def audit_overlap(left_name, left_rows, right_name, right_rows):
    left_raw = _set(left_rows, "pair_id")
    right_raw = _set(right_rows, "pair_id")
    left_canonical = _set(left_rows, "canonical_pair_id")
    right_canonical = _set(right_rows, "canonical_pair_id")
    left_flipped = _set(left_rows, "flipped_pair_id")
    right_flipped = _set(right_rows, "flipped_pair_id")
    left_orderless = _set(left_rows, "orderless_pair_id")
    right_orderless = _set(right_rows, "orderless_pair_id")
    left_groups = _set(left_rows, "canonical_group_id")
    right_groups = _set(right_rows, "canonical_group_id")
    canonical_matches = left_canonical & right_canonical
    orderless_matches = left_orderless & right_orderless
    flipped_overlap = max(0, len(orderless_matches) - len(canonical_matches))
    return {
        "left_artifact_id": left_name,
        "right_artifact_id": right_name,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "raw_overlap": len(left_raw & right_raw),
        "canonical_overlap": len(canonical_matches),
        "order_flipped_overlap": flipped_overlap,
        "orderless_overlap": len(orderless_matches),
        "group_overlap": len(left_groups & right_groups),
        "image_name_only_overlap": 0,
        "left_duplicate_keys": count_duplicate_keys(left_rows, "canonical_pair_id")["duplicate_key_count"],
        "right_duplicate_keys": count_duplicate_keys(right_rows, "canonical_pair_id")["duplicate_key_count"],
        "collision_count": 0,
        "unmatched_left_count": max(0, len(left_rows) - len(canonical_matches) - flipped_overlap),
        "unmatched_right_count": max(0, len(right_rows) - len(canonical_matches) - flipped_overlap),
        "unmatched_left_examples": _examples(left_rows, canonical_matches, "canonical_pair_id"),
        "unmatched_right_examples": _examples(right_rows, canonical_matches, "canonical_pair_id"),
        "failure_classification": "mixed_or_unresolved",
    }
