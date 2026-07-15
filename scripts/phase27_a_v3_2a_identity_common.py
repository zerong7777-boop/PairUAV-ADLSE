"""Common identity helpers for Phase27 A-v3.2a join audits."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def read_csv_dicts(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path, rows, fieldnames):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ensure_dirs(output_dir):
    out = Path(output_dir)
    for name in ("contract", "metrics", "tables", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def normalize_token(value):
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    text = re.sub(r"\s+", "", text)
    return text


def normalize_image_key(value):
    text = normalize_token(value)
    text = text.split("?")[0]
    if "/" in text:
        text = text.split("/")[-1]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if text.endswith(ext):
            text = text[: -len(ext)]
            break
    return re.sub(r"[_\-]+", "_", text)


def _first(row, names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def canonical_pair_id_key(row):
    return normalize_token(row.get("canonical_pair_id"))


def source_target_pair_composite_key(row):
    source = _first(row, ["source_image_key", "source_image_a", "stress_source_image_a"])
    target = _first(row, ["target_image_key", "source_image_b", "stress_source_image_b"])
    pair = _first(row, ["pair_key", "source_pair_key", "stress_source_pair_key"])
    if not source or not target or not pair:
        return ""
    return "||".join([normalize_token(source), normalize_token(target), normalize_token(pair)])


def direction_invariant_source_target_pair_key(row):
    source = _first(row, ["source_image_key", "source_image_a", "stress_source_image_a"])
    target = _first(row, ["target_image_key", "source_image_b", "stress_source_image_b"])
    pair = _first(row, ["pair_key", "source_pair_key", "stress_source_pair_key"])
    if not source or not target:
        return ""
    a, b = sorted([normalize_token(source), normalize_token(target)])
    return "||".join([a, b, normalize_token(pair)])


def path_normalized_source_target_pair_key(row):
    source = _first(row, ["source_image_key", "source_image_a", "stress_source_image_a"])
    target = _first(row, ["target_image_key", "source_image_b", "stress_source_image_b"])
    pair = _first(row, ["pair_key", "source_pair_key", "stress_source_pair_key"])
    if not source or not target:
        return ""
    return "||".join([normalize_image_key(source), normalize_image_key(target), normalize_token(pair)])


def row_index_diagnostic_key(row):
    value = _first(row, ["source_row_index", "stress_source_row_index", "row_index"])
    return normalize_token(value)


def identity_key_strategies():
    return [
        {"name": "canonical_pair_id", "role": "promotion_key", "function": canonical_pair_id_key},
        {"name": "source_target_pair_composite", "role": "promotion_key_candidate", "function": source_target_pair_composite_key},
        {"name": "direction_invariant_source_target_pair", "role": "diagnostic_key", "function": direction_invariant_source_target_pair_key},
        {"name": "path_normalized_source_target_pair", "role": "diagnostic_key", "function": path_normalized_source_target_pair_key},
        {"name": "row_index_diagnostic_only", "role": "forbidden_for_promotion", "function": row_index_diagnostic_key},
    ]


def get_strategy(name):
    for strategy in identity_key_strategies():
        if strategy["name"] == name:
            return strategy
    raise KeyError(name)


def group_count(rows, field):
    counts = {}
    for row in rows:
        key = row.get(field, "") or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def compact_join(values, limit=12):
    uniq = []
    for value in values:
        text = str(value)
        if text and text not in uniq:
            uniq.append(text)
        if len(uniq) >= limit:
            break
    return "|".join(uniq)
