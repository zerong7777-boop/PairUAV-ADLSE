"""Common helpers for Phase27 A-v3.1 shared outcome surface audits."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
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


def truthy(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "joined"}


def to_float(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def safe_div(num, den):
    try:
        den_f = float(den)
    except (TypeError, ValueError):
        return 0.0
    if den_f == 0:
        return 0.0
    return float(num) / den_f


def quantiles(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    def q(prob):
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * prob
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return round(vals[lo] * (1 - frac) + vals[hi] * frac, 10)

    return {"p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99)}


def stable_pair_key(row):
    canonical = str(row.get("canonical_pair_id", "")).strip()
    if canonical:
        return "canonical_pair_id", canonical
    source = str(row.get("source_image_key", "")).strip()
    target = str(row.get("target_image_key", "")).strip()
    pair = str(row.get("pair_key", "")).strip()
    if source and target and pair:
        return "source_target_pair_composite", f"{source}::{target}::{pair}"
    fallback = str(row.get("pair_id") or row.get("pair_key") or "").strip()
    return "fallback_pair_id", fallback


def detect_duplicate_keys(rows, key_fn=stable_pair_key):
    grouped = defaultdict(list)
    for row in rows:
        _, key = key_fn(row)
        if key:
            grouped[key].append(row)
    return {key: value for key, value in grouped.items() if len(value) > 1}


def group_count(rows, field):
    counts = {}
    for row in rows:
        key = row.get(field, "") or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def ensure_output_dirs(output_dir):
    out = Path(output_dir)
    for name in ("manifests", "metrics", "tables", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out
