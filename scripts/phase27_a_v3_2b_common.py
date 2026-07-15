"""Common utilities for Phase27 A-v3.2b fixed-manifest audits."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def ensure_dirs(output_dir):
    out = Path(output_dir)
    for name in ("manifests", "tables", "metrics", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def read_csv_dicts(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path, rows, fieldnames):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_token(value):
    value = "" if value is None else str(value)
    value = value.strip().replace("\\", "/").lower()
    value = re.sub(r"/+", "/", value)
    return value


def normalize_image_key(value):
    value = normalize_token(value)
    for suffix in (".jpeg", ".jpg", ".png"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def canonical_pair_key(row):
    return normalize_token(row.get("canonical_pair_id", ""))


def source_target_composite_key(row):
    source = row.get("source_image_key") or row.get("source_image_a") or row.get("source_key")
    target = row.get("target_image_key") or row.get("source_image_b") or row.get("target_key")
    source = normalize_image_key(source)
    target = normalize_image_key(target)
    if not source or not target:
        return ""
    return f"{source}|{target}"


def sha256_rows(rows, columns):
    h = hashlib.sha256()
    for row in rows:
        h.update("\t".join(str(row.get(c, "")) for c in columns).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def group_by_key(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def classify_duplicate_groups(rows, key_fn):
    groups = group_by_key(rows, key_fn)
    missing = groups.pop("", [])
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    unique = {k: v[0] for k, v in groups.items() if len(v) == 1}
    return {"unique": unique, "duplicates": duplicates, "missing": missing}


def safe_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def percentile(values, q):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


def p90(values):
    return percentile(values, 0.90)


def p95(values):
    return percentile(values, 0.95)

