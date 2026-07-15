#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


POLICY_COUNT_FIELDS = ("angle_policy_hard", "range_policy_hard", "easy_anchor")
WEIGHT_FIELDS = ("sample_weight", "angle_weight", "range_weight")


def as_bool_count(row, key):
    return int(float(row.get(key, 0.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=2048)
    args = parser.parse_args()

    path = Path(args.surface_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pair_ids = [row["pair_id"] for row in rows]
    duplicate_count = len(pair_ids) - len(set(pair_ids))
    counts = {key: sum(as_bool_count(row, key) for row in rows) for key in POLICY_COUNT_FIELDS}
    weights = {
        key: [min(float(row[key]) for row in rows), max(float(row[key]) for row in rows)]
        for key in WEIGHT_FIELDS
    } if rows else {key: [None, None] for key in WEIGHT_FIELDS}

    status = "ready"
    if len(rows) < args.min_rows or duplicate_count:
        status = "blocked"
    elif any(counts[key] == 0 for key in POLICY_COUNT_FIELDS):
        status = "weak_inconclusive"

    summary = {
        "surface_csv": str(path),
        "rows": len(rows),
        "unique_pair_ids": len(set(pair_ids)),
        "duplicate_count": duplicate_count,
        "counts": counts,
        "weights": weights,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": status,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if status in {"ready", "weak_inconclusive"} else 2)


if __name__ == "__main__":
    main()
