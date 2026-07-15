#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


REQUIRED = {"pair_id", "split", "bucket_tags", "angle_weight", "range_weight", "sample_weight", "policy_version"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    packet = Path(args.packet_jsonl)
    rows = []
    pair_ids = set()
    duplicates = []
    bucket_counts = collections.Counter()
    split_counts = collections.Counter()
    weight_minmax = {
        "sample_weight": [float("inf"), float("-inf")],
        "angle_weight": [float("inf"), float("-inf")],
        "range_weight": [float("inf"), float("-inf")],
    }

    for line_no, line in enumerate(packet.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        pair_id = row["pair_id"]
        if pair_id in pair_ids:
            duplicates.append(pair_id)
        pair_ids.add(pair_id)
        rows.append((line_no, row))

    if duplicates:
        print(f"duplicate pair_id: {duplicates[0]}", file=sys.stderr)
        return 3
    if not rows:
        print("empty policy packet", file=sys.stderr)
        return 4

    validated_rows = []
    for line_no, row in rows:
        missing = REQUIRED - set(row)
        if missing:
            print(f"line {line_no} missing fields: {sorted(missing)}", file=sys.stderr)
            return 2
        split_counts[row["split"]] += 1
        for tag in row["bucket_tags"]:
            bucket_counts[tag] += 1
        for key in weight_minmax:
            value = float(row[key])
            weight_minmax[key][0] = min(weight_minmax[key][0], value)
            weight_minmax[key][1] = max(weight_minmax[key][1], value)
        validated_rows.append(row)
    for key, (lo, hi) in weight_minmax.items():
        if lo < 0.0 or hi > 3.0:
            print(f"{key} out of allowed audit range: {lo}..{hi}", file=sys.stderr)
            return 5

    summary = {
        "packet_jsonl": str(packet),
        "rows": len(validated_rows),
        "unique_pair_ids": len(pair_ids),
        "split_counts": dict(split_counts),
        "bucket_counts": dict(bucket_counts),
        "weight_minmax": weight_minmax,
        "sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "status": "ready",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
