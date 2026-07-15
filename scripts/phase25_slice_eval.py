#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SLICE_NAMES = [
    "all",
    "target_heterogeneous",
    "heading_hard_range_easy",
    "range_hard_heading_easy",
    "middle_or_ordinary",
]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def phase13_sample_id(row):
    if row.get("sample_id"):
        return row["sample_id"]
    image_a = row.get("image_a", "")
    json_id = row.get("json_id", "")
    if "/" in image_a and json_id:
        return f"{image_a.split('/')[0]}/{json_id}"
    if row.get("group_id") and json_id:
        return f"{row['group_id']}/{json_id}"
    return None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(rows):
    return {
        "sample_count": len(rows),
        "distance_rel_error_mean": mean([as_float(row.get("distance_rel_error")) for row in rows]),
        "angle_rel_error_mean": mean([as_float(row.get("angle_rel_error")) for row in rows]),
        "final_score_mean": mean([as_float(row.get("final_score")) for row in rows]),
    }


def add_row(bucket, name, row):
    bucket[name].append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample-csv", required=True)
    parser.add_argument("--phase13-residual-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    per_sample = read_csv(args.per_sample_csv)
    phase13_rows = read_csv(args.phase13_residual_csv)
    phase13_by_id = {}
    for row in phase13_rows:
        sid = phase13_sample_id(row)
        if sid:
            phase13_by_id[sid] = row

    if not per_sample or "sample_id" not in per_sample[0]:
        payload = {"status": "blocked_identity_join", "reason": "per-sample CSV lacks sample_id"}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)

    buckets = defaultdict(list)
    joined = 0
    for row in per_sample:
        add_row(buckets, "all", row)
        residual = phase13_by_id.get(row["sample_id"])
        if residual is None:
            continue
        joined += 1
        label = residual.get("residual_target_dominance_label") or residual.get("target_dominance_label") or ""
        if label and label != "middle":
            add_row(buckets, "target_heterogeneous", row)
        if label == "heading_hard_range_easy":
            add_row(buckets, "heading_hard_range_easy", row)
        elif label == "range_hard_heading_easy":
            add_row(buckets, "range_hard_heading_easy", row)
        elif label == "middle" or not label:
            add_row(buckets, "middle_or_ordinary", row)

    summaries = []
    for name in SLICE_NAMES:
        item = {"slice": name}
        item.update(summarize(buckets[name]))
        summaries.append(item)

    payload = {
        "status": "ok",
        "per_sample_rows": len(per_sample),
        "phase13_rows": len(phase13_rows),
        "joined_phase13_rows": joined,
        "slices": summaries,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["slice", "sample_count", "distance_rel_error_mean", "angle_rel_error_mean", "final_score_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
