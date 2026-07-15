#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SLICE_NAMES = [
    "all",
    "phase11_semantic_decoupled",
    "phase11_control_low_sim_low_error",
    "phase11_control_other",
    "phase14_alignment_sensitive",
    "phase14_alignment_control_middle",
]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_row(row):
    if row.get("sample_id"):
        return row["sample_id"]
    image_a = row.get("image_a", "")
    json_id = row.get("json_id", "")
    if "/" in image_a and json_id:
        return f"{image_a.split('/')[0]}/{json_id}"
    return None


def boolish(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    values = [value for value in values if value is not None]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample-csv", required=True)
    parser.add_argument("--phase11-controlled-csv", required=True)
    parser.add_argument("--phase14-surface-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    per_sample = read_csv(args.per_sample_csv)
    phase11 = {sample_id_from_row(row): row for row in read_csv(args.phase11_controlled_csv) if sample_id_from_row(row)}
    phase14 = {sample_id_from_row(row): row for row in read_csv(args.phase14_surface_csv) if sample_id_from_row(row)}

    buckets = defaultdict(list)
    joined_phase11 = 0
    joined_phase14 = 0
    for row in per_sample:
        sid = row.get("sample_id")
        buckets["all"].append(row)
        p11 = phase11.get(sid)
        if p11 is not None:
            joined_phase11 += 1
            if boolish(p11.get("high_sim_high_error", "0")):
                buckets["phase11_semantic_decoupled"].append(row)
            elif boolish(p11.get("low_sim_low_error", "0")):
                buckets["phase11_control_low_sim_low_error"].append(row)
            else:
                buckets["phase11_control_other"].append(row)
        p14 = phase14.get(sid)
        if p14 is not None:
            joined_phase14 += 1
            label = p14.get("residual_target_dominance_label", "")
            if label and label != "middle":
                buckets["phase14_alignment_sensitive"].append(row)
            elif label == "middle":
                buckets["phase14_alignment_control_middle"].append(row)

    summaries = []
    for name in SLICE_NAMES:
        item = {"slice": name}
        item.update(summarize(buckets[name]))
        summaries.append(item)

    payload = {
        "status": "ok",
        "per_sample_rows": len(per_sample),
        "joined_phase11_rows": joined_phase11,
        "joined_phase14_rows": joined_phase14,
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
