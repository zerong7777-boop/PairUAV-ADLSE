#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SOURCE_FIELDS = [
    "roma_mean",
    "roma_certainty_mean",
    "vggt_mean",
    "vggt_conf_mean",
    "mast3r_mean",
    "mast3r_conf_mean",
]


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv_by_pair(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def wrapped_angle_error(pred, target):
    diff = (float(pred) - float(target) + 180.0) % 360.0 - 180.0
    return abs(diff)


def f(row, key, default=0.0):
    if row is None:
        return float(default)
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--source-stats-csv", required=True)
    parser.add_argument("--anchor-prediction-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest_jsonl)
    source = read_csv_by_pair(args.source_stats_csv)
    anchor = read_csv_by_pair(args.anchor_prediction_csv)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair_id",
        "split",
        "group_id",
        "heading_deg",
        "range_value",
        "anchor_angle_error",
        "anchor_range_error",
        *SOURCE_FIELDS,
        "source_disagreement",
        "angle_policy_hard",
        "range_policy_hard",
        "easy_anchor",
        "sample_weight",
        "angle_weight",
        "range_weight",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            pair_id = row["pair_id"]
            source_row = source.get(pair_id, {})
            anchor_row = anchor.get(pair_id, {})
            target_heading = float(row["heading_deg"])
            target_range = float(row["range_value"])
            anchor_angle = wrapped_angle_error(f(anchor_row, "pred_heading_deg"), target_heading)
            anchor_range = abs(f(anchor_row, "pred_range_value") - target_range)
            disagreement = abs(f(source_row, "roma_mean") - f(source_row, "vggt_mean"))
            disagreement += abs(f(source_row, "mast3r_mean") - f(source_row, "vggt_mean"))
            # Source disagreement is logged as a separate policy signal.
            # Do not make it a hard-label by itself: source magnitudes are not normalized
            # across RoMa/VGGT/MASt3R, and the cached-hybrid surface can otherwise mark
            # every sample as hard, removing the easy-anchor control slice.
            angle_hard = int(anchor_angle >= 5.0)
            range_hard = int(anchor_range >= max(2.0, 0.05 * abs(target_range)))
            easy_anchor = int(angle_hard == 0 and range_hard == 0)
            angle_weight = min(2.5, 1.0 + anchor_angle / 20.0)
            range_weight = min(2.5, 1.0 + anchor_range / max(10.0, abs(target_range) * 0.2))
            sample_weight = min(2.0, 0.75 + 0.25 * angle_weight + 0.25 * range_weight)
            if easy_anchor:
                sample_weight = min(sample_weight, 1.0)
            output_row = {
                "pair_id": pair_id,
                "split": row.get("split", "train"),
                "group_id": row.get("group_id", ""),
                "heading_deg": row["heading_deg"],
                "range_value": row["range_value"],
                "anchor_angle_error": anchor_angle,
                "anchor_range_error": anchor_range,
                "source_disagreement": disagreement,
                "angle_policy_hard": angle_hard,
                "range_policy_hard": range_hard,
                "easy_anchor": easy_anchor,
                "sample_weight": sample_weight,
                "angle_weight": angle_weight,
                "range_weight": range_weight,
            }
            for key in SOURCE_FIELDS:
                output_row[key] = f(source_row, key)
            writer.writerow(output_row)
    print(json.dumps({"output_csv": str(out), "rows": len(manifest)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
