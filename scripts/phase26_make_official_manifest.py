#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from reloc3r.datasets.pairuav import PairUAV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    with Path(args.subset_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        subset_rows = [row for row in csv.DictReader(handle) if row.get("split") == "eval"]
    subset_by_id = {row["sample_id"]: row for row in subset_rows}

    dataset = PairUAV(
        json_root=args.json_root,
        image_root=args.image_root,
        split="dev",
        resolution=(512, 384),
        seed=777,
        require_labels=True,
    )
    out_rows = []
    missing = []
    for sample in dataset.samples:
        sample_id = f"{sample['group_id']}/{sample['json_id']}"
        if sample_id not in subset_by_id:
            missing.append(sample_id)
        out_rows.append(
            {
                "sample_id": sample_id,
                "gt_angle": sample["heading_deg"],
                "gt_distance": sample["range_value"],
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "gt_angle", "gt_distance"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        json.dumps(
            {
                "output_csv": str(output),
                "rows": len(out_rows),
                "subset_eval_rows": len(subset_rows),
                "missing_from_subset_manifest": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
