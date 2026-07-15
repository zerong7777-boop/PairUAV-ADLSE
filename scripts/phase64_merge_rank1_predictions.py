#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_rows(paths):
    seen = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pair_id = row.get("pair_id")
                if pair_id and pair_id not in seen:
                    seen[pair_id] = row
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No rows found")
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = {
        "output_csv": str(output),
        "rows": len(rows),
        "inputs": [str(Path(path)) for path in args.inputs],
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
