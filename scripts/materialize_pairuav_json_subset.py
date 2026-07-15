#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--output-json-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest_jsonl).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    out_root = Path(args.output_json_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        src = Path(row["json_path"])
        dst = out_root / row["group_id"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written += 1

    summary = {"written": written, "output_json_root": str(out_root)}
    (out_root.parent / "json_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
