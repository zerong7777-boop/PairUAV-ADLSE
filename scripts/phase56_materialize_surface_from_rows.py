#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a PairUAV train/val JSON surface from a rows.jsonl file.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--source-json-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def copy_row(row: dict, source_json_root: Path, split_root: Path) -> dict:
    pair_id = str(row["pair_id"])
    group_id, json_stem = pair_id.split("/", 1)
    src = source_json_root / group_id / f"{json_stem}.json"
    if not src.is_file():
        raise FileNotFoundError(f"missing source json for {pair_id}: {src}")
    dst = split_root / group_id / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "pair_id": pair_id,
        "group_id": group_id,
        "json_id": json_stem,
        "split": row["split"],
        "source_json": str(src),
        "materialized_json": str(dst),
        "target_heading": row.get("target_heading"),
        "target_distance": row.get("target_distance"),
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_root = args.output_root / "train_json"
    val_root = args.output_root / "val_json"
    manifest_path = args.output_root / "surface_manifest.jsonl"

    counts = {"train": 0, "val": 0}
    duplicate_ids = set()
    seen_ids = set()
    rows_written = 0
    with args.rows_jsonl.open("r", encoding="utf-8") as src_rows, manifest_path.open("w", encoding="utf-8") as out:
        for raw in src_rows:
            if args.limit and rows_written >= args.limit:
                break
            if not raw.strip():
                continue
            row = json.loads(raw)
            split = str(row.get("split", ""))
            if split not in counts:
                continue
            pair_id = str(row["pair_id"])
            if pair_id in seen_ids:
                duplicate_ids.add(pair_id)
            seen_ids.add(pair_id)
            split_root = train_root if split == "train" else val_root
            record = copy_row(row, args.source_json_root, split_root)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
            rows_written += 1

    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "source_json_root": str(args.source_json_root),
        "output_root": str(args.output_root),
        "train_json_root": str(train_root),
        "val_json_root": str(val_root),
        "counts": counts,
        "rows_written": rows_written,
        "duplicate_pair_id_count": len(duplicate_ids),
        "duplicate_pair_ids": sorted(duplicate_ids)[:50],
        "manifest": str(manifest_path),
    }
    summary_path = args.output_root / "surface_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
