#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def candidate_rows(json_root):
    json_root = Path(json_root)
    for path in json_root.rglob("*.json"):
        if not path.is_file():
            continue
        group_id = path.parent.name
        pair_id = f"{group_id}/{path.stem}"
        yield {
            "pair_id": pair_id,
            "group_id": group_id,
            "json_path": str(path),
        }


def stable_key(pair_id, seed):
    text = f"{seed}:{pair_id}".encode("utf-8")
    return hashlib.sha1(text).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=64064)
    args = parser.parse_args()

    rows = sorted(candidate_rows(args.json_root), key=lambda row: stable_key(row["pair_id"], args.seed))
    selected = rows[: int(args.limit)]
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    groups = sorted({row["group_id"] for row in selected})
    summary = {
        "json_root": str(Path(args.json_root)),
        "output_jsonl": str(output),
        "rows_seen": len(rows),
        "rows_selected": len(selected),
        "limit": int(args.limit),
        "seed": int(args.seed),
        "groups_selected": len(groups),
        "first_pair_ids": [row["pair_id"] for row in selected[:10]],
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
