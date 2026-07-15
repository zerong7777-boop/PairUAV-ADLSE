#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_id(path, row):
    group_id = str(row.get("group_id", path.parent.name))
    json_id = str(row.get("json_id", path.stem))
    return f"{group_id}/{json_id}"


def numeric(row, names, default=0.0):
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    return float(default)


def load_rows(root, split):
    rows = []
    for path in sorted(Path(root).rglob("*.json")):
        row = read_json(path)
        pid = pair_id(path, row)
        heading = numeric(row, ("heading_deg", "heading_num", "heading", "angle"))
        distance = numeric(row, ("range_value", "range_num", "range", "distance"))
        group = pid.split("/", 1)[0]
        rows.append(
            {
                "pair_id": pid,
                "split": split,
                "group_id": group,
                "json_path": str(path),
                "heading_deg": heading,
                "range_value": distance,
                "heading_bin": int(heading // 45) % 8,
                "range_bin": int(min(9, max(0, distance // 20))),
            }
        )
    return rows


def stable_bucket(row):
    return (row["group_id"], row["heading_bin"], row["range_bin"])


def select_balanced(rows, target_rows, seed):
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[stable_bucket(row)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    selected = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < target_rows and keys:
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop())
        else:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json-root", required=True)
    parser.add_argument("--val-json-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--target-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    train_rows = load_rows(args.train_json_root, "train")
    val_ids = {row["pair_id"] for row in load_rows(args.val_json_root, "val")}
    train_candidates = [row for row in train_rows if row["pair_id"] not in val_ids]
    selected = select_balanced(train_candidates, args.target_rows, args.seed)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    selected_ids = {row["pair_id"] for row in selected}
    group_counts = collections.Counter(row["group_id"] for row in selected)
    summary = {
        "rows": len(selected),
        "target_rows": args.target_rows,
        "train_total_rows": len(train_rows),
        "train_candidate_rows": len(train_candidates),
        "group_count": len(group_counts),
        "val_overlap": len(selected_ids & val_ids),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "output_jsonl": str(out),
        "status": "ready"
        if len(selected) >= min(args.target_rows, len(train_candidates)) and not (selected_ids & val_ids)
        else "check",
    }
    Path(args.audit_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
