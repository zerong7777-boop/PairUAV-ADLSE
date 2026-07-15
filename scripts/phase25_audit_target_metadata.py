#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def stable_target_index(value, modulo):
    if int(modulo) <= 0:
        raise ValueError("num_target_groups must be positive")
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits) % int(modulo)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(modulo)


def scan_root(root, modulo):
    rows = []
    for path in sorted(Path(root).rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        group_id = str(raw.get("group_id", path.parent.name))
        json_id = str(raw.get("json_id", path.stem))
        scene_id = str(raw.get("scene_id", f"group_{group_id}"))
        rows.append(
            {
                "json_path": str(path),
                "group_id": group_id,
                "json_id": json_id,
                "scene_id": scene_id,
                "target_group_index": stable_target_index(group_id, modulo),
            }
        )
    return rows


def summarize(rows):
    groups = Counter(row["group_id"] for row in rows)
    indices = Counter(row["target_group_index"] for row in rows)
    return {
        "sample_count": len(rows),
        "unique_group_id_count": len(groups),
        "unique_target_group_index_count": len(indices),
        "largest_group_size": max(groups.values()) if groups else 0,
        "largest_index_bucket_size": max(indices.values()) if indices else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json-root", required=True)
    parser.add_argument("--val-json-root", required=True)
    parser.add_argument("--num-target-groups", type=int, default=4096)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = {
        "num_target_groups": args.num_target_groups,
        "train": summarize(scan_root(args.train_json_root, args.num_target_groups)),
        "val": summarize(scan_root(args.val_json_root, args.num_target_groups)),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
