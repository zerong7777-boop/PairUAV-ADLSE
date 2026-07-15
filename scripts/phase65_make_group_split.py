#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.phase64_token_shards import Phase64TokenShardDataset


def group_id_from_sample_id(sample_id: str) -> str:
    text = str(sample_id)
    if "/" not in text:
        raise ValueError(f"sample_id does not include group prefix: {sample_id!r}")
    group_id, _ = text.split("/", 1)
    if not group_id:
        raise ValueError(f"empty group_id in sample_id: {sample_id!r}")
    return group_id


def make_group_split(sample_ids: list[str], val_fraction: float, seed: int) -> dict:
    if not sample_ids:
        raise ValueError("sample_ids is empty")
    groups_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups_to_indices[group_id_from_sample_id(sample_id)].append(index)

    groups = sorted(groups_to_indices)
    rng = random.Random(int(seed))
    rng.shuffle(groups)

    target_val_rows = max(1, int(round(len(sample_ids) * float(val_fraction))))
    val_groups: list[str] = []
    val_rows = 0
    for group in groups:
        if val_rows >= target_val_rows and val_groups:
            break
        val_groups.append(group)
        val_rows += len(groups_to_indices[group])

    val_group_set = set(val_groups)
    train_groups = [group for group in sorted(groups_to_indices) if group not in val_group_set]
    train_indices = sorted(index for group in train_groups for index in groups_to_indices[group])
    val_indices = sorted(index for group in val_groups for index in groups_to_indices[group])

    overlap = sorted(set(train_groups).intersection(val_group_set))
    if overlap:
        raise AssertionError(f"group overlap: {overlap[:5]}")
    if not train_indices or not val_indices:
        raise ValueError("split produced empty train or val set")

    return {
        "format": "phase65_group_split_v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "rows": len(sample_ids),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "groups": len(groups_to_indices),
        "train_groups": train_groups,
        "val_groups": sorted(val_group_set),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "group_overlap": overlap,
        "first_train_sample_ids": [sample_ids[index] for index in train_indices[:10]],
        "first_val_sample_ids": [sample_ids[index] for index in val_indices[:10]],
    }


def sample_ids_from_manifest(manifest_path: Path) -> list[str]:
    dataset = Phase64TokenShardDataset(manifest_path, to_torch=False, preload=False)
    return [dataset[index]["sample_id"] for index in range(len(dataset))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=65065)
    args = parser.parse_args()

    sample_ids = sample_ids_from_manifest(Path(args.manifest))
    split = make_group_split(sample_ids, args.val_fraction, args.seed)
    split["manifest"] = str(Path(args.manifest))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: split[key] for key in [
        "format",
        "seed",
        "val_fraction",
        "rows",
        "train_rows",
        "val_rows",
        "groups",
        "group_overlap",
        "first_train_sample_ids",
        "first_val_sample_ids",
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
