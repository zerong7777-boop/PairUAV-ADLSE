#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_to_match_path(cache_root, sample_id):
    group, pair_id = str(sample_id).split("/", 1)
    left, right = pair_id.split("_", 1)
    return Path(cache_root) / group / f"image-{left}_image-{right}_matches.npz"


def inspect_npz(path):
    data = np.load(path, allow_pickle=True)
    fields = {}
    for key in data.files:
        value = data[key]
        fields[key] = {
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")),
        }
    return fields


def sample_packet_files(packet_root):
    root = Path(packet_root)
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.glob("*")) if path.is_file()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--bounded-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-inspect", type=int, default=8)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in cache_root.rglob("*") if path.is_file())
    suffix_counts = Counter(path.suffix for path in files)
    npz_files = [path for path in files if path.suffix == ".npz"]

    inspected = {}
    available_fields = set()
    for path in npz_files[: args.max_inspect]:
        fields = inspect_npz(path)
        inspected[str(path)] = fields
        available_fields.update(fields)

    manifest_rows = read_csv(args.bounded_manifest)
    missing = []
    matched = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        match_path = sample_to_match_path(cache_root, sample_id)
        if match_path.is_file():
            matched.append(sample_id)
        else:
            missing.append({"sample_id": sample_id, "expected_path": str(match_path)})

    bounded_eval_rows = len(manifest_rows)
    matched_eval_rows = len(matched)
    coverage_rate = matched_eval_rows / bounded_eval_rows if bounded_eval_rows else 0.0
    required_npz_fields = {"keypoints0", "keypoints1", "matches", "match_confidence"}
    schema_usable = bool(npz_files) and required_npz_fields.issubset(available_fields) and coverage_rate >= 0.95

    report = {
        "cache_root": str(cache_root),
        "packet_root": str(Path(args.packet_root)),
        "inspected_files": len(inspected),
        "total_cache_files": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "detected_format": "npz" if npz_files and len(suffix_counts) == 1 else "mixed",
        "available_fields": sorted(available_fields),
        "required_npz_fields": sorted(required_npz_fields),
        "sample_id_join_rule": "<group>/<left>_<right> -> <cache_root>/<group>/image-<left>_image-<right>_matches.npz",
        "bounded_manifest": str(Path(args.bounded_manifest)),
        "bounded_eval_rows": bounded_eval_rows,
        "matched_eval_rows": matched_eval_rows,
        "coverage_rate": coverage_rate,
        "missing_examples": missing[:20],
        "inspected_schema": inspected,
        "packet_files": sample_packet_files(args.packet_root)[:20],
        "schema_verdict": "usable" if schema_usable else "blocked",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
