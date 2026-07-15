#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from reloc3r.datasets.pairuav_matcher_features import build_bscr_feature_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=16)
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_summary = build_bscr_feature_manifest(
        subset_root / "subset_manifest.csv",
        args.cache_root,
        output_root / "train_bscr_features.jsonl",
        stats_json=output_root / "train_bscr_summary.json",
        split="train",
        grid_size=args.grid_size,
        topk=args.topk,
    )
    eval_summary = build_bscr_feature_manifest(
        subset_root / "eval_official_manifest.csv",
        args.cache_root,
        output_root / "eval_bscr_features.jsonl",
        stats_json=output_root / "eval_bscr_summary.json",
        split=None,
        grid_size=args.grid_size,
        topk=args.topk,
    )
    summary = {
        "cache_root": str(Path(args.cache_root)),
        "subset_root": str(subset_root),
        "output_root": str(output_root),
        "train": train_summary,
        "eval": eval_summary,
        "verdict": "usable" if eval_summary["coverage_rate"] >= 0.95 else "blocked",
    }
    (output_root / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
