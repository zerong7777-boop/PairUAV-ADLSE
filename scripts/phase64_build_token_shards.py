#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def _load_tokens_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "datasets" / "pairuav_correspondence_tokens.py"
    spec = importlib.util.spec_from_file_location("pairuav_correspondence_tokens_phase64", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOKENS = _load_tokens_module()


def read_prediction_rows(path, split=None, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is not None and row.get("split") != split:
                continue
            if not row.get("pair_id"):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def train_match_path(cache_root, pair_id):
    return TOKENS.sample_to_train_match_path(cache_root, pair_id)


def build_arrays(rows, cache_root, image_size, topk, grid_size, residual_threshold):
    sample_ids = []
    target_heading = []
    target_distance = []
    rank1_heading = []
    rank1_distance = []
    rank1_angle_abs_error = []
    tokens = []
    token_mask = []
    hypothesis_features = []
    global_stats = []
    fallback_used = []
    valid_matches = []
    total_matches = []
    match_paths = []

    for row in rows:
        pair_id = str(row["pair_id"])
        match_path = train_match_path(cache_root, pair_id)
        packet = TOKENS.build_correspondence_token_packet(
            match_path,
            image_size=image_size,
            topk=topk,
            grid_size=grid_size,
            residual_threshold=residual_threshold,
        )
        sample_ids.append(pair_id)
        target_heading.append(f(row, "target_heading"))
        target_distance.append(f(row, "target_distance"))
        rank1_heading.append(f(row, "rank1_heading"))
        rank1_distance.append(f(row, "rank1_distance"))
        rank1_angle_abs_error.append(f(row, "rank1_angle_abs_error"))
        tokens.append(np.asarray(packet["tokens"], dtype=np.float32))
        token_mask.append(np.asarray(packet["token_mask"], dtype=np.float32))
        hypothesis_features.append(np.asarray(packet["hypothesis_features"], dtype=np.float32))
        global_stats.append(np.asarray(packet["global_stats"], dtype=np.float32))
        fallback_used.append(float(packet["fallback_used"]))
        valid_matches.append(int(packet["raw_counts"].get("valid_matches", 0)))
        total_matches.append(int(packet["raw_counts"].get("total_matches", 0)))
        match_paths.append(str(match_path))

    return {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "target_heading": np.asarray(target_heading, dtype=np.float32),
        "target_distance": np.asarray(target_distance, dtype=np.float32),
        "rank1_heading": np.asarray(rank1_heading, dtype=np.float32),
        "rank1_distance": np.asarray(rank1_distance, dtype=np.float32),
        "rank1_angle_abs_error": np.asarray(rank1_angle_abs_error, dtype=np.float32),
        "tokens": np.stack(tokens, axis=0).astype(np.float32) if tokens else np.zeros((0, topk, len(TOKENS.TOKEN_FEATURE_NAMES)), dtype=np.float32),
        "token_mask": np.stack(token_mask, axis=0).astype(np.float32) if token_mask else np.zeros((0, topk), dtype=np.float32),
        "hypothesis_features": np.stack(hypothesis_features, axis=0).astype(np.float32)
        if hypothesis_features
        else np.zeros((0, len(TOKENS.HYPOTHESIS_NAMES), len(TOKENS.HYPOTHESIS_FEATURE_NAMES)), dtype=np.float32),
        "global_stats": np.stack(global_stats, axis=0).astype(np.float32)
        if global_stats
        else np.zeros((0, len(TOKENS.GLOBAL_FEATURE_NAMES)), dtype=np.float32),
        "fallback_used": np.asarray(fallback_used, dtype=np.float32),
        "valid_matches": np.asarray(valid_matches, dtype=np.int32),
        "total_matches": np.asarray(total_matches, dtype=np.int32),
        "match_path": np.asarray(match_paths, dtype=object),
    }


def write_shard(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def shard_rows(rows, shard_size):
    shard_size = int(shard_size)
    for start in range(0, len(rows), shard_size):
        yield start, rows[start : start + shard_size]


def manifest_summary(shards, rows, args):
    valid_counts = []
    fallback = 0
    for shard in shards:
        valid_counts.extend(shard["valid_matches_summary"])
        fallback += int(shard["fallback"])
    covered = len(rows) - fallback
    return {
        "format": "phase64_token_shards_v1",
        "source_predictions": str(Path(args.predictions_csv)),
        "cache_root": str(Path(args.cache_root)),
        "output_root": str(Path(args.output_root)),
        "rows": len(rows),
        "covered": int(covered),
        "fallback": int(fallback),
        "coverage_rate": float(covered / len(rows)) if rows else 0.0,
        "shard_size": int(args.shard_size),
        "num_shards": len(shards),
        "topk": int(args.topk),
        "grid_size": int(args.grid_size),
        "image_size": [int(args.image_width), int(args.image_height)],
        "residual_threshold": float(args.residual_threshold),
        "token_feature_names": list(TOKENS.TOKEN_FEATURE_NAMES),
        "hypothesis_names": list(TOKENS.HYPOTHESIS_NAMES),
        "hypothesis_feature_names": list(TOKENS.HYPOTHESIS_FEATURE_NAMES),
        "global_feature_names": list(TOKENS.GLOBAL_FEATURE_NAMES),
        "valid_matches_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "valid_matches_p50": float(np.percentile(valid_counts, 50)) if valid_counts else 0.0,
        "valid_matches_p10": float(np.percentile(valid_counts, 10)) if valid_counts else 0.0,
        "shards": shards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--residual-threshold", type=float, default=0.035)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prediction_rows(args.predictions_csv, split=args.split, limit=args.limit)
    shards = []
    for shard_index, (start, shard_input_rows) in enumerate(shard_rows(rows, args.shard_size)):
        arrays = build_arrays(
            shard_input_rows,
            cache_root=args.cache_root,
            image_size=(args.image_width, args.image_height),
            topk=args.topk,
            grid_size=args.grid_size,
            residual_threshold=args.residual_threshold,
        )
        shard_name = f"shard_{shard_index:06d}.npz"
        shard_path = shard_dir / shard_name
        write_shard(shard_path, arrays)
        fallback = int(np.asarray(arrays["fallback_used"] >= 0.5, dtype=np.int32).sum())
        valid_summary = [int(x) for x in arrays["valid_matches"].tolist()]
        shards.append(
            {
                "path": str(shard_path),
                "name": shard_name,
                "start": int(start),
                "rows": int(len(shard_input_rows)),
                "covered": int(len(shard_input_rows) - fallback),
                "fallback": int(fallback),
                "valid_matches_summary": valid_summary,
            }
        )

    manifest = manifest_summary(shards, rows, args)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = dict(manifest)
    compact["shards"] = [
        {key: value for key, value in shard.items() if key != "valid_matches_summary"} for shard in manifest["shards"]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
