#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch

from reloc3r.datasets.pairuav_matcher_features import sample_to_match_path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_ids(rows, split=None):
    ids = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        sample_id = row.get("sample_id")
        if sample_id:
            ids.append(str(sample_id))
    return ids


def dedupe(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def coverage(cache_root, ids):
    ids = dedupe(ids)
    missing = []
    matched = 0
    for sample_id in ids:
        path = sample_to_match_path(cache_root, sample_id)
        if path.is_file():
            matched += 1
        else:
            missing.append({"sample_id": sample_id, "expected_path": str(path)})
    total = len(ids)
    return {
        "rows": total,
        "matched": matched,
        "missing": total - matched,
        "coverage_rate": matched / total if total else 0.0,
        "missing_examples": missing[:20],
    }


def checkpoint_compatibility(path):
    result = {
        "checkpoint_path": str(Path(path)),
        "checkpoint_exists": Path(path).is_file(),
        "checkpoint_compatible": False,
        "top_level_keys": [],
        "state_key_examples": [],
        "note": "",
    }
    if not result["checkpoint_exists"]:
        result["note"] = "checkpoint_missing"
        return result
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        result["note"] = f"checkpoint_load_failed: {exc}"
        return result
    if isinstance(ckpt, dict):
        result["top_level_keys"] = sorted(str(k) for k in ckpt.keys())[:20]
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        if isinstance(state, dict):
            keys = sorted(str(k) for k in state.keys())
            result["state_key_examples"] = keys[:20]
            required_prefixes = ("patch_embed", "enc_blocks", "dec_blocks", "pose_head")
            result["checkpoint_compatible"] = any(key.startswith(required_prefixes) for key in keys)
            if not result["checkpoint_compatible"]:
                result["note"] = "no_reloc3r_key_prefix_detected"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--b1b-reference-record", default="")
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    subset_manifest = subset_root / "subset_manifest.csv"
    eval_manifest = subset_root / "eval_official_manifest.csv"
    train_rows = read_csv(subset_manifest)
    eval_rows = read_csv(eval_manifest)
    train_ids = sample_ids(train_rows, split="train")
    eval_ids = sample_ids(eval_rows)
    train_cov = coverage(Path(args.cache_root), train_ids)
    eval_cov = coverage(Path(args.cache_root), eval_ids)
    ckpt = checkpoint_compatibility(args.checkpoint)

    verdict = "usable"
    if eval_cov["coverage_rate"] < 0.95:
        verdict = "blocked"

    report = {
        "cache_root": str(Path(args.cache_root)),
        "subset_root": str(subset_root),
        "subset_manifest": str(subset_manifest),
        "eval_manifest": str(eval_manifest),
        "bounded_train_rows": len(train_ids),
        "bounded_eval_rows": len(eval_ids),
        "matched_train_rows": train_cov["matched"],
        "matched_eval_rows": eval_cov["matched"],
        "coverage_rate_train": train_cov["coverage_rate"],
        "coverage_rate_eval": eval_cov["coverage_rate"],
        "train_missing_examples": train_cov["missing_examples"],
        "eval_missing_examples": eval_cov["missing_examples"],
        "sample_id_join_rule": "<group>/<left>_<right> -> <cache_root>/<group>/image-<left>_image-<right>_matches.npz",
        "checkpoint_path": str(Path(args.checkpoint)),
        "checkpoint_compatible": bool(ckpt["checkpoint_compatible"]),
        "checkpoint_report": ckpt,
        "b1b_reference_record": args.b1b_reference_record,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
