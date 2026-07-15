#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit PairUAV reverse-pair heading/range semantics.")
    parser.add_argument("--train-json-root", type=Path, required=True)
    parser.add_argument("--val-json-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files-per-split", type=int, default=250000)
    parser.add_argument("--max-reverse-examples", type=int, default=20000)
    return parser


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_abs_diff(a: float, b: float) -> float:
    return abs(wrap_deg(float(a) - float(b)))


def coerce_float(row: dict, *keys: str) -> float:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    raise KeyError(f"missing numeric key; tried {keys}")


def iter_json_paths(root: Path, max_files: int):
    count = 0
    for path in sorted(root.rglob("*.json")):
        if count >= max_files:
            return
        yield path
        count += 1


def parse_pair_id(path: Path):
    try:
        left, right = path.stem.split("_", 1)
    except ValueError:
        return None
    return left, right


def load_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        row = json.load(handle)
    return {
        "path": str(path),
        "group_id": str(row.get("group_id", path.parent.name)),
        "pair_id": str(row.get("json_id", path.stem)),
        "heading": coerce_float(row, "heading_deg", "heading_num"),
        "range": coerce_float(row, "range_value", "range_num"),
    }


def audit_split(root: Path, max_files: int, max_reverse_examples: int) -> dict:
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    scanned = 0
    for path in iter_json_paths(root, max_files=max_files):
        pair = parse_pair_id(path)
        if pair is None:
            continue
        try:
            record = load_record(path)
        except Exception:
            continue
        by_group[record["group_id"]][record["pair_id"]] = record
        scanned += 1

    reverse_examples = []
    for group_id, records in by_group.items():
        for pair_id, record in records.items():
            pair = parse_pair_id(Path(pair_id))
            if pair is None:
                continue
            left, right = pair
            reverse_id = f"{right}_{left}"
            reverse = records.get(reverse_id)
            if reverse is None:
                continue
            if str(pair_id) > reverse_id:
                continue
            reverse_examples.append((record, reverse))
            if len(reverse_examples) >= max_reverse_examples:
                break
        if len(reverse_examples) >= max_reverse_examples:
            break

    angle_errors = {
        "same": [],
        "neg_deg": [],
        "neg_vec_plus180": [],
    }
    range_errors = {
        "same": [],
        "neg": [],
    }
    sample_rows = []
    for forward, reverse in reverse_examples:
        h1 = forward["heading"]
        h2 = reverse["heading"]
        r1 = forward["range"]
        r2 = reverse["range"]
        angle_errors["same"].append(angle_abs_diff(h2, h1))
        angle_errors["neg_deg"].append(angle_abs_diff(h2, -h1))
        angle_errors["neg_vec_plus180"].append(angle_abs_diff(h2, wrap_deg(h1 + 180.0)))
        range_errors["same"].append(abs(r2 - r1))
        range_errors["neg"].append(abs(r2 + r1))
        if len(sample_rows) < 20:
            sample_rows.append(
                {
                    "forward_pair": forward["pair_id"],
                    "reverse_pair": reverse["pair_id"],
                    "heading_forward": h1,
                    "heading_reverse": h2,
                    "range_forward": r1,
                    "range_reverse": r2,
                }
            )

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "mean_abs_error": None, "max_abs_error": None, "near_zero_count": 0}
        return {
            "count": len(values),
            "mean_abs_error": float(mean(values)),
            "max_abs_error": float(max(values)),
            "near_zero_count": int(sum(v < 1e-4 for v in values)),
        }

    angle_summary = {key: summarize(values) for key, values in angle_errors.items()}
    range_summary = {key: summarize(values) for key, values in range_errors.items()}
    best_angle = min(
        (key for key, item in angle_summary.items() if item["count"] > 0),
        key=lambda key: angle_summary[key]["mean_abs_error"],
        default=None,
    )
    best_range = min(
        (key for key, item in range_summary.items() if item["count"] > 0),
        key=lambda key: range_summary[key]["mean_abs_error"],
        default=None,
    )
    return {
        "root": str(root),
        "max_files_per_split": max_files,
        "scanned_records": scanned,
        "truncated": scanned >= max_files,
        "groups_scanned": len(by_group),
        "reverse_pair_examples": len(reverse_examples),
        "angle_candidate_summary": angle_summary,
        "range_candidate_summary": range_summary,
        "recommended_inverse_heading_policy": best_angle,
        "recommended_inverse_range_policy": best_range,
        "sample_rows": sample_rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "train": audit_split(args.train_json_root, args.max_files_per_split, args.max_reverse_examples),
        "val": audit_split(args.val_json_root, args.max_files_per_split, args.max_reverse_examples),
    }
    result["recommended_policy"] = {
        "inverse_heading_policy": result["train"]["recommended_inverse_heading_policy"],
        "inverse_range_policy": result["train"]["recommended_inverse_range_policy"],
        "source": "train split reverse-pair audit",
    }
    out_path = args.output_dir / "g0_inverse_semantics.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = args.output_dir / "README.md"
    readme.write_text(
        "# Phase56 G0 Audit\n\n"
        f"- inverse semantics: `{out_path}`\n"
        f"- recommended heading policy: `{result['recommended_policy']['inverse_heading_policy']}`\n"
        f"- recommended range policy: `{result['recommended_policy']['inverse_range_policy']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(result["recommended_policy"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
