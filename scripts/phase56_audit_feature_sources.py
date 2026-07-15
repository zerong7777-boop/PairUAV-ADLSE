#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit cheap matcher/BSCR feature manifests for Phase56.")
    parser.add_argument("--uavm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-json-root", type=Path, default=None)
    parser.add_argument("--val-json-root", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, action="append", default=[])
    parser.add_argument("--max-json-scan", type=int, default=300000)
    parser.add_argument("--max-manifest-rows", type=int, default=1000000)
    return parser


def sample_id_from_json(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def collect_json_ids(root: Path | None, max_rows: int) -> tuple[set[str], bool]:
    if root is None or not root.exists():
        return set(), False
    ids = set()
    truncated = False
    for path in sorted(root.rglob("*.json")):
        if len(ids) >= max_rows:
            truncated = True
            break
        ids.add(sample_id_from_json(path))
    return ids, truncated


def candidate_paths(root: Path) -> list[Path]:
    patterns = [
        "experiments/**/features/*.jsonl",
        "runs/**/features/**/*.jsonl",
        "runs/**/*features*.jsonl",
        "runs/**/*bscr*.jsonl",
        "runs/**/*matcher*.jsonl",
    ]
    found = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and any(token in path.name.lower() or token in str(path).lower() for token in ("feature", "matcher", "bscr")):
                found[str(path)] = path
    return [found[key] for key in sorted(found)]


def inspect_manifest(path: Path, train_ids: set[str], val_ids: set[str], max_rows: int) -> dict:
    rows = 0
    fallback = 0
    finite_like = 0
    sample_ids = set()
    feature_lens = {}
    errors = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if rows >= max_rows:
                break
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                errors += 1
                continue
            sid = str(row.get("sample_id", ""))
            if sid:
                sample_ids.add(sid)
            fallback += int(bool(row.get("fallback_used", False)))
            values = row.get("features", row.get("global_stats", []))
            if isinstance(values, list):
                feature_lens[str(len(values))] = feature_lens.get(str(len(values)), 0) + 1
                if all(isinstance(v, (int, float)) for v in values):
                    finite_like += 1

    train_overlap = len(sample_ids & train_ids) if train_ids else None
    val_overlap = len(sample_ids & val_ids) if val_ids else None
    return {
        "path": str(path),
        "rows_read": rows,
        "truncated": rows >= max_rows,
        "json_parse_errors": errors,
        "unique_sample_ids": len(sample_ids),
        "train_overlap_in_scanned_json_ids": train_overlap,
        "val_overlap_in_scanned_json_ids": val_overlap,
        "fallback_rows": fallback,
        "fallback_rate": float(fallback / rows) if rows else None,
        "finite_like_rows": finite_like,
        "feature_length_histogram": feature_lens,
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_root = args.train_json_root or args.uavm_root / "runs/devsplit_v1/train_json"
    val_root = args.val_json_root or args.uavm_root / "runs/devsplit_v1/val_json"
    train_ids, train_truncated = collect_json_ids(train_root, args.max_json_scan)
    val_ids, val_truncated = collect_json_ids(val_root, args.max_json_scan)

    paths = [path for path in args.candidate_path if path.is_file()] if args.candidate_path else candidate_paths(args.uavm_root)
    manifests = [
        inspect_manifest(path, train_ids=train_ids, val_ids=val_ids, max_rows=args.max_manifest_rows)
        for path in paths
    ]
    result = {
        "uavm_root": str(args.uavm_root),
        "train_json_root": str(train_root),
        "val_json_root": str(val_root),
        "train_json_ids_scanned": len(train_ids),
        "train_json_scan_truncated": train_truncated,
        "val_json_ids_scanned": len(val_ids),
        "val_json_scan_truncated": val_truncated,
        "manifest_count": len(manifests),
        "manifests": manifests,
    }
    out_path = args.output_dir / "g0_feature_sources.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_count": len(manifests), "output": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
