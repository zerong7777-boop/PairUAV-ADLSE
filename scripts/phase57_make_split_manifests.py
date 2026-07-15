#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_surface_row(row: dict[str, Any]) -> dict[str, Any]:
    pair_id = str(row.get("pair_id", "")).strip()
    if not pair_id:
        raise ValueError(f"surface row missing pair_id: {row}")
    group_id = str(row.get("group_id") or pair_id.split("/", 1)[0]).strip()
    json_path = str(row.get("materialized_json") or row.get("json_path") or row.get("source_json") or "").strip()
    if not json_path:
        raise ValueError(f"surface row missing JSON path for {pair_id}")
    source_split = str(row.get("split") or row.get("source_split") or "").strip()
    if source_split not in {"train", "val"}:
        raise ValueError(f"surface row has unsupported split for {pair_id}: {source_split!r}")
    return {
        "pair_id": pair_id,
        "group_id": group_id,
        "json_path": json_path,
        "source_json": str(row.get("source_json") or json_path),
        "source_split": source_split,
        "target_heading": row.get("target_heading"),
        "target_distance": row.get("target_distance"),
    }


def assign_group_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[str(row["group_id"])] = group_counts.get(str(row["group_id"]), 0) + 1
    ordered_groups = sorted(group_counts, key=lambda group: stable_digest(f"{seed}:{group}"))
    fold_loads = [0 for _ in range(folds)]
    assignments: dict[str, int] = {}
    for group in ordered_groups:
        fold_id = min(range(folds), key=lambda idx: (fold_loads[idx], idx))
        assignments[group] = fold_id
        fold_loads[fold_id] += group_counts[group]
    return assignments


def _with_role(row: dict[str, Any], *, role: str, fold_id: int | None) -> dict[str, Any]:
    out = dict(row)
    out["role"] = role
    out["fold_id"] = fold_id
    return out


def _duplicate_pair_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in seen:
            dupes.add(pair_id)
        seen.add(pair_id)
    return sorted(dupes)


def build_split_manifests(rows: list[dict[str, Any]], folds: int, seed: int) -> dict[str, Any]:
    normalized = [normalize_surface_row(row) for row in rows]
    train_rows = [row for row in normalized if row["source_split"] == "train"]
    fixed_val_rows = [row for row in normalized if row["source_split"] == "val"]
    assignments = assign_group_folds(train_rows, folds=folds, seed=seed)

    fold_rows: dict[str, list[dict[str, Any]]] = {}
    fold_counts: list[dict[str, Any]] = []
    for fold_id in range(folds):
        calib = [_with_role(row, role="calib", fold_id=fold_id) for row in train_rows if assignments[row["group_id"]] != fold_id]
        holdout = [_with_role(row, role="holdout", fold_id=fold_id) for row in train_rows if assignments[row["group_id"]] == fold_id]
        fold_rows[f"cv_fold_{fold_id:02d}_calib.jsonl"] = calib
        fold_rows[f"cv_fold_{fold_id:02d}_holdout.jsonl"] = holdout
        fold_counts.append(
            {
                "fold_id": fold_id,
                "calib_rows": len(calib),
                "holdout_rows": len(holdout),
                "holdout_groups": len({row["group_id"] for row in holdout}),
                "calib_groups": len({row["group_id"] for row in calib}),
            }
        )

    all_labeled = [_with_role(row, role="all", fold_id=None) for row in normalized]
    fixed_val = [_with_role(row, role="fixed_val", fold_id=None) for row in fixed_val_rows]
    duplicate_ids = _duplicate_pair_ids(normalized)

    return {
        "files": {
            "all_labeled.jsonl": all_labeled,
            "fixed_val811.jsonl": fixed_val,
            **fold_rows,
        },
        "summary": {
            "folds": folds,
            "seed": seed,
            "all_labeled_rows": len(all_labeled),
            "train_rows": len(train_rows),
            "fixed_val_rows": len(fixed_val),
            "unique_pair_id_count": len({row["pair_id"] for row in normalized}),
            "duplicate_pair_id_count": len(duplicate_ids),
            "duplicate_pair_ids": duplicate_ids[:50],
            "unique_group_count": len({row["group_id"] for row in normalized}),
            "train_group_count": len({row["group_id"] for row in train_rows}),
            "fixed_val_group_count": len({row["group_id"] for row in fixed_val_rows}),
            "fold_counts": fold_counts,
        },
    }


def write_split_manifests(bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = bundle["files"]
    written: dict[str, int] = {}
    for filename, rows in files.items():
        write_jsonl(output_dir / filename, rows)
        written[filename] = len(rows)
    summary = dict(bundle["summary"])
    summary["output_dir"] = str(output_dir)
    summary["written_files"] = written
    write_json(output_dir / "split_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase57 labeled split manifests from a PairUAV surface manifest.")
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=57)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_split_manifests(read_jsonl(args.surface_manifest), folds=args.folds, seed=args.seed)
    summary = write_split_manifests(bundle, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
