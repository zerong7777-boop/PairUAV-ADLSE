#!/usr/bin/env python3
"""Inventory Phase27 A evidence-state manifest inputs.

This script is read-only with respect to project inputs. It samples candidate
CSV/JSONL files, classifies columns for construction versus validation, and
writes a compact inventory JSON for the manifest implementation phase.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "phase27_a_input_inventory_v1"

FORBIDDEN_PATTERNS = [
    "angle_err",
    "range_err",
    "combined_error",
    "error",
    "residual",
    "score_from_official",
    "official",
    "leaderboard",
    "phase11_slice",
    "phase13_slice",
    "phase14_slice",
    "slice_label",
]

SAFE_IDENTITY_PATTERNS = [
    "pair",
    "query",
    "reference",
    "ref",
    "image",
    "path",
    "name",
    "target",
    "group",
    "split",
    "scene",
    "index",
]

CONSTRUCTION_ALLOWED_PATTERNS = [
    "match",
    "valid",
    "confidence",
    "conf",
    "entropy",
    "occupied",
    "spread",
    "scale",
    "anchor",
    "spatial",
    "topk",
    "target",
    "group",
    "query",
    "reference",
    "pair",
    "image",
    "path",
    "name",
]

CANDIDATE_FILES = [
    (
        "experiments/phase26_b_selective_correspondence_reasoning/slice_metrics/slice_metrics.csv",
        "validation_only",
        "B-SCR slice metrics; labels/errors must not construct states.",
    ),
    (
        "experiments/phase26_b_selective_correspondence_reasoning/slice_metrics_gate_off/slice_metrics.csv",
        "validation_only",
        "B-SCR gate-off slice metrics; validation only.",
    ),
    (
        "experiments/phase26_b1_frozen_external_matcher/slice_metrics/slice_metrics.csv",
        "validation_only",
        "B1 frozen matcher slice metrics; validation only.",
    ),
    (
        "experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl",
        "construction_allowed",
        "Frozen observable B-SCR eval feature packet candidate.",
    ),
    (
        "experiments/phase26_b_selective_correspondence_reasoning/features/train_bscr_features.jsonl",
        "construction_allowed",
        "Frozen observable B-SCR train feature packet candidate.",
    ),
    (
        "experiments/paper_pillars/15_train_test_distribution_gap/manifests/train_labeled_manifest.csv",
        "validation_only",
        "Labeled train manifest; identity/metadata safe, labels validation only.",
    ),
    (
        "experiments/paper_pillars/15_train_test_distribution_gap/manifests/dev_labeled_manifest.csv",
        "validation_only",
        "Labeled dev manifest; identity/metadata safe, labels validation only.",
    ),
    (
        "experiments/paper_pillars/20_official_metric_aware_distance_calibration/manifests/devsplit_v1_official_metric_manifest.csv",
        "validation_only",
        "Official-metric-aware dev manifest; validation only.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def normalize(name: str) -> str:
    return name.strip().lower()


def matches_any(name: str, patterns: list[str]) -> bool:
    lowered = normalize(name)
    return any(pattern in lowered for pattern in patterns)


def flatten_keys(record: dict[str, Any], prefix: str = "", limit: int = 256) -> list[str]:
    keys: list[str] = []
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)
        if len(keys) >= limit:
            return keys
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, full_key, limit=limit - len(keys)))
            if len(keys) >= limit:
                return keys[:limit]
    return keys


def sample_csv(path: Path) -> tuple[int | None, list[str], list[str], str | None]:
    try:
        row_count = 0
        sample_keys: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for row in reader:
                row_count += 1
                if not sample_keys:
                    sample_keys = [key for key, value in row.items() if value not in (None, "")]
            return row_count, columns, sample_keys, None
    except UnicodeDecodeError:
        row_count = 0
        sample_keys = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for row in reader:
                row_count += 1
                if not sample_keys:
                    sample_keys = [key for key, value in row.items() if value not in (None, "")]
            return row_count, columns, sample_keys, None
    except Exception as exc:  # pragma: no cover - inventory must be robust.
        return None, [], [], f"{type(exc).__name__}: {exc}"


def sample_jsonl(path: Path, max_rows: int = 10000) -> tuple[int | None, list[str], list[str], str | None]:
    columns: set[str] = set()
    sample_keys: list[str] = []
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row_count += 1
                if row_count <= max_rows:
                    record = json.loads(stripped)
                    if isinstance(record, dict):
                        flat = flatten_keys(record)
                        columns.update(flat)
                        if not sample_keys:
                            sample_keys = flat[:64]
                elif row_count > max_rows and columns:
                    # Keep counting cheaply after schema sample has been collected.
                    continue
        return row_count, sorted(columns), sample_keys, None
    except Exception as exc:  # pragma: no cover - inventory must be robust.
        return None, sorted(columns), sample_keys, f"{type(exc).__name__}: {exc}"


def classify_columns(columns: list[str], category: str) -> tuple[list[str], list[str], list[str]]:
    forbidden = [col for col in columns if matches_any(col, FORBIDDEN_PATTERNS)]
    if category == "construction_allowed":
        construction = [
            col
            for col in columns
            if col not in forbidden and matches_any(col, CONSTRUCTION_ALLOWED_PATTERNS)
        ]
        validation = [col for col in columns if col in forbidden]
    else:
        construction = [
            col
            for col in columns
            if col not in forbidden and matches_any(col, SAFE_IDENTITY_PATTERNS)
        ]
        validation = [col for col in columns if col not in construction]
    return construction, validation, forbidden


def inspect_file(project_root: Path, rel_path: str, category: str, note: str) -> dict[str, Any]:
    path = project_root / rel_path
    record: dict[str, Any] = {
        "path": rel_path,
        "exists": path.exists(),
        "category": category,
        "row_count": None,
        "columns": [],
        "sample_keys": [],
        "construction_allowed_columns": [],
        "validation_only_columns": [],
        "forbidden_for_construction_columns": [],
        "notes": [note],
    }
    if not path.exists():
        record["notes"].append("missing")
        return record
    if path.suffix.lower() == ".jsonl":
        row_count, columns, sample_keys, error = sample_jsonl(path)
    elif path.suffix.lower() == ".csv":
        row_count, columns, sample_keys, error = sample_csv(path)
    else:
        row_count, columns, sample_keys, error = None, [], [], f"unsupported suffix: {path.suffix}"
    construction, validation, forbidden = classify_columns(columns, category)
    record.update(
        {
            "row_count": row_count,
            "columns": columns,
            "sample_keys": sample_keys,
            "construction_allowed_columns": construction,
            "validation_only_columns": validation,
            "forbidden_for_construction_columns": forbidden,
        }
    )
    if error:
        record["notes"].append(error)
    if category == "validation_only" and construction:
        record["notes"].append("only identity/metadata columns are construction-safe; metrics remain validation-only")
    return record


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    files = [
        inspect_file(project_root, rel_path, category, note)
        for rel_path, category, note in CANDIDATE_FILES
    ]
    summary = {
        "total_files": len(files),
        "existing_files": sum(1 for item in files if item["exists"]),
        "missing_files": sum(1 for item in files if not item["exists"]),
        "construction_allowed_files": sum(
            1 for item in files if item["exists"] and item["category"] == "construction_allowed"
        ),
        "validation_only_files": sum(
            1 for item in files if item["exists"] and item["category"] == "validation_only"
        ),
        "files_with_forbidden_columns": sum(
            1 for item in files if item["forbidden_for_construction_columns"]
        ),
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "summary": summary,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
