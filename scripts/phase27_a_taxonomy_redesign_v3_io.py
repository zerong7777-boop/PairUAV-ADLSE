"""CSV IO and canonical joining helpers for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def build_canonical_pair_id(row: dict) -> str:
    existing = str(row.get("canonical_pair_id") or "").strip()
    if existing:
        return existing
    source = str(row.get("source_image_key") or row.get("image_a") or row.get("source_image_a") or "").strip()
    target = str(row.get("target_image_key") or row.get("image_b") or row.get("source_image_b") or "").strip()
    pair_key = str(row.get("pair_key") or row.get("source_pair_key") or "").strip()
    group = str(row.get("group_id") or row.get("source_group_id") or "").strip()
    if source and target:
        return f"{source}::{target}"
    if pair_key:
        return f"pair::{pair_key}"
    if group:
        return f"group::{group}"
    return ""


def left_join_by_pair_id(left_rows: list[dict], right_rows: list[dict], right_prefix: str) -> list[dict]:
    right_index = {}
    for row in right_rows:
        key = build_canonical_pair_id(row)
        if key and key not in right_index:
            right_index[key] = row
    joined = []
    for row in left_rows:
        out = dict(row)
        key = build_canonical_pair_id(row)
        out["canonical_pair_id"] = key
        status_field = f"{right_prefix}_join_status"
        if not key:
            out[status_field] = "missing_key"
        elif key in right_index:
            out[status_field] = "joined"
            for k, v in right_index[key].items():
                if k != "canonical_pair_id":
                    out[f"{right_prefix}_{k}"] = v
        else:
            out[status_field] = "unjoined"
        joined.append(out)
    return joined


def summarize_join_status(rows: Iterable[dict], status_field: str) -> dict[str, int]:
    counts = {}
    for row in rows:
        status = str(row.get(status_field) or "missing_key")
        counts[status] = counts.get(status, 0) + 1
    return counts


def find_input_paths(root: str | Path) -> dict:
    root = Path(root)
    experiments = root / "experiments"
    manifests = experiments / "phase27_a_evidence_state_manifest" / "manifests"
    evidence = manifests / "a_evidence_state_manifest_v3_calibrated_v2.csv"
    if not evidence.exists():
        matches = sorted(manifests.glob("*calibrated*v2*.csv"))
        evidence = matches[0] if matches else evidence
    surfaces = experiments / "phase27_a_validation_spine" / "baseline_surfaces"
    full_dev = sorted(surfaces.glob("*full_dev*surface.csv"))
    stress = sorted(surfaces.glob("*stress*surface.csv"))
    return {
        "root": str(root),
        "candidate_manifest": str(evidence),
        "taxonomy_v2_dir": str(experiments / "phase27_a_taxonomy_redesign_v2"),
        "taxonomy_v2_diagnostics_dir": str(experiments / "phase27_a_taxonomy_redesign_v2_diagnostics"),
        "full_dev_surface": str(full_dev[0]) if full_dev else "",
        "stress_surfaces": [str(p) for p in stress],
    }


def write_input_manifest(root: str | Path, output_path: str | Path) -> dict:
    paths = find_input_paths(root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(paths, indent=2, sort_keys=True), encoding="utf-8")
    return paths
