"""Discover A-v3.1 shared-surface input artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, write_json


def discover_csv_files(root_paths):
    files = []
    for root in root_paths:
        path = Path(root)
        if path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        elif path.exists():
            files.extend(sorted(path.rglob("*.csv")))
    return sorted(files)


def preview_csv_header(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, {})
        return {"path": str(path), "fieldnames": reader.fieldnames or [], "first_row": first}


def classify_surface_file(path, fieldnames):
    name = Path(path).name.lower()
    fields = set(fieldnames or [])
    if "canonical_pair_id" in fields and "evidence_sufficient_candidate" in fields:
        if "training_readiness" in name:
            return "candidate_outcome_manifest"
        return "candidate_manifest"
    if any(f.startswith("stress_") for f in fields) or "stress" in name:
        return "stress_surface"
    if "baseline" in name or any(f.startswith("baseline_") for f in fields):
        return "baseline_surface"
    return "unknown"


def build_input_manifest(root):
    root = Path(root)
    search_roots = [
        root / "experiments" / "phase27_a_taxonomy_redesign_v3",
        root / "experiments" / "phase27_a_validation_spine" / "baseline_surfaces",
        root / "experiments" / "phase27_a_v3_validation_extension_outcome_consistency_audit",
    ]
    entries = []
    for file_path in discover_csv_files(search_roots):
        preview = preview_csv_header(file_path)
        kind = classify_surface_file(file_path, preview["fieldnames"])
        entries.append({"path": str(file_path), "kind": kind, "fieldnames": preview["fieldnames"]})
    kinds = {entry["kind"] for entry in entries}
    missing = []
    for required in ("candidate_outcome_manifest", "baseline_surface", "stress_surface"):
        if required not in kinds:
            missing.append(required)
    return {"root": str(root), "search_roots": [str(p) for p in search_roots], "entries": entries, "missing_expected_families": missing}


def write_input_manifest(root, output_dir):
    out = ensure_output_dirs(output_dir)
    manifest = build_input_manifest(root)
    write_json(out / "input_manifest.json", manifest)
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_input_manifest(args.root, args.output_dir)


if __name__ == "__main__":
    main()
