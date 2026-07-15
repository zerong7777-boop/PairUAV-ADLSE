#!/usr/bin/env python3
"""Build a joinable baseline prediction surface from Phase21 replay artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--per-sample", required=True)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--metrics-json", required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(args.prepared_manifest)
    per_sample_rows = read_csv(args.per_sample)
    evidence_rows = read_csv(args.evidence_manifest)

    manifest_by_index = {row["__row_index__"]: row for row in manifest_rows}
    evidence_keys = {row["canonical_pair_id"] for row in evidence_rows if row.get("canonical_pair_id")}

    out_rows = []
    missing_manifest = 0
    for eval_row in per_sample_rows:
        manifest = manifest_by_index.get(eval_row.get("sample_id", ""))
        if manifest is None:
            missing_manifest += 1
            continue
        canonical_pair_id = f"{manifest['group_id']}/{manifest['json_id']}"
        out = {
            "canonical_pair_id": canonical_pair_id,
            "source_row_index": manifest["__row_index__"],
            "source_split": manifest.get("split", ""),
            "source_json_path": manifest.get("json_path", ""),
            "source_group_id": manifest.get("group_id", ""),
            "source_json_id": manifest.get("json_id", ""),
            "source_image_a": manifest.get("image_a", ""),
            "source_image_b": manifest.get("image_b", ""),
            "source_pair_key": manifest.get("pair_key", ""),
            "baseline_pred_angle": eval_row.get("pred_angle", ""),
            "baseline_pred_distance": eval_row.get("pred_distance", ""),
            "baseline_final_score": eval_row.get("final_score", ""),
            "baseline_angle_abs_error": eval_row.get("angle_abs_error", ""),
            "baseline_angle_rel_error": eval_row.get("angle_rel_error", ""),
            "baseline_distance_abs_error": eval_row.get("distance_abs_error", ""),
            "baseline_distance_rel_error": eval_row.get("distance_rel_error", ""),
            "analysis_only_gt_angle": eval_row.get("gt_angle", ""),
            "analysis_only_gt_distance": eval_row.get("gt_distance", ""),
            "analysis_only_distance_valid": eval_row.get("distance_valid", ""),
            "analysis_only_angle_valid": eval_row.get("angle_valid", ""),
            "baseline_surface_source": "phase21_full_dev_replay_row_index_join",
        }
        out_rows.append(out)

    out_keys = {row["canonical_pair_id"] for row in out_rows}
    overlap = out_keys & evidence_keys
    metrics = {
        "prepared_manifest": args.prepared_manifest,
        "per_sample": args.per_sample,
        "evidence_manifest": args.evidence_manifest,
        "output_csv": args.out_csv,
        "manifest_rows": len(manifest_rows),
        "per_sample_rows": len(per_sample_rows),
        "output_rows": len(out_rows),
        "missing_manifest_rows": missing_manifest,
        "evidence_rows": len(evidence_rows),
        "evidence_unique_keys": len(evidence_keys),
        "output_unique_keys": len(out_keys),
        "overlap_with_evidence": len(overlap),
        "overlap_examples": sorted(overlap)[:20],
        "verdict": "joinable_baseline_surface_ready" if overlap else "joinable_baseline_surface_failed",
    }
    write_csv(args.out_csv, out_rows)
    write_json(args.metrics_json, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if overlap else 2


if __name__ == "__main__":
    raise SystemExit(main())
