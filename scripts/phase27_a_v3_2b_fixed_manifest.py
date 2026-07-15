"""Build fixed bounded pair manifests for Phase27 A-v3.2b."""
import argparse
import json

from scripts.phase27_a_v3_2b_common import (
    canonical_pair_key,
    ensure_dirs,
    normalize_token,
    read_csv_dicts,
    sha256_rows,
    source_target_composite_key,
    write_csv_dicts,
    write_json,
)


MANIFEST_COLUMNS = [
    "manifest_version",
    "manifest_checksum",
    "canonical_pair_id",
    "source_key",
    "target_key",
    "source_image_key",
    "target_image_key",
    "pair_direction",
    "pair_key",
    "target_key_metadata",
    "group_id",
    "scene_key",
    "split_key",
    "candidate_state",
    "candidate_flags_json",
    "source_manifest_row_index_diagnostic_only",
]


def _state(row):
    for name in ("validation_status", "candidate_state", "final_state"):
        if row.get(name):
            return row.get(name)
    return "unknown"


def _flags(row):
    prefixes = ("READY_", "QUARANTINE_", "ANALYSIS_ONLY", "NOT_READY")
    data = {}
    for k, v in row.items():
        if k.endswith("_candidate") or k.startswith(prefixes):
            data[k] = v
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def build_fixed_manifest(candidate_rows, full_rows, limit=10000, manifest_version="phase27_a_v3_2b_bounded_v1"):
    full_ids = {canonical_pair_key(r) for r in full_rows if canonical_pair_key(r)}
    seen = set()
    out = []
    duplicate_candidate_ids = 0
    for idx, row in enumerate(candidate_rows):
        cid = canonical_pair_key(row)
        if not cid or cid not in full_ids:
            continue
        if cid in seen:
            duplicate_candidate_ids += 1
            continue
        source_key = normalize_token(row.get("source_image_key"))
        target_key = normalize_token(row.get("target_image_key"))
        if not source_key or not target_key:
            continue
        seen.add(cid)
        out.append(
            {
                "manifest_version": manifest_version,
                "manifest_checksum": "",
                "canonical_pair_id": cid,
                "source_key": source_key,
                "target_key": target_key,
                "source_image_key": source_key,
                "target_image_key": target_key,
                "pair_direction": "ordered_source_target",
                "pair_key": row.get("pair_key", ""),
                "target_key_metadata": row.get("target_key", ""),
                "group_id": row.get("group_id", ""),
                "scene_key": row.get("scene_key", ""),
                "split_key": row.get("split_key", row.get("source_split", "")),
                "candidate_state": _state(row),
                "candidate_flags_json": _flags(row),
                "source_manifest_row_index_diagnostic_only": str(idx),
            }
        )
        if limit and len(out) >= limit:
            break
    checksum_columns = [c for c in MANIFEST_COLUMNS if c != "manifest_checksum"]
    checksum = sha256_rows(out, checksum_columns)
    for row in out:
        row["manifest_checksum"] = checksum
    metrics = {
        "row_count": len(out),
        "unique_canonical_pair_id_count": len({r["canonical_pair_id"] for r in out}),
        "missing_source_key_count": sum(1 for r in out if not r["source_key"]),
        "missing_target_key_count": sum(1 for r in out if not r["target_key"]),
        "duplicate_candidate_ids_skipped": duplicate_candidate_ids,
        "manifest_checksum": checksum,
        "manifest_version": manifest_version,
        "source_target_composite_unique_count": len({source_target_composite_key(r) for r in out}),
    }
    return out, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--full-dev", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--mode", default="bounded_clean_candidate_full")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = ensure_dirs(args.output_dir)
    manifest, metrics = build_fixed_manifest(read_csv_dicts(args.candidate), read_csv_dicts(args.full_dev), args.limit)
    write_csv_dicts(out / "manifests" / "fixed_shared_pair_manifest_bounded.csv", manifest, MANIFEST_COLUMNS)
    write_json(out / "manifests" / "fixed_shared_pair_manifest_bounded.metrics.json", metrics)


if __name__ == "__main__":
    main()

