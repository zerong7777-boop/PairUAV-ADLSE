"""Build a tiny fixed manifest for A-v3.2c runner smoke."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


MANIFEST_COLUMNS = [
    "manifest_version",
    "manifest_hash",
    "manifest_row_id",
    "canonical_pair_id",
    "source_image_key",
    "target_image_key",
    "source_image_path",
    "target_image_path",
    "pair_direction",
    "target_key",
    "group_id",
    "scene_key",
    "split_key",
    "gt_heading",
    "gt_range",
    "json_path",
    "pair_key",
    "candidate_state",
    "candidate_flags_json",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_manifest_hash(rows):
    cols = [c for c in MANIFEST_COLUMNS if c != "manifest_hash"]
    h = hashlib.sha256()
    for row in rows:
        h.update("\t".join(str(row.get(c, "")) for c in cols).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_tiny_manifest(source_rows, full_rows, limit=16):
    full_by_id = {r.get("canonical_pair_id", ""): r for r in full_rows}
    rows = []
    for i, row in enumerate(source_rows):
        if len(rows) >= limit:
            break
        cid = row.get("canonical_pair_id", "")
        if not cid:
            continue
        full = full_by_id.get(cid, {})
        source_key = row.get("source_image_key") or row.get("source_key", "")
        target_key = row.get("target_image_key") or row.get("target_key", "")
        if not source_key or not target_key:
            continue
        rows.append(
            {
                "manifest_version": "phase27_a_v3_2c_tiny_v1",
                "manifest_hash": "",
                "manifest_row_id": str(len(rows)),
                "canonical_pair_id": cid,
                "source_image_key": source_key,
                "target_image_key": target_key,
                "source_image_path": source_key,
                "target_image_path": target_key,
                "pair_direction": row.get("pair_direction", "ordered_source_target"),
                "target_key": row.get("target_key_metadata", row.get("target_key", "")),
                "group_id": row.get("group_id", ""),
                "scene_key": row.get("scene_key", ""),
                "split_key": row.get("split_key", ""),
                "gt_heading": full.get("analysis_only_gt_angle", ""),
                "gt_range": full.get("analysis_only_gt_distance", ""),
                "json_path": row.get("json_path", ""),
                "pair_key": row.get("pair_key", ""),
                "candidate_state": row.get("candidate_state", ""),
                "candidate_flags_json": row.get("candidate_flags_json", ""),
            }
        )
    manifest_hash = compute_manifest_hash(rows)
    for row in rows:
        row["manifest_hash"] = manifest_hash
    return rows, {
        "row_count": len(rows),
        "unique_canonical_pair_id_count": len({r["canonical_pair_id"] for r in rows}),
        "missing_source_target_count": sum(1 for r in rows if not r["source_image_key"] or not r["target_image_key"]),
        "manifest_hash": manifest_hash,
        "manifest_version": "phase27_a_v3_2c_tiny_v1",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--full-dev-surface", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    rows, metrics = build_tiny_manifest(read_csv(args.source_manifest), read_csv(args.full_dev_surface), args.limit)
    write_csv(out / "manifests" / "fixed_manifest_tiny.csv", rows, MANIFEST_COLUMNS)
    write_json(out / "metrics" / "fixed_manifest_tiny_metrics.json", metrics)


if __name__ == "__main__":
    main()

