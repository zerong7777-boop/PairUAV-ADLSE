#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCE_KEYS = [
    "roma_mean",
    "roma_max",
    "roma_certainty_mean",
    "mast3r_mean",
    "mast3r_max",
    "mast3r_conf_mean",
    "vggt_mean",
    "vggt_max",
    "vggt_conf_mean",
]


def pair_id_from_json(path, data):
    group_id = str(data.get("group_id", path.parent.name))
    json_id = str(data.get("json_id", path.stem))
    return f"{group_id}/{json_id}"


def load_source_table(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = row["pair_id"]
            if pair_id in rows:
                raise ValueError(f"duplicate source pair_id: {pair_id}")
            rows[pair_id] = row
    return rows


def f(row, key, default=0.0):
    raw = row.get(key, "")
    if raw == "" or raw is None:
        return float(default)
    return float(raw)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def build_policy_row(pair_id, split, source, policy_version):
    roma_angle = f(source, "roma_angle_error", 0.0)
    full_angle = f(source, "roma_mast3r_vggt_angle_error", 0.0)
    vggt_angle = f(source, "vggt_angle_error", 0.0)
    mast3r_vggt_angle = f(source, "mast3r_vggt_angle_error", 0.0)

    roma_distance = f(source, "roma_distance_error", 0.0)
    full_distance = f(source, "roma_mast3r_vggt_distance_error", 0.0)
    vggt_distance = f(source, "vggt_distance_error", 0.0)
    mast3r_vggt_distance = f(source, "mast3r_vggt_distance_error", 0.0)

    best_angle = min(roma_angle, full_angle, vggt_angle, mast3r_vggt_angle)
    best_distance = min(roma_distance, full_distance, vggt_distance, mast3r_vggt_distance)
    angle_gap = max(0.0, vggt_angle - best_angle)
    distance_gap = max(0.0, vggt_distance - best_distance)

    tags = []
    if full_angle <= min(roma_angle, vggt_angle, mast3r_vggt_angle):
        tags.append("roma_full_angle_helpful")
    if mast3r_vggt_distance <= min(roma_distance, full_distance, vggt_distance):
        tags.append("mast3r_vggt_distance_helpful")
    if angle_gap >= 5.0:
        tags.append("angle_policy_hard")
    if distance_gap >= 2.0:
        tags.append("range_policy_hard")
    if not tags:
        tags.append("easy_anchor")

    disagreement = abs(f(source, "roma_mean") - f(source, "vggt_mean")) + abs(f(source, "mast3r_mean") - f(source, "vggt_mean"))
    if disagreement >= 0.5:
        tags.append("source_disagreement_high")

    angle_weight = clamp(1.0 + angle_gap / 20.0, 0.5, 2.5)
    range_weight = clamp(1.0 + distance_gap / 10.0, 0.5, 2.5)
    sample_weight = clamp(0.75 + 0.25 * angle_weight + 0.25 * range_weight, 0.5, 2.0)
    if "easy_anchor" in tags:
        sample_weight = min(sample_weight, 1.0)

    source_stats = {key: f(source, key) for key in SOURCE_KEYS}
    source_errors_train_only = {
        "roma_angle_error": roma_angle,
        "roma_distance_error": roma_distance,
        "vggt_angle_error": vggt_angle,
        "vggt_distance_error": vggt_distance,
        "mast3r_vggt_angle_error": mast3r_vggt_angle,
        "mast3r_vggt_distance_error": mast3r_vggt_distance,
        "roma_mast3r_vggt_angle_error": full_angle,
        "roma_mast3r_vggt_distance_error": full_distance,
    }
    return {
        "pair_id": pair_id,
        "split": split,
        "bucket_tags": sorted(set(tags)),
        "source_stats": source_stats,
        "source_errors_train_only": source_errors_train_only,
        "angle_weight": angle_weight,
        "range_weight": range_weight,
        "sample_weight": sample_weight,
        "teacher_target": None,
        "teacher_active": False,
        "policy_version": policy_version,
    }


def source_stats_available(source):
    keys = (
        "roma_mean",
        "roma_certainty_mean",
        "vggt_mean",
        "vggt_conf_mean",
        "mast3r_mean",
        "mast3r_conf_mean",
    )
    return sum(1 for key in keys if f(source, key, 0.0) != 0.0) >= 2


def build_surface_policy_row(surface, split, policy_version):
    tags = []
    if int(float(surface.get("angle_policy_hard", 0.0))):
        tags.append("angle_policy_hard")
    if int(float(surface.get("range_policy_hard", 0.0))):
        tags.append("range_policy_hard")
    if int(float(surface.get("easy_anchor", 0.0))):
        tags.append("easy_anchor")
    if f(surface, "source_disagreement", 0.0) >= 0.5:
        tags.append("source_disagreement_high")
    if source_stats_available(surface):
        tags.append("source_stats_available")
    if not tags:
        tags.append("uncategorized_surface")

    source_stats = {key: f(surface, key) for key in SOURCE_KEYS}
    return {
        "pair_id": surface["pair_id"],
        "split": surface.get("split") or split,
        "bucket_tags": sorted(set(tags)),
        "source_stats": source_stats,
        "source_errors_train_only": {},
        "angle_weight": f(surface, "angle_weight", 1.0),
        "range_weight": f(surface, "range_weight", 1.0),
        "sample_weight": f(surface, "sample_weight", 1.0),
        "teacher_target": None,
        "teacher_active": False,
        "policy_version": policy_version,
    }


def write_surface_packet(surface_csv, output_jsonl, split, policy_version):
    surface_path = Path(surface_csv)
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with surface_path.open("r", encoding="utf-8", newline="") as handle, output.open("w", encoding="utf-8") as out:
        for surface in csv.DictReader(handle):
            row = build_surface_policy_row(surface, split, policy_version)
            row["provenance"] = {"surface_csv": str(surface_path)}
            out.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    return {
        "output_jsonl": str(output),
        "rows": written,
        "missing_source_rows": 0,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy_version": policy_version,
        "surface_csv": str(surface_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-root")
    parser.add_argument("--source-table-csv")
    parser.add_argument("--surface-csv")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--policy-version", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--require-source-row", action="store_true")
    args = parser.parse_args()

    if args.surface_csv:
        policy_version = args.policy_version or "source_policy_train_balanced_v1"
        summary = write_surface_packet(args.surface_csv, args.output_jsonl, args.split, policy_version)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if not args.json_root or not args.source_table_csv:
        parser.error("--json-root and --source-table-csv are required unless --surface-csv is provided")

    json_root = Path(args.json_root)
    source_rows = load_source_table(args.source_table_csv)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_version = args.policy_version or "source_policy_core_v1"

    written = 0
    missing = 0
    with output.open("w", encoding="utf-8") as handle:
        for json_path in sorted(json_root.rglob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            pair_id = pair_id_from_json(json_path, data)
            source = source_rows.get(pair_id)
            if source is None:
                missing += 1
                if args.require_source_row:
                    continue
                source = {"pair_id": pair_id}
            row = build_policy_row(pair_id, args.split, source, policy_version)
            row["provenance"] = {
                "json_root": str(json_root),
                "source_table_csv": str(Path(args.source_table_csv)),
                "json_path": str(json_path),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "output_jsonl": str(output),
        "rows": written,
        "missing_source_rows": missing,
        "sha256": digest,
        "policy_version": policy_version,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
