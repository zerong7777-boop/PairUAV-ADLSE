"""Duplicate, missing-key, and disjoint-universe exports for A-v3.2a."""
from scripts.phase27_a_v3_2a_identity_common import compact_join, ensure_dirs, identity_key_strategies, read_csv_dicts, write_csv_dicts
from scripts.phase27_a_v3_2a_pairwise_join_matrix import compute_pairwise_join_matrix, key_groups


def _row_id(idx, row):
    return row.get("canonical_pair_id") or row.get("pair_id") or row.get("source_row_index") or str(idx)


def build_duplicate_blocked_pairs(artifact_rows_map):
    output = []
    for artifact_name, rows in artifact_rows_map.items():
        for strategy in identity_key_strategies():
            groups = key_groups(rows, strategy["name"])
            for key, items in groups.items():
                if len(items) <= 1:
                    continue
                dup_type = "many_rows_same_key"
                output.append(
                    {
                        "artifact_name": artifact_name,
                        "key_strategy": strategy["name"],
                        "join_key": key,
                        "duplicate_type": dup_type,
                        "row_count": len(items),
                        "candidate_row_ids": compact_join([_row_id(idx, row) for idx, row in items]),
                        "source_image_keys": compact_join([row.get("source_image_key") or row.get("source_image_a") or row.get("stress_source_image_a") or "" for _, row in items]),
                        "target_image_keys": compact_join([row.get("target_image_key") or row.get("source_image_b") or row.get("stress_source_image_b") or "" for _, row in items]),
                        "pair_keys": compact_join([row.get("pair_key") or row.get("source_pair_key") or row.get("stress_source_pair_key") or "" for _, row in items]),
                        "target_keys": compact_join([row.get("target_key", "") for _, row in items]),
                        "group_ids": compact_join([row.get("group_id") or row.get("source_group_id") or row.get("stress_source_group_id") or "" for _, row in items]),
                        "scene_keys": compact_join([row.get("scene_key", "") for _, row in items]),
                        "surface_sources": compact_join([row.get("baseline_surface_source") or row.get("stress_baseline_surface_source") or row.get("stress_baseline_surface_source") or "" for _, row in items]),
                        "reason_codes": "duplicate_key_blocks_promotion",
                        "promotion_allowed": "false",
                    }
                )
    return output


def build_missing_key_rows(artifact_rows_map):
    output = []
    for artifact_name, rows in artifact_rows_map.items():
        for strategy in identity_key_strategies():
            fn = strategy["function"]
            for idx, row in enumerate(rows):
                if fn(row):
                    continue
                output.append(
                    {
                        "artifact_name": artifact_name,
                        "key_strategy": strategy["name"],
                        "row_id": _row_id(idx, row),
                        "target_key": row.get("target_key", ""),
                        "group_id": row.get("group_id") or row.get("source_group_id") or "",
                        "scene_key": row.get("scene_key", ""),
                        "reason_codes": "missing_identity_key",
                    }
                )
    return output


def build_disjoint_universe_summary(pairwise_matrix_rows):
    return [
        {
            "left_artifact": row["left_artifact"],
            "right_artifact": row["right_artifact"],
            "key_strategy": row["key_strategy"],
            "left_only_count": row["left_only_count"],
            "right_only_count": row["right_only_count"],
            "intersection_count": row["intersection_count"],
            "promotion_eligible": row["promotion_eligible"],
            "reason_codes": row["reason_codes"],
        }
        for row in pairwise_matrix_rows
    ]


def write_duplicate_blocked_pairs(artifact_rows_map, output_dir):
    out = ensure_dirs(output_dir)
    rows = build_duplicate_blocked_pairs(artifact_rows_map)
    fields = ["artifact_name", "key_strategy", "join_key", "duplicate_type", "row_count", "candidate_row_ids", "source_image_keys", "target_image_keys", "pair_keys", "target_keys", "group_ids", "scene_keys", "surface_sources", "reason_codes", "promotion_allowed"]
    write_csv_dicts(out / "tables" / "duplicate_blocked_pairs.csv", rows, fields)
    return rows


def write_missing_and_disjoint_exports(artifact_rows_map, pairwise_matrix_rows, output_dir):
    out = ensure_dirs(output_dir)
    missing = build_missing_key_rows(artifact_rows_map)
    disjoint = build_disjoint_universe_summary(pairwise_matrix_rows)
    write_csv_dicts(out / "tables" / "missing_key_rows.csv", missing, ["artifact_name", "key_strategy", "row_id", "target_key", "group_id", "scene_key", "reason_codes"])
    write_csv_dicts(out / "tables" / "disjoint_universe_summary.csv", disjoint, ["left_artifact", "right_artifact", "key_strategy", "left_only_count", "right_only_count", "intersection_count", "promotion_eligible", "reason_codes"])
    return missing, disjoint


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    artifacts = {}
    for item in args.artifact:
        name, path = item.split("=", 1)
        artifacts[name] = read_csv_dicts(path)
    matrix = compute_pairwise_join_matrix(artifacts)
    write_duplicate_blocked_pairs(artifacts, args.output_dir)
    write_missing_and_disjoint_exports(artifacts, matrix, args.output_dir)


if __name__ == "__main__":
    main()
