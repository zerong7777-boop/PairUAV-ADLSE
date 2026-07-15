"""Artifact-level identity profiles for A-v3.2a."""
from collections import Counter, defaultdict

from scripts.phase27_a_v3_2a_identity_common import (
    compact_join,
    ensure_dirs,
    group_count,
    identity_key_strategies,
    read_csv_dicts,
    write_csv_dicts,
    write_json,
)


def _key_groups(rows, strategy):
    groups = defaultdict(list)
    missing = []
    fn = strategy["function"]
    for idx, row in enumerate(rows):
        key = fn(row)
        if key:
            groups[key].append((idx, row))
        else:
            missing.append((idx, row))
    return groups, missing


def _distribution_for_duplicate_rows(groups, field):
    dup_rows = [row for items in groups.values() if len(items) > 1 for _, row in items]
    return group_count(dup_rows, field)


def profile_artifact_rows(artifact_name, rows):
    profiles = []
    for strategy in identity_key_strategies():
        groups, missing = _key_groups(rows, strategy)
        duplicate_groups = {key: items for key, items in groups.items() if len(items) > 1}
        top = sorted(duplicate_groups.items(), key=lambda item: len(item[1]), reverse=True)[:10]
        profiles.append(
            {
                "artifact_name": artifact_name,
                "key_strategy": strategy["name"],
                "key_role": strategy["role"],
                "row_count": len(rows),
                "non_empty_key_count": sum(len(items) for items in groups.values()),
                "unique_key_count": len(groups),
                "duplicate_key_count": len(duplicate_groups),
                "duplicate_row_count": sum(len(items) for items in duplicate_groups.values()),
                "missing_key_count": len(missing),
                "top_duplicate_keys": compact_join([f"{key}:{len(items)}" for key, items in top]),
                "duplicate_target_distribution": str(_distribution_for_duplicate_rows(duplicate_groups, "target_key")),
                "duplicate_group_distribution": str(_distribution_for_duplicate_rows(duplicate_groups, "group_id")),
                "duplicate_scene_distribution": str(_distribution_for_duplicate_rows(duplicate_groups, "scene_key")),
            }
        )
    return profiles


def profile_all_artifacts(artifact_rows_map):
    rows = []
    for name, artifact_rows in artifact_rows_map.items():
        rows.extend(profile_artifact_rows(name, artifact_rows))
    return rows


def write_artifact_identity_profile(artifact_rows_map, output_dir):
    out = ensure_dirs(output_dir)
    rows = profile_all_artifacts(artifact_rows_map)
    fields = [
        "artifact_name",
        "key_strategy",
        "key_role",
        "row_count",
        "non_empty_key_count",
        "unique_key_count",
        "duplicate_key_count",
        "duplicate_row_count",
        "missing_key_count",
        "top_duplicate_keys",
        "duplicate_target_distribution",
        "duplicate_group_distribution",
        "duplicate_scene_distribution",
    ]
    write_csv_dicts(out / "tables" / "artifact_identity_profile.csv", rows, fields)
    metrics = {"profiles": rows}
    write_json(out / "metrics" / "identity_profile_metrics.json", metrics)
    return rows


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
    write_artifact_identity_profile(artifacts, args.output_dir)


if __name__ == "__main__":
    main()
