"""Pairwise identity join matrix for A-v3.2a."""
from collections import defaultdict
from itertools import combinations

from scripts.phase27_a_v3_2a_identity_common import ensure_dirs, get_strategy, identity_key_strategies, read_csv_dicts, write_csv_dicts, write_json


def key_groups(rows, strategy_name):
    strategy = get_strategy(strategy_name)
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        key = strategy["function"](row)
        if key:
            groups[key].append((idx, row))
    return groups


def _promotion_eligible(strategy, intersection_count, duplicate_blocked_count, one_to_one_count):
    if strategy["role"] in {"diagnostic_key", "forbidden_for_promotion"}:
        return False, "diagnostic_or_forbidden_key"
    if intersection_count == 0:
        return False, "zero_intersection"
    if duplicate_blocked_count > 0:
        return False, "duplicates_block_promotion"
    if one_to_one_count == 0:
        return False, "no_one_to_one_overlap"
    return True, "clean_one_to_one_overlap"


def compute_pairwise_join_for_strategy(left_name, left_rows, right_name, right_rows, strategy_name):
    strategy = get_strategy(strategy_name)
    left = key_groups(left_rows, strategy_name)
    right = key_groups(right_rows, strategy_name)
    left_keys = set(left)
    right_keys = set(right)
    inter = left_keys & right_keys
    one_to_one = one_to_many = many_to_one = many_to_many = duplicate_blocked = 0
    for key in inter:
        l_count = len(left[key])
        r_count = len(right[key])
        if l_count == 1 and r_count == 1:
            one_to_one += 1
        elif l_count == 1 and r_count > 1:
            one_to_many += 1
            duplicate_blocked += 1
        elif l_count > 1 and r_count == 1:
            many_to_one += 1
            duplicate_blocked += 1
        else:
            many_to_many += 1
            duplicate_blocked += 1
    eligible, reason = _promotion_eligible(strategy, len(inter), duplicate_blocked, one_to_one)
    return {
        "left_artifact": left_name,
        "right_artifact": right_name,
        "key_strategy": strategy_name,
        "key_role": strategy["role"],
        "left_unique_keys": len(left_keys),
        "right_unique_keys": len(right_keys),
        "intersection_count": len(inter),
        "left_only_count": len(left_keys - right_keys),
        "right_only_count": len(right_keys - left_keys),
        "one_to_one_count": one_to_one,
        "one_to_many_count": one_to_many,
        "many_to_one_count": many_to_one,
        "many_to_many_count": many_to_many,
        "duplicate_blocked_count": duplicate_blocked,
        "promotion_eligible": str(eligible).lower(),
        "reason_codes": reason,
    }


def compute_pairwise_join_matrix(artifact_rows_map):
    rows = []
    names = list(artifact_rows_map)
    for left_name, right_name in combinations(names, 2):
        for strategy in identity_key_strategies():
            rows.append(compute_pairwise_join_for_strategy(left_name, artifact_rows_map[left_name], right_name, artifact_rows_map[right_name], strategy["name"]))
    return rows


def write_pairwise_join_matrix(artifact_rows_map, output_dir):
    out = ensure_dirs(output_dir)
    rows = compute_pairwise_join_matrix(artifact_rows_map)
    fields = [
        "left_artifact",
        "right_artifact",
        "key_strategy",
        "key_role",
        "left_unique_keys",
        "right_unique_keys",
        "intersection_count",
        "left_only_count",
        "right_only_count",
        "one_to_one_count",
        "one_to_many_count",
        "many_to_one_count",
        "many_to_many_count",
        "duplicate_blocked_count",
        "promotion_eligible",
        "reason_codes",
    ]
    write_csv_dicts(out / "tables" / "pairwise_join_matrix.csv", rows, fields)
    write_json(out / "metrics" / "pairwise_join_matrix_metrics.json", {"pairwise_rows": rows})
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
    write_pairwise_join_matrix(artifacts, args.output_dir)


if __name__ == "__main__":
    main()
