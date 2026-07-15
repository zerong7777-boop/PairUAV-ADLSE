"""Re-export existing baseline/stress outcomes against a fixed manifest."""
import argparse

from scripts.phase27_a_v3_2b_common import (
    canonical_pair_key,
    classify_duplicate_groups,
    ensure_dirs,
    read_csv_dicts,
    source_target_composite_key,
    write_csv_dicts,
    write_json,
)


BASELINE_COLUMNS = [
    "canonical_pair_id",
    "baseline_join_status",
    "baseline_duplicate_status",
    "baseline_missing_status",
    "baseline_source_target_composite_present",
    "baseline_angle_abs_error",
    "baseline_distance_abs_error",
    "baseline_angle_rel_error",
    "baseline_distance_rel_error",
    "baseline_surface_source",
]

STRESS_COLUMNS = [
    "variant_id",
    "canonical_pair_id",
    "stress_join_status",
    "stress_duplicate_status",
    "stress_missing_status",
    "stress_source_target_composite_present",
    "stress_angle_abs_error",
    "stress_distance_abs_error",
    "stress_angle_rel_error",
    "stress_distance_rel_error",
    "stress_surface_source",
    "source_row_index",
]

DUP_COLUMNS = ["variant_id", "canonical_pair_id", "row_count", "source_row_indices", "composite_present_count"]


def _baseline_row(cid, rows):
    if not rows:
        return {
            "canonical_pair_id": cid,
            "baseline_join_status": "missing",
            "baseline_duplicate_status": "none",
            "baseline_missing_status": "missing",
            "baseline_source_target_composite_present": "false",
        }
    if len(rows) > 1:
        status = "duplicate"
    else:
        status = "joined"
    row = rows[0]
    return {
        "canonical_pair_id": cid,
        "baseline_join_status": status,
        "baseline_duplicate_status": "duplicate" if len(rows) > 1 else "none",
        "baseline_missing_status": "none",
        "baseline_source_target_composite_present": "true" if source_target_composite_key(row) else "false",
        "baseline_angle_abs_error": row.get("baseline_angle_abs_error", ""),
        "baseline_distance_abs_error": row.get("baseline_distance_abs_error", ""),
        "baseline_angle_rel_error": row.get("baseline_angle_rel_error", ""),
        "baseline_distance_rel_error": row.get("baseline_distance_rel_error", ""),
        "baseline_surface_source": row.get("baseline_surface_source", ""),
    }


def reexport(fixed_rows, baseline_rows, stress_inputs):
    manifest_ids = [canonical_pair_key(r) for r in fixed_rows]
    baseline_groups = {}
    for row in baseline_rows:
        baseline_groups.setdefault(canonical_pair_key(row), []).append(row)
    baseline_out = [_baseline_row(cid, baseline_groups.get(cid, [])) for cid in manifest_ids]

    stress_long = []
    duplicate_rows = []
    stress_metrics = {}
    for variant_id, rows in stress_inputs:
        groups = {}
        for row in rows:
            groups.setdefault(canonical_pair_key(row), []).append(row)
        dup_count = 0
        missing_count = 0
        composite_missing = 0
        for cid in manifest_ids:
            matches = groups.get(cid, [])
            if not matches:
                missing_count += 1
                stress_long.append(
                    {
                        "variant_id": variant_id,
                        "canonical_pair_id": cid,
                        "stress_join_status": "missing",
                        "stress_duplicate_status": "none",
                        "stress_missing_status": "missing",
                        "stress_source_target_composite_present": "false",
                    }
                )
                continue
            if len(matches) > 1:
                dup_count += 1
                duplicate_rows.append(
                    {
                        "variant_id": variant_id,
                        "canonical_pair_id": cid,
                        "row_count": str(len(matches)),
                        "source_row_indices": "|".join(r.get("source_row_index", "") for r in matches),
                        "composite_present_count": str(sum(1 for r in matches if source_target_composite_key(r))),
                    }
                )
            for row in matches:
                composite = source_target_composite_key(row)
                if not composite:
                    composite_missing += 1
                stress_long.append(
                    {
                        "variant_id": variant_id,
                        "canonical_pair_id": cid,
                        "stress_join_status": "duplicate" if len(matches) > 1 else "joined",
                        "stress_duplicate_status": "duplicate" if len(matches) > 1 else "none",
                        "stress_missing_status": "none",
                        "stress_source_target_composite_present": "true" if composite else "false",
                        "stress_angle_abs_error": row.get("baseline_angle_abs_error", ""),
                        "stress_distance_abs_error": row.get("baseline_distance_abs_error", ""),
                        "stress_angle_rel_error": row.get("baseline_angle_rel_error", ""),
                        "stress_distance_rel_error": row.get("baseline_distance_rel_error", ""),
                        "stress_surface_source": row.get("baseline_surface_source", ""),
                        "source_row_index": row.get("source_row_index", ""),
                    }
                )
        stress_metrics[variant_id] = {
            "manifest_row_count": len(manifest_ids),
            "joined_or_duplicate_id_count": len([cid for cid in manifest_ids if groups.get(cid)]),
            "missing_id_count": missing_count,
            "duplicate_id_count": dup_count,
            "source_target_composite_missing_row_count": composite_missing,
        }
    metrics = {
        "fixed_manifest_row_count": len(manifest_ids),
        "baseline_joined_count": sum(1 for r in baseline_out if r["baseline_join_status"] == "joined"),
        "baseline_duplicate_count": sum(1 for r in baseline_out if r["baseline_join_status"] == "duplicate"),
        "baseline_missing_count": sum(1 for r in baseline_out if r["baseline_join_status"] == "missing"),
        "stress_metrics": stress_metrics,
        "stress_duplicate_group_count": len(duplicate_rows),
    }
    return baseline_out, stress_long, duplicate_rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--baseline-surface", required=True)
    parser.add_argument("--stress-surface", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    stress_inputs = []
    for item in args.stress_surface:
        name, path = item.split("=", 1)
        stress_inputs.append((name, read_csv_dicts(path)))
    out = ensure_dirs(args.output_dir)
    baseline, stress, dup, metrics = reexport(read_csv_dicts(args.fixed_manifest), read_csv_dicts(args.baseline_surface), stress_inputs)
    write_csv_dicts(out / "tables" / "baseline_on_fixed_manifest_bounded.csv", baseline, BASELINE_COLUMNS)
    write_csv_dicts(out / "tables" / "stress_on_fixed_manifest_bounded_long.csv", stress, STRESS_COLUMNS)
    write_csv_dicts(out / "tables" / "stress_duplicate_groups.csv", dup, DUP_COLUMNS)
    write_json(out / "metrics" / "outcome_reexport_metrics.json", metrics)


if __name__ == "__main__":
    main()

