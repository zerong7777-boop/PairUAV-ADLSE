"""Build a wide shared baseline/stress surface from fixed-manifest re-exports."""
import argparse
from collections import defaultdict

from scripts.phase27_a_v3_2b_common import ensure_dirs, read_csv_dicts, safe_float, write_csv_dicts, write_json


BASE_COLUMNS = [
    "canonical_pair_id",
    "source_key",
    "target_key",
    "group_id",
    "scene_key",
    "candidate_state",
    "baseline_join_status",
    "baseline_angle_abs_error",
    "baseline_distance_abs_error",
    "shared_pair_status",
]


def _variants(stress_rows):
    return sorted({r.get("variant_id", "") for r in stress_rows if r.get("variant_id", "")})


def build_shared_wide(fixed_rows, baseline_rows, stress_rows):
    baseline = {r["canonical_pair_id"]: r for r in baseline_rows}
    variants = _variants(stress_rows)
    stress = defaultdict(list)
    for row in stress_rows:
        stress[(row.get("canonical_pair_id", ""), row.get("variant_id", ""))].append(row)
    rows = []
    for fixed in fixed_rows:
        cid = fixed["canonical_pair_id"]
        b = baseline.get(cid, {})
        out = {
            "canonical_pair_id": cid,
            "source_key": fixed.get("source_key", ""),
            "target_key": fixed.get("target_key", ""),
            "group_id": fixed.get("group_id", ""),
            "scene_key": fixed.get("scene_key", ""),
            "candidate_state": fixed.get("candidate_state", ""),
            "baseline_join_status": b.get("baseline_join_status", "missing"),
            "baseline_angle_abs_error": b.get("baseline_angle_abs_error", ""),
            "baseline_distance_abs_error": b.get("baseline_distance_abs_error", ""),
        }
        ready = out["baseline_join_status"] == "joined"
        b_angle = safe_float(out["baseline_angle_abs_error"])
        b_dist = safe_float(out["baseline_distance_abs_error"])
        for variant in variants:
            matches = stress.get((cid, variant), [])
            prefix = f"stress_{variant}"
            unique = [r for r in matches if r.get("stress_join_status") == "joined"]
            duplicate = any(r.get("stress_duplicate_status") == "duplicate" for r in matches)
            missing = not matches or any(r.get("stress_missing_status") == "missing" for r in matches)
            row = unique[0] if len(unique) == 1 and not duplicate else {}
            out[f"{prefix}_join_status"] = "joined" if row else ("duplicate" if duplicate else "missing")
            out[f"{prefix}_duplicate_status"] = "duplicate" if duplicate else "none"
            out[f"{prefix}_missing_status"] = "missing" if missing else "none"
            out[f"{prefix}_composite_present"] = row.get("stress_source_target_composite_present", "false") if row else "false"
            out[f"{prefix}_heading_error"] = row.get("stress_angle_abs_error", "") if row else ""
            out[f"{prefix}_range_error"] = row.get("stress_distance_abs_error", "") if row else ""
            s_angle = safe_float(out[f"{prefix}_heading_error"])
            s_dist = safe_float(out[f"{prefix}_range_error"])
            out[f"{prefix}_delta_heading"] = "" if s_angle is None or b_angle is None else f"{s_angle - b_angle:.12g}"
            out[f"{prefix}_delta_range"] = "" if s_dist is None or b_dist is None else f"{s_dist - b_dist:.12g}"
            if not row or duplicate or missing or out[f"{prefix}_composite_present"] != "true":
                ready = False
        out["shared_pair_status"] = "ready" if ready else "not_ready"
        out["coverage_bucket"] = "shared_ready" if ready else "not_shared"
        out["bias_bucket"] = out["shared_pair_status"]
        rows.append(out)
    fields = list(BASE_COLUMNS)
    for variant in variants:
        prefix = f"stress_{variant}"
        fields.extend([
            f"{prefix}_join_status",
            f"{prefix}_duplicate_status",
            f"{prefix}_missing_status",
            f"{prefix}_composite_present",
            f"{prefix}_heading_error",
            f"{prefix}_range_error",
            f"{prefix}_delta_heading",
            f"{prefix}_delta_range",
        ])
    fields.extend(["coverage_bucket", "bias_bucket"])
    metrics = {
        "fixed_manifest_row_count": len(fixed_rows),
        "variant_count": len(variants),
        "shared_ready_count": sum(1 for r in rows if r["shared_pair_status"] == "ready"),
        "shared_coverage_ratio": 0.0 if not rows else sum(1 for r in rows if r["shared_pair_status"] == "ready") / len(rows),
        "variants": variants,
    }
    return rows, fields, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stress-long", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ensure_dirs(args.output_dir)
    rows, fields, metrics = build_shared_wide(read_csv_dicts(args.fixed_manifest), read_csv_dicts(args.baseline), read_csv_dicts(args.stress_long))
    write_csv_dicts(out / "tables" / "baseline_stress_shared_wide_surface_bounded.csv", rows, fields)
    write_json(out / "metrics" / "shared_wide_surface_metrics.json", metrics)


if __name__ == "__main__":
    main()

