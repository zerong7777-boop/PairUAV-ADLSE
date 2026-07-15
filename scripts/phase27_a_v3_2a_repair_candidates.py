"""Infer identity repair candidates without applying repairs."""
from scripts.phase27_a_v3_2a_identity_common import ensure_dirs, read_csv_dicts, write_csv_dicts, write_json


def _max_intersection(rows, strategy_name):
    vals = [int(row.get("intersection_count", 0)) for row in rows if row.get("key_strategy") == strategy_name]
    return max(vals) if vals else 0


def _sum_duplicates(profile_rows, strategy_name):
    return sum(int(row.get("duplicate_key_count", 0)) for row in profile_rows if row.get("key_strategy") == strategy_name)


def infer_repair_candidates(profile_rows, pairwise_rows):
    candidates = []
    raw = _max_intersection(pairwise_rows, "source_target_pair_composite")
    path_norm = _max_intersection(pairwise_rows, "path_normalized_source_target_pair")
    direction = _max_intersection(pairwise_rows, "direction_invariant_source_target_pair")
    canonical = _max_intersection(pairwise_rows, "canonical_pair_id")
    row_idx = _max_intersection(pairwise_rows, "row_index_diagnostic_only")
    dup = _sum_duplicates(profile_rows, "canonical_pair_id")

    if path_norm > raw:
        candidates.append(("path_normalization_candidate", f"path-normalized overlap {path_norm} exceeds raw composite {raw}", path_norm, "medium", "yes", "diagnostic-only", "manual spot-check before promotion"))
    if direction > raw:
        candidates.append(("direction_normalization_candidate", f"direction-invariant overlap {direction} exceeds ordered composite {raw}", direction, "medium", "yes", "diagnostic-only", "inspect pair direction semantics"))
    if raw == 0 and canonical == 0 and path_norm == 0 and row_idx > 0:
        candidates.append(("namespace_mapping_candidate", "only row-index diagnostic overlap exists", row_idx, "high", "yes", "no", "inspect namespace mapping"))
    if canonical == 0 and raw == 0 and path_norm == 0:
        candidates.append(("manifest_reacquisition_required", "no allowed key recovers overlap", 0, "low_false_match_high_missing", "no", "no", "define fixed shared pair manifest"))
    if dup > 0:
        candidates.append(("unrepairable_identity_conflict", f"canonical duplicate key groups observed: {dup}", dup, "high", "yes", "no", "resolve duplicate identity before promotion"))
    if canonical > 0 and raw > 0:
        candidates.append(("surface_regeneration_required", "identity overlap exists but surfaces may still need reacquisition for shared outcome", min(canonical, raw), "low", "no", "possible", "regenerate baseline/stress on fixed manifest"))

    if not candidates:
        candidates.append(("manifest_reacquisition_required", "no repair signal beyond reacquisition", 0, "unknown", "yes", "no", "build fixed pair manifest"))

    return [
        {
            "repair_class": c[0],
            "supporting_evidence": c[1],
            "affected_row_count": c[2],
            "false_match_risk": c[3],
            "manual_spot_check_required": c[4],
            "promotion_support": c[5],
            "next_action": c[6],
        }
        for c in candidates
    ]


def write_identity_repair_candidates(profile_rows, pairwise_rows, output_dir):
    out = ensure_dirs(output_dir)
    rows = infer_repair_candidates(profile_rows, pairwise_rows)
    fields = ["repair_class", "supporting_evidence", "affected_row_count", "false_match_risk", "manual_spot_check_required", "promotion_support", "next_action"]
    write_csv_dicts(out / "tables" / "identity_repair_candidates.csv", rows, fields)
    write_json(out / "metrics" / "identity_repair_candidate_metrics.json", {"repair_candidates": rows})
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_identity_repair_candidates(read_csv_dicts(args.profile), read_csv_dicts(args.pairwise), args.output_dir)


if __name__ == "__main__":
    main()
