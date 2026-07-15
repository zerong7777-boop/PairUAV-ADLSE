#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_phase_row(row):
    group = str(row["image_a"]).split("/")[0]
    return f"{group}/{row['json_id']}"


def boolish(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def phase11_label(row):
    if not row:
        return None
    if boolish(row.get("high_sim_high_error", "0")):
        return "phase11_semantic_decoupled"
    if boolish(row.get("low_sim_low_error", "0")):
        return "phase11_control_low_sim_low_error"
    if boolish(row.get("within_controlled_scope", "0")):
        return "phase11_control_other"
    return "phase11_unlabeled"


def phase14_label(row):
    if not row:
        return None
    residual = row.get("residual_target_dominance_label", "")
    if residual and residual != "middle":
        return "phase14_alignment_sensitive"
    if residual == "middle":
        return "phase14_alignment_control_middle"
    return "phase14_unlabeled"


def mean(rows, key):
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return sum(vals) / len(vals) if vals else None


def metric_row(sample_ids, baseline, fusion, coverage):
    ids = [sid for sid in sample_ids if sid in baseline and sid in fusion]
    base_rows = [baseline[sid] for sid in ids]
    fusion_rows = [fusion[sid] for sid in ids]
    base_final = mean(base_rows, "final_score")
    fusion_final = mean(fusion_rows, "final_score")
    base_dist = mean(base_rows, "distance_rel_error")
    fusion_dist = mean(fusion_rows, "distance_rel_error")
    base_angle = mean(base_rows, "angle_rel_error")
    fusion_angle = mean(fusion_rows, "angle_rel_error")
    covered = sum(1 for sid in ids if coverage.get(sid, {}).get("covered", False))
    fallback = sum(1 for sid in ids if coverage.get(sid, {}).get("fallback_used", False))
    return {
        "sample_count": len(ids),
        "baseline_final": base_final,
        "fusion_final": fusion_final,
        "delta_final_fusion_minus_baseline": None if base_final is None or fusion_final is None else fusion_final - base_final,
        "baseline_distance_rel_error": base_dist,
        "fusion_distance_rel_error": fusion_dist,
        "delta_distance_rel_error": None if base_dist is None or fusion_dist is None else fusion_dist - base_dist,
        "baseline_angle_rel_error": base_angle,
        "fusion_angle_rel_error": fusion_angle,
        "delta_angle_rel_error": None if base_angle is None or fusion_angle is None else fusion_angle - base_angle,
        "matcher_covered": covered,
        "matcher_fallback": fallback,
    }


def load_per_sample(root):
    rows = read_csv(Path(root) / "official_per_sample.csv")
    return {row["sample_id"]: row for row in rows}


def load_coverage(path):
    coverage = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            coverage[row["sample_id"]] = {
                "covered": not bool(row.get("fallback_used", False)),
                "fallback_used": bool(row.get("fallback_used", False)),
            }
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-eval-root", required=True)
    parser.add_argument("--fusion-eval-root", required=True)
    parser.add_argument("--phase11-controlled-csv", required=True)
    parser.add_argument("--phase14-surface-csv", required=True)
    parser.add_argument("--matcher-features", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    baseline = load_per_sample(args.baseline_eval_root)
    fusion = load_per_sample(args.fusion_eval_root)
    coverage = load_coverage(args.matcher_features)

    phase11 = {sample_id_from_phase_row(row): row for row in read_csv(args.phase11_controlled_csv)}
    phase14 = {sample_id_from_phase_row(row): row for row in read_csv(args.phase14_surface_csv)}
    all_ids = sorted(set(baseline) & set(fusion))

    slices = {"aggregate_all": metric_row(all_ids, baseline, fusion, coverage)}
    for label in ["phase11_semantic_decoupled", "phase11_control_low_sim_low_error", "phase11_control_other"]:
        ids = [sid for sid in all_ids if phase11_label(phase11.get(sid)) == label]
        slices[label] = metric_row(ids, baseline, fusion, coverage)
    for label in ["phase14_alignment_sensitive", "phase14_alignment_control_middle"]:
        ids = [sid for sid in all_ids if phase14_label(phase14.get(sid)) == label]
        slices[label] = metric_row(ids, baseline, fusion, coverage)

    out = {
        "baseline_eval_root": str(Path(args.baseline_eval_root)),
        "fusion_eval_root": str(Path(args.fusion_eval_root)),
        "joined_rows": len(all_ids),
        "phase11_joined_rows": sum(1 for sid in all_ids if sid in phase11),
        "phase14_joined_rows": sum(1 for sid in all_ids if sid in phase14),
        "slices": slices,
    }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slice_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = output_root / "slice_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["slice"] + list(next(iter(slices.values())).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in slices.items():
            writer.writerow({"slice": name, **row})
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
