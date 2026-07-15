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


def load_per_sample(root):
    return {row["sample_id"]: row for row in read_csv(Path(root) / "official_per_sample.csv")}


def load_bscr_coverage(path):
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


def load_diagnostics(path):
    if not path or not Path(path).is_file():
        return {}
    return {row["sample_id"]: row for row in read_csv(path)}


def metric_row(sample_ids, baseline, bscr, coverage, diagnostics):
    ids = [sid for sid in sample_ids if sid in baseline and sid in bscr]
    base_rows = [baseline[sid] for sid in ids]
    bscr_rows = [bscr[sid] for sid in ids]
    base_final = mean(base_rows, "final_score")
    bscr_final = mean(bscr_rows, "final_score")
    base_dist = mean(base_rows, "distance_rel_error")
    bscr_dist = mean(bscr_rows, "distance_rel_error")
    base_angle = mean(base_rows, "angle_rel_error")
    bscr_angle = mean(bscr_rows, "angle_rel_error")
    diag_rows = [diagnostics[sid] for sid in ids if sid in diagnostics]
    covered = sum(1 for sid in ids if coverage.get(sid, {}).get("covered", False))
    fallback = sum(1 for sid in ids if coverage.get(sid, {}).get("fallback_used", False))
    gate_mean = mean(diag_rows, "bscr_gate_mean")
    gate_max = mean(diag_rows, "bscr_gate_max")
    residual_abs = mean(diag_rows, "bscr_heading_residual_abs")
    return {
        "sample_count": len(ids),
        "baseline_final": base_final,
        "bscr_final": bscr_final,
        "delta_final_bscr_minus_baseline": None if base_final is None or bscr_final is None else bscr_final - base_final,
        "baseline_distance_rel_error": base_dist,
        "bscr_distance_rel_error": bscr_dist,
        "delta_distance_rel_error": None if base_dist is None or bscr_dist is None else bscr_dist - base_dist,
        "baseline_angle_rel_error": base_angle,
        "bscr_angle_rel_error": bscr_angle,
        "delta_angle_rel_error": None if base_angle is None or bscr_angle is None else bscr_angle - base_angle,
        "bscr_covered": covered,
        "bscr_fallback": fallback,
        "diagnostic_rows": len(diag_rows),
        "bscr_gate_mean": gate_mean,
        "bscr_gate_max_mean": gate_max,
        "bscr_heading_residual_abs_mean": residual_abs,
    }


def verdict_from_slices(slices):
    aggregate = slices["aggregate_all"]
    semantic = slices["phase11_semantic_decoupled"]
    control = slices["phase11_control_low_sim_low_error"]
    phase14 = slices["phase14_alignment_sensitive"]
    phase14_control = slices["phase14_alignment_control_middle"]
    reasons = []
    verdict = "method-promising"
    if semantic["sample_count"] <= 0 or semantic["delta_final_bscr_minus_baseline"] is None or semantic["delta_final_bscr_minus_baseline"] >= 0:
        verdict = "rejected"
        reasons.append("phase11_semantic_decoupled_not_improved")
    if control["sample_count"] <= 0 or control["delta_final_bscr_minus_baseline"] is None or control["delta_final_bscr_minus_baseline"] > 0.01:
        verdict = "rejected"
        reasons.append("phase11_control_degradation_exceeds_0.01")
    if phase14["sample_count"] > 0 and phase14["delta_final_bscr_minus_baseline"] is not None and phase14["delta_final_bscr_minus_baseline"] > 0.01:
        verdict = "rejected"
        reasons.append("phase14_alignment_sensitive_degrades")
    if phase14_control["sample_count"] > 0 and phase14_control["delta_final_bscr_minus_baseline"] is not None and phase14_control["delta_final_bscr_minus_baseline"] > 0.01:
        verdict = "rejected"
        reasons.append("phase14_control_middle_degrades")
    if aggregate["delta_final_bscr_minus_baseline"] is not None and aggregate["delta_final_bscr_minus_baseline"] > 0.005:
        verdict = "rejected"
        reasons.append("aggregate_degradation_exceeds_0.005")
    if not reasons:
        sem_gate = semantic.get("bscr_gate_mean")
        ctrl_gate = control.get("bscr_gate_mean")
        if sem_gate is None or ctrl_gate is None or sem_gate < ctrl_gate + 0.10:
            verdict = "weak-inconclusive"
            reasons.append("gate_selectivity_margin_below_0.10")
    return {"verdict": verdict, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-eval-root", required=True)
    parser.add_argument("--bscr-eval-root", required=True)
    parser.add_argument("--phase11-controlled-csv", required=True)
    parser.add_argument("--phase14-surface-csv", required=True)
    parser.add_argument("--bscr-features", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    baseline = load_per_sample(args.baseline_eval_root)
    bscr = load_per_sample(args.bscr_eval_root)
    coverage = load_bscr_coverage(args.bscr_features)
    diagnostics = load_diagnostics(args.diagnostics)
    phase11 = {sample_id_from_phase_row(row): row for row in read_csv(args.phase11_controlled_csv)}
    phase14 = {sample_id_from_phase_row(row): row for row in read_csv(args.phase14_surface_csv)}
    all_ids = sorted(set(baseline) & set(bscr))

    slices = {"aggregate_all": metric_row(all_ids, baseline, bscr, coverage, diagnostics)}
    for label in ["phase11_semantic_decoupled", "phase11_control_low_sim_low_error", "phase11_control_other"]:
        ids = [sid for sid in all_ids if phase11_label(phase11.get(sid)) == label]
        slices[label] = metric_row(ids, baseline, bscr, coverage, diagnostics)
    for label in ["phase14_alignment_sensitive", "phase14_alignment_control_middle"]:
        ids = [sid for sid in all_ids if phase14_label(phase14.get(sid)) == label]
        slices[label] = metric_row(ids, baseline, bscr, coverage, diagnostics)

    out = {
        "baseline_eval_root": str(Path(args.baseline_eval_root)),
        "bscr_eval_root": str(Path(args.bscr_eval_root)),
        "diagnostics": str(Path(args.diagnostics)),
        "joined_rows": len(all_ids),
        "phase11_joined_rows": sum(1 for sid in all_ids if sid in phase11),
        "phase14_joined_rows": sum(1 for sid in all_ids if sid in phase14),
        "slices": slices,
        "verdict_inputs": verdict_from_slices(slices),
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
