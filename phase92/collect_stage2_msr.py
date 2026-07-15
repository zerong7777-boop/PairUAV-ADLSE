#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from phase92.stage2_modes import STAGE2_CONTROL_RUNS


METRICS = ["final_score", "angle_rel_error", "distance_rel_error"]
REFERENCE_IDS = ["B1", "B3", "B4"]
STAGE1_IDS = ["C", "D", "E"]


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def load_stage1_metric_rows(phase92_root: Path) -> list[dict[str, object]]:
    path = phase92_root / "ablations" / "msr_stage1_metrics.csv"
    return list(read_csv(path))


def control_run_name(spec: dict[str, object], run_group: str, steps: int, samples: int, lr: str) -> str:
    return (
        f"phase92_{spec['row_id']}_{run_group}_steps{steps}_val{samples}_"
        f"lr{lr}_seed{spec['train_seed']}_bdim{spec['msr_bottleneck_dim']}"
    )


def load_control_metric_rows(
    phase92_root: Path,
    run_group: str,
    steps: int,
    samples: int,
    lr: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eval_root = phase92_root / "ablations" / f"{run_group}_eval"
    for spec in STAGE2_CONTROL_RUNS:
        run_name = control_run_name(spec, run_group, steps, samples, lr)
        eval_dir = eval_root / run_name
        metrics_path = eval_dir / "official_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        rows.append(
            {
                "model_id": spec["row_id"],
                "model_name": spec["name"],
                "family": "phase92_stage2_control",
                "claim": spec["claim"],
                "output_mode": spec["output_mode"],
                "msr_bottleneck_dim": spec["msr_bottleneck_dim"],
                "train_seed": spec["train_seed"],
                "data_seed": spec["data_seed"],
                "eval_dir": str(eval_dir),
                "metrics_exists": metrics_path.exists(),
                **{metric: metrics.get(metric, "") for metric in METRICS},
            }
        )
    return rows


def load_samples(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, str]]]:
    samples: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        eval_dir = Path(str(row.get("eval_dir", "")))
        sample_path = eval_dir / "official_per_sample.csv"
        if sample_path.exists():
            samples[str(row["model_id"])] = {r["sample_id"]: r for r in read_csv(sample_path)}
    return samples


def common_ids(samples: dict[str, dict[str, dict[str, str]]], ids: list[str]) -> list[str]:
    present = [samples[mid] for mid in ids if mid in samples]
    if len(present) != len(ids):
        return []
    return sorted(set.intersection(*(set(s.keys()) for s in present)))


def delta(samples: dict[str, dict[str, dict[str, str]]], sid: str, model: str, ref: str, metric: str) -> float | None:
    a = parse_float(samples[model][sid].get(metric))
    b = parse_float(samples[ref][sid].get(metric))
    if a is None or b is None:
        return None
    return a - b


def pairwise_summary(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    ref_ids: list[str],
    ids: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in model_ids:
        if model not in samples:
            continue
        for ref in ref_ids:
            if ref not in samples:
                continue
            for metric in METRICS:
                values: list[float] = []
                better = worse = tie = 0
                severe_regression = 0
                threshold = 0.003 if metric == "angle_rel_error" else 0.006
                if metric == "final_score":
                    threshold = 0.005
                for sid in ids:
                    d = delta(samples, sid, model, ref, metric)
                    if d is None:
                        continue
                    values.append(d)
                    if d < -1e-12:
                        better += 1
                    elif d > 1e-12:
                        worse += 1
                    else:
                        tie += 1
                    if d > threshold:
                        severe_regression += 1
                n = len(values)
                rows.append(
                    {
                        "model_id": model,
                        "reference_id": ref,
                        "metric": metric,
                        "valid_count": n,
                        "mean_delta_model_minus_ref": mean(values),
                        "better_count": better,
                        "worse_count": worse,
                        "tie_count": tie,
                        "better_rate": better / n if n else "",
                        "material_regression_count": severe_regression,
                        "material_regression_rate": severe_regression / n if n else "",
                    }
                )
    return rows


def quadrant_summary(
    samples: dict[str, dict[str, dict[str, str]]],
    model: str,
    ref: str,
    ids: list[str],
) -> list[dict[str, object]]:
    buckets: dict[str, list[str]] = {
        "angle_and_distance_win": [],
        "angle_only_win": [],
        "distance_only_win": [],
        "both_lose_or_tie": [],
    }
    for sid in ids:
        ad = delta(samples, sid, model, ref, "angle_rel_error")
        dd = delta(samples, sid, model, ref, "distance_rel_error")
        if ad is None or dd is None:
            continue
        angle_win = ad < 0
        distance_win = dd < 0
        if angle_win and distance_win:
            key = "angle_and_distance_win"
        elif angle_win:
            key = "angle_only_win"
        elif distance_win:
            key = "distance_only_win"
        else:
            key = "both_lose_or_tie"
        buckets[key].append(sid)

    rows: list[dict[str, object]] = []
    total = sum(len(v) for v in buckets.values())
    for name, bucket_ids in buckets.items():
        row: dict[str, object] = {
            "model_id": model,
            "reference_id": ref,
            "quadrant": name,
            "count": len(bucket_ids),
            "rate": len(bucket_ids) / total if total else "",
        }
        for metric in METRICS:
            values = [d for sid in bucket_ids if (d := delta(samples, sid, model, ref, metric)) is not None]
            row[f"{metric}_mean_delta"] = mean(values)
        rows.append(row)
    return rows


def sample_mean(samples: dict[str, dict[str, dict[str, str]]], model: str, ids: list[str], metric: str) -> float | None:
    vals = [v for sid in ids if (v := parse_float(samples[model][sid].get(metric))) is not None]
    return mean(vals)


def best_model(samples: dict[str, dict[str, dict[str, str]]], models: list[str], ids: list[str], metric: str) -> tuple[str, float | None]:
    vals = {model: sample_mean(samples, model, ids, metric) for model in models if model in samples}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return "", None
    best = min(vals, key=lambda k: vals[k])
    return best, vals[best]


def conflict_subsets(samples: dict[str, dict[str, dict[str, str]]], ids: list[str]) -> dict[str, list[str]]:
    subsets = {
        "core_b4_angle_b3_distance_conflict": [],
        "b4_angle_better_only": [],
        "b3_distance_better_only": [],
        "b4_final_better_b3_distance_better": [],
        "no_anchor_conflict": [],
    }
    for sid in ids:
        b4_angle = delta(samples, sid, "B4", "B3", "angle_rel_error")
        b3_distance = delta(samples, sid, "B3", "B4", "distance_rel_error")
        b4_final = delta(samples, sid, "B4", "B3", "final_score")
        if b4_angle is None or b3_distance is None:
            continue
        angle_conflict = b4_angle < 0
        distance_conflict = b3_distance < 0
        if angle_conflict and distance_conflict:
            subsets["core_b4_angle_b3_distance_conflict"].append(sid)
        elif angle_conflict:
            subsets["b4_angle_better_only"].append(sid)
        elif distance_conflict:
            subsets["b3_distance_better_only"].append(sid)
        else:
            subsets["no_anchor_conflict"].append(sid)
        if b4_final is not None and b4_final < 0 and distance_conflict:
            subsets["b4_final_better_b3_distance_better"].append(sid)
    return subsets


def conflict_rows(
    samples: dict[str, dict[str, dict[str, str]]],
    ids: list[str],
    models: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subsets = conflict_subsets(samples, ids)
    for subset_name, subset_ids in subsets.items():
        if not subset_ids:
            continue
        for metric in METRICS:
            best, best_value = best_model(samples, models, subset_ids, metric)
            row: dict[str, object] = {
                "subset": subset_name,
                "metric": metric,
                "sample_count": len(subset_ids),
                "best_model": best,
                "best_value": best_value,
            }
            for model in models:
                if model in samples:
                    row[f"{model}_mean"] = sample_mean(samples, model, subset_ids, metric)
            rows.append(row)
    return rows


def bucket_matrix_rows(
    samples: dict[str, dict[str, dict[str, str]]],
    ids: list[str],
    models: list[str],
    min_count: int,
) -> list[dict[str, object]]:
    keyed: dict[tuple[str, str], list[str]] = {}
    anchor = "B4" if "B4" in samples else next(iter(samples))
    for sid in ids:
        a_bucket = samples[anchor][sid].get("angle_gt_bucket", "")
        d_bucket = samples[anchor][sid].get("gt_distance_bucket", "")
        if not a_bucket or not d_bucket:
            continue
        keyed.setdefault((a_bucket, d_bucket), []).append(sid)

    rows: list[dict[str, object]] = []
    for (a_bucket, d_bucket), bucket_ids in sorted(keyed.items()):
        if len(bucket_ids) < min_count:
            continue
        for metric in METRICS:
            best, best_value = best_model(samples, models, bucket_ids, metric)
            rows.append(
                {
                    "angle_gt_bucket": a_bucket,
                    "gt_distance_bucket": d_bucket,
                    "metric": metric,
                    "sample_count": len(bucket_ids),
                    "best_model": best,
                    "best_value": best_value,
                    "E_mean": sample_mean(samples, "E", bucket_ids, metric) if "E" in samples else "",
                    "B4_mean": sample_mean(samples, "B4", bucket_ids, metric) if "B4" in samples else "",
                    "B3_mean": sample_mean(samples, "B3", bucket_ids, metric) if "B3" in samples else "",
                }
            )
    return rows


def metric_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["model_id"]): row for row in rows}


def stage2_verdict(stage1_rows: list[dict[str, object]], control_rows: list[dict[str, object]]) -> dict[str, object]:
    by_id = metric_by_id(stage1_rows + control_rows)
    b4 = by_id.get("B4", {})
    e = by_id.get("E", {})
    if not b4 or not e:
        return {
            "verdict": "repeat_with_one_missing_control",
            "reason": "missing B4 or E metrics",
            "authorized_next": "repair_metrics",
        }

    e_final_delta = parse_float(e.get("final_score")) - parse_float(b4.get("final_score"))
    e_angle_delta = parse_float(e.get("angle_rel_error")) - parse_float(b4.get("angle_rel_error"))
    e_distance_delta = parse_float(e.get("distance_rel_error")) - parse_float(b4.get("distance_rel_error"))
    distance_compensated = e_final_delta <= 0 and e_angle_delta > 0.003 and e_distance_delta < -0.006

    controls_present = [
        row for row in control_rows if str(row.get("metrics_exists")).lower() == "true" or row.get("metrics_exists") is True
    ]
    payload: dict[str, object] = {
        "e_final_delta_vs_b4": e_final_delta,
        "e_angle_delta_vs_b4": e_angle_delta,
        "e_distance_delta_vs_b4": e_distance_delta,
        "distance_compensated_gain": distance_compensated,
        "completed_controls": [row["model_id"] for row in controls_present],
    }
    if len(controls_present) < len(STAGE2_CONTROL_RUNS):
        payload.update(
            {
                "verdict": "launch_stage2_controls_before_method_claim",
                "reason": "E gain is promising but distance-compensated; controls are required before a method claim",
                "authorized_next": "run_E64_E256_E128S1_controls",
            }
        )
        return payload

    b4_final = parse_float(b4.get("final_score"))
    control_finals = {str(row["model_id"]): parse_float(row.get("final_score")) for row in controls_present}
    seed_ok = control_finals.get("E128S1") is not None and control_finals["E128S1"] <= b4_final + 0.005
    compact_ok = control_finals.get("E64") is not None and control_finals["E64"] <= b4_final + 0.005
    capacity_only = control_finals.get("E256") is not None and control_finals.get("E64") is not None and control_finals["E256"] + 0.003 < control_finals["E64"]
    if seed_ok and compact_ok and not capacity_only:
        payload.update(
            {
                "verdict": "promote_to_stage3_secondary_probe_or_longer_E_validation",
                "reason": "E survives seed/compact bottleneck controls without a clear capacity-only signature",
                "authorized_next": "choose_one_secondary_probe_or_longer_E_validation",
            }
        )
    elif seed_ok or compact_ok:
        payload.update(
            {
                "verdict": "repeat_with_one_missing_control",
                "reason": "some E controls support MSR but the pattern is not stable enough",
                "authorized_next": "repeat_or_add_evidence_mask_control",
            }
        )
    else:
        payload.update(
            {
                "verdict": "downgrade_to_diagnostic_route",
                "reason": "E does not survive compact/seed controls",
                "authorized_next": "analyze_failure_buckets_before_more_training",
            }
        )
    return payload


def write_summary(
    path: Path,
    stage1_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    quadrant_rows: list[dict[str, object]],
    conflict: list[dict[str, object]],
    verdict_payload: dict[str, object],
) -> None:
    by_id = metric_by_id(stage1_rows + control_rows)
    e = by_id.get("E", {})
    b4 = by_id.get("B4", {})
    b3 = by_id.get("B3", {})
    e_b4_pairs = [r for r in pair_rows if r["model_id"] == "E" and r["reference_id"] == "B4"]
    e_b4_by_metric = {r["metric"]: r for r in e_b4_pairs}
    q_e_b4 = [r for r in quadrant_rows if r["model_id"] == "E" and r["reference_id"] == "B4"]
    core_conflict = [r for r in conflict if r["subset"] == "core_b4_angle_b3_distance_conflict"]

    lines = [
        "# Phase92 Stage2 MSR Diagnostics",
        "",
        f"- created_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        f"- verdict: `{verdict_payload.get('verdict')}`",
        f"- reason: {verdict_payload.get('reason')}",
        f"- authorized_next: `{verdict_payload.get('authorized_next')}`",
        "",
        "## E Versus Anchors",
        "",
        "| comparison | final | angle | distance |",
        "|---|---:|---:|---:|",
        (
            "| E | "
            f"{fmt(parse_float(e.get('final_score')))} | "
            f"{fmt(parse_float(e.get('angle_rel_error')))} | "
            f"{fmt(parse_float(e.get('distance_rel_error')))} |"
        ),
        (
            "| B4 | "
            f"{fmt(parse_float(b4.get('final_score')))} | "
            f"{fmt(parse_float(b4.get('angle_rel_error')))} | "
            f"{fmt(parse_float(b4.get('distance_rel_error')))} |"
        ),
        (
            "| B3 | "
            f"{fmt(parse_float(b3.get('final_score')))} | "
            f"{fmt(parse_float(b3.get('angle_rel_error')))} | "
            f"{fmt(parse_float(b3.get('distance_rel_error')))} |"
        ),
        (
            "| E - B4 | "
            f"{fmt(verdict_payload.get('e_final_delta_vs_b4'))} | "
            f"{fmt(verdict_payload.get('e_angle_delta_vs_b4'))} | "
            f"{fmt(verdict_payload.get('e_distance_delta_vs_b4'))} |"
        ),
        "",
        "## E Pairwise Versus B4",
        "",
        "| metric | mean_delta | better_rate | material_regression_rate |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = e_b4_by_metric.get(metric, {})
        lines.append(
            "| "
            f"{metric} | {fmt(row.get('mean_delta_model_minus_ref'))} | "
            f"{fmt(row.get('better_rate'))} | {fmt(row.get('material_regression_rate'))} |"
        )
    lines.extend(["", "## E/B4 Angle-Distance Quadrants", "", "| quadrant | count | rate | final_delta | angle_delta | distance_delta |", "|---|---:|---:|---:|---:|---:|"])
    for row in q_e_b4:
        lines.append(
            "| "
            f"{row['quadrant']} | {row['count']} | {fmt(row.get('rate'))} | "
            f"{fmt(row.get('final_score_mean_delta'))} | "
            f"{fmt(row.get('angle_rel_error_mean_delta'))} | "
            f"{fmt(row.get('distance_rel_error_mean_delta'))} |"
        )
    lines.extend(["", "## Core Anchor Conflict", "", "| metric | count | best_model | best_value | B4_mean | B3_mean | E_mean |", "|---|---:|---|---:|---:|---:|---:|"])
    for row in core_conflict:
        lines.append(
            "| "
            f"{row['metric']} | {row['sample_count']} | {row['best_model']} | {fmt(row.get('best_value'))} | "
            f"{fmt(row.get('B4_mean'))} | {fmt(row.get('B3_mean'))} | {fmt(row.get('E_mean'))} |"
        )
    lines.extend(["", "## Control Rows", "", "| row | exists | final | angle | distance | claim |", "|---|---|---:|---:|---:|---|"])
    for row in control_rows:
        lines.append(
            "| "
            f"{row['model_id']} | {row.get('metrics_exists')} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} | "
            f"{row.get('claim')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage1 E is a real local signal, but the aggregate win over B4 is distance-compensated rather than angle-preserving.",
            "- C and D failing alone make a pure single-factor explanation weak, but they do not rule out seed noise or capacity effects.",
            "- The next valid step is the E64/E256/E128S1 control panel before any method-level or leaderboard migration claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase92-root", type=Path, default=Path("/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase92_minimum_sufficient_relation_feasibility_v1"))
    parser.add_argument("--control-run-group", default="stage2_msr_e_controls")
    parser.add_argument("--max-train-steps", type=int, default=2500)
    parser.add_argument("--eval-max-samples", type=int, default=811)
    parser.add_argument("--lr", default="5e-6")
    parser.add_argument("--bucket-min-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase92_root: Path = args.phase92_root
    stage1_rows = load_stage1_metric_rows(phase92_root)
    control_rows = load_control_metric_rows(
        phase92_root,
        args.control_run_group,
        args.max_train_steps,
        args.eval_max_samples,
        args.lr,
    )
    all_rows = stage1_rows + control_rows
    samples = load_samples(all_rows)
    stage1_common = common_ids(samples, REFERENCE_IDS + STAGE1_IDS)
    model_ids = [mid for mid in REFERENCE_IDS + STAGE1_IDS + [str(r["row_id"]) for r in STAGE2_CONTROL_RUNS] if mid in samples]

    pair_rows = pairwise_summary(samples, model_ids, ["B1", "B3", "B4"], stage1_common)
    quadrant_rows: list[dict[str, object]] = []
    for model in [mid for mid in model_ids if mid not in REFERENCE_IDS]:
        for ref in ["B4", "B3"]:
            quadrant_rows.extend(quadrant_summary(samples, model, ref, stage1_common))
    conflict = conflict_rows(samples, stage1_common, [mid for mid in ["B1", "B3", "B4", "C", "D", "E"] if mid in samples])
    bucket_rows = bucket_matrix_rows(samples, stage1_common, [mid for mid in ["B1", "B3", "B4", "C", "D", "E"] if mid in samples], args.bucket_min_count)
    verdict_payload = stage2_verdict(stage1_rows, control_rows)
    verdict_payload.update(
        {
            "phase_id": "phase92-minimum-sufficient-relation-feasibility",
            "stage": "stage2_conflict_and_sufficiency_diagnostics",
            "stage1_common_sample_count": len(stage1_common),
            "control_run_group": args.control_run_group,
        }
    )

    diag_dir = phase92_root / "diagnostics"
    abl_dir = phase92_root / "ablations"
    reports_dir = phase92_root / "reports"
    manifests_dir = phase92_root / "manifests"

    write_csv(
        abl_dir / "msr_stage2_control_metrics.csv",
        control_rows,
        [
            "model_id",
            "model_name",
            "family",
            "claim",
            "output_mode",
            "msr_bottleneck_dim",
            "train_seed",
            "data_seed",
            "eval_dir",
            "metrics_exists",
            "final_score",
            "angle_rel_error",
            "distance_rel_error",
        ],
    )
    write_csv(
        abl_dir / "msr_stage2_pairwise_vs_refs.csv",
        pair_rows,
        [
            "model_id",
            "reference_id",
            "metric",
            "valid_count",
            "mean_delta_model_minus_ref",
            "better_count",
            "worse_count",
            "tie_count",
            "better_rate",
            "material_regression_count",
            "material_regression_rate",
        ],
    )
    write_csv(
        diag_dir / "msr_stage2_angle_distance_quadrants.csv",
        quadrant_rows,
        [
            "model_id",
            "reference_id",
            "quadrant",
            "count",
            "rate",
            "final_score_mean_delta",
            "angle_rel_error_mean_delta",
            "distance_rel_error_mean_delta",
        ],
    )
    write_csv(
        diag_dir / "msr_stage2_conflict_subsets.csv",
        conflict,
        [
            "subset",
            "metric",
            "sample_count",
            "best_model",
            "best_value",
            "B1_mean",
            "B3_mean",
            "B4_mean",
            "C_mean",
            "D_mean",
            "E_mean",
        ],
    )
    write_csv(
        diag_dir / "msr_stage2_bucket_matrix.csv",
        bucket_rows,
        [
            "angle_gt_bucket",
            "gt_distance_bucket",
            "metric",
            "sample_count",
            "best_model",
            "best_value",
            "E_mean",
            "B4_mean",
            "B3_mean",
        ],
    )
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "phase92_stage2_diagnostics_verdict.json").write_text(
        json.dumps(verdict_payload, indent=2),
        encoding="utf-8",
    )
    write_summary(
        reports_dir / "phase92_stage2_diagnostics_summary.md",
        stage1_rows,
        control_rows,
        pair_rows,
        quadrant_rows,
        conflict,
        verdict_payload,
    )
    print(json.dumps(verdict_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
