#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from phase92.stage1_modes import STAGE1_RUNS


METRICS = ["final_score", "angle_rel_error", "distance_rel_error"]
REFERENCE_ROWS = [
    {
        "id": "B1",
        "name": "ordinary two-head reference",
        "eval_dir": "baseline_replay_eval/phase91_B1_matched_steps2500_val811_lr5e-06",
    },
    {
        "id": "B3",
        "name": "distance-side reference",
        "eval_dir": "baseline_replay_eval/phase91_B3_matched_steps2500_val811_lr5e-06",
    },
    {
        "id": "B4",
        "name": "final-angle reference",
        "eval_dir": "baseline_replay_eval/phase91_B4_matched_steps2500_val811_lr5e-06",
    },
]


def parse_float(value: str | float | int | None) -> float | None:
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
    return "" if value is None else str(value)


def load_model_rows(phase91_root: Path, phase92_root: Path, run_group: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ref in REFERENCE_ROWS:
        eval_dir = phase91_root / ref["eval_dir"]
        metrics_path = eval_dir / "official_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_id": ref["id"],
                "model_name": ref["name"],
                "family": "phase91_reference",
                "eval_dir": str(eval_dir),
                "metrics_exists": metrics_path.exists(),
                **{metric: metrics.get(metric, "") for metric in METRICS},
            }
        )

    eval_root = phase92_root / "ablations" / f"{run_group}_eval"
    for spec in STAGE1_RUNS:
        prefix = f"phase92_{spec['row_id']}_{run_group}_"
        matches = sorted(eval_root.glob(f"{prefix}*/official_metrics.json"))
        metrics_path = matches[-1] if matches else None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else {}
        rows.append(
            {
                "model_id": spec["row_id"],
                "model_name": spec["name"],
                "family": "phase92_stage1",
                "claim": spec["claim"],
                "output_mode": spec["output_mode"],
                "eval_dir": str(metrics_path.parent) if metrics_path else "",
                "metrics_exists": bool(metrics_path),
                **{metric: metrics.get(metric, "") for metric in METRICS},
            }
        )
    return rows


def load_samples(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        eval_dir = Path(str(row["eval_dir"]))
        sample_path = eval_dir / "official_per_sample.csv"
        if not sample_path.exists():
            continue
        out[str(row["model_id"])] = {r["sample_id"]: r for r in read_csv(sample_path)}
    return out


def common_sample_ids(samples: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    if not samples:
        return []
    return sorted(set.intersection(*(set(v.keys()) for v in samples.values())))


def pairwise(samples: dict[str, dict[str, dict[str, str]]], ids: list[str]) -> list[dict[str, object]]:
    rows = []
    for model in ["C", "D", "E"]:
        if model not in samples:
            continue
        for ref in ["B1", "B3", "B4"]:
            if ref not in samples:
                continue
            for metric in METRICS:
                deltas: list[float] = []
                better = worse = tie = 0
                for sid in ids:
                    a = parse_float(samples[model][sid].get(metric))
                    b = parse_float(samples[ref][sid].get(metric))
                    if a is None or b is None:
                        continue
                    d = a - b
                    deltas.append(d)
                    if d < -1e-12:
                        better += 1
                    elif d > 1e-12:
                        worse += 1
                    else:
                        tie += 1
                n = len(deltas)
                rows.append(
                    {
                        "model_id": model,
                        "reference_id": ref,
                        "metric": metric,
                        "valid_count": n,
                        "mean_delta_model_minus_ref": mean(deltas),
                        "better_count": better,
                        "worse_count": worse,
                        "tie_count": tie,
                        "better_rate": better / n if n else "",
                    }
                )
    return rows


def bucket_bests(samples: dict[str, dict[str, dict[str, str]]], ids: list[str]) -> list[dict[str, object]]:
    rows = []
    model_ids = [mid for mid in ["B1", "B3", "B4", "C", "D", "E"] if mid in samples]
    for bucket_type in ["angle_gt_bucket", "gt_distance_bucket"]:
        buckets = sorted({samples["B4"][sid].get(bucket_type, "") for sid in ids})
        for bucket in buckets:
            if not bucket:
                continue
            bucket_ids = [sid for sid in ids if samples["B4"][sid].get(bucket_type) == bucket]
            for metric in METRICS:
                vals_by_model = {}
                for mid in model_ids:
                    vals = [
                        v
                        for sid in bucket_ids
                        if (v := parse_float(samples[mid][sid].get(metric))) is not None
                    ]
                    if vals:
                        vals_by_model[mid] = sum(vals) / len(vals)
                if vals_by_model:
                    best = min(vals_by_model, key=vals_by_model.get)
                    rows.append(
                        {
                            "bucket_type": bucket_type,
                            "bucket": bucket,
                            "metric": metric,
                            "best_model": best,
                            "best_value": vals_by_model[best],
                            "bucket_count": len(bucket_ids),
                        }
                    )
    return rows


def verdict(metrics_rows: list[dict[str, object]]) -> dict[str, object]:
    by_id = {str(row["model_id"]): row for row in metrics_rows}
    missing = [mid for mid in ["C", "D", "E"] if mid not in by_id or not by_id[mid].get("metrics_exists")]
    if missing:
        return {
            "verdict": "repeat_with_one_missing_control",
            "supported_mechanism": "none_supported",
            "reason": f"missing metrics for {','.join(missing)}",
        }

    b1 = by_id["B1"]
    b3 = by_id["B3"]
    b4 = by_id["B4"]
    c = by_id["C"]
    d = by_id["D"]
    e = by_id["E"]

    b1_final = float(b1["final_score"])
    b4_final = float(b4["final_score"])
    b4_angle = float(b4["angle_rel_error"])
    b3_distance = float(b3["distance_rel_error"])

    c_support = float(c["final_score"]) < b1_final
    d_support = float(d["final_score"]) < b1_final
    e_angle_preserved = float(e["angle_rel_error"]) <= b4_angle + 0.003
    e_distance_moves_to_b3 = abs(float(e["distance_rel_error"]) - b3_distance) < abs(float(b4["distance_rel_error"]) - b3_distance)
    e_final_beats_b4 = float(e["final_score"]) <= b4_final

    if e_final_beats_b4 or (e_angle_preserved and e_distance_moves_to_b3):
        return {
            "verdict": "promote_to_next_stage",
            "supported_mechanism": "both_factorization_and_evidence_split",
            "reason": "E meets final/angle-distance Stage1 promotion condition",
        }
    if c_support and d_support:
        return {
            "verdict": "repeat_with_one_missing_control",
            "supported_mechanism": "both_factorization_and_evidence_split",
            "reason": "C and D are positive but E did not satisfy promotion gate",
        }
    if c_support:
        return {
            "verdict": "repeat_with_one_missing_control",
            "supported_mechanism": "factorized_relation_variables",
            "reason": "C improves over B1 but combined evidence is not yet supported",
        }
    if d_support:
        return {
            "verdict": "repeat_with_one_missing_control",
            "supported_mechanism": "static_factor_evidence_split",
            "reason": "D improves over B1 but combined evidence is not yet supported",
        }
    return {
        "verdict": "downgrade_to_diagnostic_route",
        "supported_mechanism": "none_supported",
        "reason": "C/D/E do not beat the ordinary two-head reference",
    }


def write_summary(path: Path, metrics_rows: list[dict[str, object]], verdict_payload: dict[str, object]) -> None:
    lines = [
        "# Phase92 Stage1 MSR Smoke Summary",
        "",
        f"- created_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        f"- verdict: `{verdict_payload['verdict']}`",
        f"- supported_mechanism: `{verdict_payload['supported_mechanism']}`",
        f"- reason: {verdict_payload['reason']}",
        "",
        "## Metrics",
        "",
        "| Model | family | final | angle | distance |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['model_id']} {row['model_name']} | {row['family']} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} |"
        )
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "This is a local matched val811 mechanism smoke. It is not an official-test result and does not authorize CodaBench submission.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase91-root", type=Path, default=Path("/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase91_polarrel_problem_mechanism_validation_v1/router_smokes"))
    parser.add_argument("--phase92-root", type=Path, default=Path("/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase92_minimum_sufficient_relation_feasibility_v1"))
    parser.add_argument("--run-group", default="stage1_msr_cde_matched")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_rows = load_model_rows(args.phase91_root, args.phase92_root, args.run_group)
    samples = load_samples(metrics_rows)
    ids = common_sample_ids(samples)
    pairwise_rows = pairwise(samples, ids)
    bucket_rows = bucket_bests(samples, ids)
    verdict_payload = verdict(metrics_rows)
    verdict_payload.update(
        {
            "phase_id": "phase92-minimum-sufficient-relation-feasibility",
            "stage": "stage1_minimum_relation_baseline_panel",
            "run_group": args.run_group,
            "common_sample_count": len(ids),
        }
    )

    write_csv(
        args.phase92_root / "ablations/msr_stage1_metrics.csv",
        metrics_rows,
        ["model_id", "model_name", "family", "claim", "output_mode", "eval_dir", "metrics_exists", *METRICS],
    )
    write_csv(
        args.phase92_root / "ablations/msr_stage1_pairwise_vs_refs.csv",
        pairwise_rows,
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
        ],
    )
    write_csv(
        args.phase92_root / "diagnostics/msr_stage1_failure_buckets.csv",
        bucket_rows,
        ["bucket_type", "bucket", "metric", "best_model", "best_value", "bucket_count"],
    )
    (args.phase92_root / "manifests").mkdir(parents=True, exist_ok=True)
    (args.phase92_root / "manifests/phase92_stage1_verdict.json").write_text(
        json.dumps(verdict_payload, indent=2), encoding="utf-8"
    )
    write_summary(args.phase92_root / "reports/phase92_stage1_msr_smoke_summary.md", metrics_rows, verdict_payload)
    print(json.dumps(verdict_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

