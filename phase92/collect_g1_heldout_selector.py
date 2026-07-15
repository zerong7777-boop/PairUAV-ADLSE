#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path


METRICS = ["final_score", "angle_rel_error", "distance_rel_error"]
SELECTOR_NAME = "b4_pred_distance_bucket"
SOURCE_MODEL_IDS = ["B3", "B4", "E", "E256"]
ANCHOR_MODEL_ID = "B4"

MODEL_SPECS = {
    "B3": {
        "family": "phase91_baseline",
        "role": "distance_side_anchor",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B3_matched_steps2500_val811_lr5e-06",
    },
    "B4": {
        "family": "phase91_baseline",
        "role": "h8_mid_late_angle_anchor",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B4_matched_steps2500_val811_lr5e-06",
    },
    "E": {
        "family": "phase92_stage1",
        "role": "two_bottlenecks_static_split",
        "eval_rel": "phase92:ablations/stage1_msr_cde_matched_eval/phase92_E_stage1_msr_cde_matched_steps2500_val811_lr5e-6",
    },
    "E256": {
        "family": "phase92_stage2_control",
        "role": "capacity_control",
        "eval_rel": "phase92:ablations/stage2_msr_e_controls_eval/phase92_E256_stage2_msr_e_controls_steps2500_val811_lr5e-6_seed0_bdim256",
    },
}


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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resolve_eval_dir(spec: dict[str, str], phase91_root: Path, phase92_root: Path) -> Path:
    prefix, rel = spec["eval_rel"].split(":", 1)
    if prefix == "phase91":
        return phase91_root / rel
    if prefix == "phase92":
        return phase92_root / rel
    raise ValueError(f"unknown eval prefix: {prefix}")


def load_source_rows(
    phase91_root: Path,
    phase92_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]], list[dict[str, object]]]:
    metrics: dict[str, dict[str, str]] = {}
    samples: dict[str, dict[str, dict[str, str]]] = {}
    manifest_rows: list[dict[str, object]] = []
    for model_id, spec in MODEL_SPECS.items():
        eval_dir = resolve_eval_dir(spec, phase91_root, phase92_root)
        metrics_path = eval_dir / "official_metrics.json"
        sample_path = eval_dir / "official_per_sample.csv"
        result_path = eval_dir / "result.txt"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        sample_rows = {row["sample_id"]: row for row in read_csv(sample_path)} if sample_path.exists() else {}
        metrics[model_id] = metrics_payload
        samples[model_id] = sample_rows
        manifest_rows.append(
            {
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "eval_dir": str(eval_dir),
                "metrics_exists": metrics_path.exists(),
                "per_sample_exists": sample_path.exists(),
                "result_exists": result_path.exists(),
                "sample_count": len(sample_rows),
                **{metric: metrics_payload.get(metric, "") for metric in METRICS},
            }
        )
    return metrics, samples, manifest_rows


def common_ids(samples: dict[str, dict[str, dict[str, str]]], model_ids: list[str]) -> list[str]:
    present = [samples[model_id] for model_id in model_ids if model_id in samples]
    if len(present) != len(model_ids) or not present:
        return []
    return sorted(set.intersection(*(set(rows) for rows in present)))


def fold_id(sample_id: str) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def metric_value(samples: dict[str, dict[str, dict[str, str]]], model_id: str, sample_id: str, metric: str) -> float | None:
    return parse_float(samples[model_id][sample_id].get(metric))


def best_model_for_ids(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    sample_ids: list[str],
    fallback: str = "",
) -> tuple[str, float | None, int]:
    best_model = fallback
    best_value: float | None = None
    best_count = 0
    for model_id in model_ids:
        values = [
            value
            for sid in sample_ids
            if (value := metric_value(samples, model_id, sid, "final_score")) is not None
        ]
        value = mean(values)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_model = model_id
            best_value = value
            best_count = len(values)
    return best_model, best_value, best_count


def selector_key(samples: dict[str, dict[str, dict[str, str]]], sample_id: str) -> str:
    return samples[ANCHOR_MODEL_ID][sample_id].get("pred_distance_bucket", "missing") or "missing"


def train_mapping_for_fold(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    train_ids: list[str],
    min_train_count: int,
) -> tuple[dict[str, str], list[dict[str, object]], str]:
    fallback, fallback_value, fallback_count = best_model_for_ids(samples, model_ids, train_ids)
    by_key: dict[str, list[str]] = {}
    for sid in train_ids:
        by_key.setdefault(selector_key(samples, sid), []).append(sid)

    mapping: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for key, ids in sorted(by_key.items()):
        selected, selected_value, selected_count = best_model_for_ids(samples, model_ids, ids, fallback=fallback)
        if len(ids) < min_train_count:
            selected = fallback
            selected_value = fallback_value
            selected_count = fallback_count
        mapping[key] = selected
        row: dict[str, object] = {
            "selector_key": key,
            "train_sample_count": len(ids),
            "selected_model": selected,
            "selected_train_final": selected_value,
            "selected_train_valid_count": selected_count,
            "fallback_model": fallback,
            "used_fallback": len(ids) < min_train_count,
        }
        for model_id in model_ids:
            values = [
                value
                for sid in ids
                if (value := metric_value(samples, model_id, sid, "final_score")) is not None
            ]
            row[f"{model_id}_train_final"] = mean(values)
            row[f"{model_id}_train_valid_count"] = len(values)
        rows.append(row)
    return mapping, rows, fallback


def selected_metrics(selected_rows: list[dict[str, object]]) -> dict[str, object]:
    angle_values: list[float] = []
    distance_values: list[float] = []
    final_values: list[float] = []
    for row in selected_rows:
        angle = parse_float(row.get("angle_rel_error"))
        distance = parse_float(row.get("distance_rel_error"))
        final = parse_float(row.get("final_score"))
        if angle is not None:
            angle_values.append(angle)
        if distance is not None:
            distance_values.append(distance)
        if final is not None:
            final_values.append(final)
    return {
        "angle_rel_error": mean(angle_values),
        "angle_valid_count": len(angle_values),
        "distance_rel_error": mean(distance_values),
        "distance_valid_count": len(distance_values),
        "final_score": mean(final_values),
        "sample_final_valid_count": len(final_values),
        "num_joined_rows": len(selected_rows),
        "num_manifest_rows": len(selected_rows),
        "num_prediction_rows": len(selected_rows),
    }


def source_metric_summary(metrics: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for model_id in SOURCE_MODEL_IDS:
        payload = metrics[model_id]
        rows.append(
            {
                "model_id": model_id,
                "final_score": payload.get("final_score", ""),
                "angle_rel_error": payload.get("angle_rel_error", ""),
                "distance_rel_error": payload.get("distance_rel_error", ""),
            }
        )
    return rows


def write_result_txt(path: Path, selected_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{parse_float(row['pred_angle']):.6f} {parse_float(row['pred_distance']):.6f}"
        for row in selected_rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selector(
    samples: dict[str, dict[str, dict[str, str]]],
    ordered_sample_ids: list[str],
    min_train_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    selected_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for eval_fold in [0, 1]:
        train_ids = [sid for sid in ordered_sample_ids if fold_id(sid) != eval_fold]
        eval_ids = [sid for sid in ordered_sample_ids if fold_id(sid) == eval_fold]
        mapping, fold_mapping_rows, fallback = train_mapping_for_fold(samples, SOURCE_MODEL_IDS, train_ids, min_train_count)
        for row in fold_mapping_rows:
            row["eval_fold"] = eval_fold
            row["train_count"] = len(train_ids)
            row["eval_count"] = len(eval_ids)
            mapping_rows.append(row)

        fold_selected: list[dict[str, object]] = []
        for sid in eval_ids:
            key = selector_key(samples, sid)
            selected_model = mapping.get(key, fallback)
            source = samples[selected_model][sid]
            selected = {
                "sample_id": sid,
                "eval_fold": eval_fold,
                "selector_name": SELECTOR_NAME,
                "selector_key": key,
                "selected_model": selected_model,
                "gt_angle": source.get("gt_angle", ""),
                "gt_distance": source.get("gt_distance", ""),
                "pred_angle": source.get("pred_angle", ""),
                "pred_distance": source.get("pred_distance", ""),
                "pred_angle_norm": source.get("pred_angle_norm", ""),
                "gt_angle_norm": source.get("gt_angle_norm", ""),
                "angle_abs_error": source.get("angle_abs_error", ""),
                "angle_rel_error": source.get("angle_rel_error", ""),
                "distance_abs_error": source.get("distance_abs_error", ""),
                "distance_rel_error": source.get("distance_rel_error", ""),
                "final_score": source.get("final_score", ""),
                "distance_valid": source.get("distance_valid", ""),
                "angle_valid": source.get("angle_valid", ""),
                "gt_distance_bucket": source.get("gt_distance_bucket", ""),
                "pred_distance_bucket": source.get("pred_distance_bucket", ""),
                "angle_gt_bucket": source.get("angle_gt_bucket", ""),
                "b4_pred_distance_bucket": key,
            }
            selected_rows.append(selected)
            fold_selected.append(selected)
        fold_metrics = selected_metrics(fold_selected)
        fold_rows.append(
            {
                "eval_fold": eval_fold,
                "train_count": len(train_ids),
                "eval_count": len(eval_ids),
                "fallback_model": fallback,
                **fold_metrics,
            }
        )
    order = {sid: idx for idx, sid in enumerate(ordered_sample_ids)}
    selected_rows.sort(key=lambda row: order[str(row["sample_id"])])
    return selected_rows, mapping_rows, fold_rows


def count_selected_models(selected_rows: list[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in selected_rows:
        model = str(row["selected_model"])
        out[model] = out.get(model, 0) + 1
    return dict(sorted(out.items()))


def verdict_payload(
    metrics: dict[str, object],
    source_metrics: dict[str, dict[str, str]],
    selected_rows: list[dict[str, object]],
    min_train_count: int,
) -> dict[str, object]:
    best_single = min(
        SOURCE_MODEL_IDS,
        key=lambda model: parse_float(source_metrics[model].get("final_score")) or float("inf"),
    )
    best_single_final = parse_float(source_metrics[best_single].get("final_score"))
    final_score = parse_float(metrics.get("final_score"))
    gain = final_score - best_single_final if final_score is not None and best_single_final is not None else None
    verdict = "selector_smoke_diagnostic_only"
    reason = "held-out selector did not clearly beat the best single expert"
    authorized_next = "review_buckets_or_stop_selector_route"
    if gain is not None and gain <= -0.002:
        verdict = "heldout_selector_positive"
        reason = "cross-fit held-out selector beats the best single expert on fixed val811"
        authorized_next = "export_full_val_calibrated_selector_or_minimal_testtime_router"
    return {
        "phase_id": "phase92-minimum-sufficient-relation-feasibility",
        "stage": "g1_heldout_selector_smoke",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selector_name": SELECTOR_NAME,
        "candidate_models": SOURCE_MODEL_IDS,
        "min_train_count": min_train_count,
        "verdict": verdict,
        "reason": reason,
        "authorized_next": authorized_next,
        "best_single_model": best_single,
        "best_single_final_score": best_single_final,
        "heldout_selector_final_score": final_score,
        "heldout_selector_angle_rel_error": metrics.get("angle_rel_error"),
        "heldout_selector_distance_rel_error": metrics.get("distance_rel_error"),
        "heldout_selector_gain_vs_best_single": gain,
        "selected_model_counts": count_selected_models(selected_rows),
        "uses_official_test": False,
    }


def write_summary(
    path: Path,
    verdict: dict[str, object],
    source_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Phase92-G1 Held-Out Selector Smoke",
        "",
        f"- created_at: `{verdict.get('created_at')}`",
        f"- selector: `{SELECTOR_NAME}`",
        f"- candidate_models: `{','.join(SOURCE_MODEL_IDS)}`",
        f"- verdict: `{verdict.get('verdict')}`",
        f"- reason: {verdict.get('reason')}",
        f"- authorized_next: `{verdict.get('authorized_next')}`",
        f"- uses_official_test: `{verdict.get('uses_official_test')}`",
        "",
        "## Source Experts",
        "",
        "| model | final | angle | distance |",
        "|---|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            "| "
            f"{row['model_id']} | {fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Selector Result",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| final_score | {fmt(verdict.get('heldout_selector_final_score'))} |",
            f"| angle_rel_error | {fmt(verdict.get('heldout_selector_angle_rel_error'))} |",
            f"| distance_rel_error | {fmt(verdict.get('heldout_selector_distance_rel_error'))} |",
            f"| gain_vs_best_single | {fmt(verdict.get('heldout_selector_gain_vs_best_single'))} |",
            "",
            "## Fold Results",
            "",
            "| eval_fold | train_count | eval_count | fallback | final | angle | distance |",
            "|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            "| "
            f"{row['eval_fold']} | {row['train_count']} | {row['eval_count']} | {row['fallback_model']} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a cross-fit fixed-val811 smoke: each sample is evaluated by a bucket-to-expert rule learned from the opposite fold.",
            "- It does not use official hidden test data or leaderboard feedback.",
            "- A positive result authorizes a deployable selector export path, not a paper-level method claim by itself.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase91-root", type=Path, default=Path("/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase91_polarrel_problem_mechanism_validation_v1"))
    parser.add_argument("--phase92-root", type=Path, default=Path("/media/jgzn/SSD_lexar/RZ/UAVM/runs/phase92_minimum_sufficient_relation_feasibility_v1"))
    parser.add_argument("--min-train-count", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.phase92_root / "selector_g1" / "heldout_b4_pred_distance_bucket"
    output_root.mkdir(parents=True, exist_ok=True)

    source_metrics, samples, source_manifest = load_source_rows(args.phase91_root, args.phase92_root)
    ordered_sample_ids = common_ids(samples, SOURCE_MODEL_IDS)
    if not ordered_sample_ids:
        raise SystemExit("no common sample ids for source experts")

    selected_rows, mapping_rows, fold_rows = run_selector(samples, ordered_sample_ids, args.min_train_count)
    metrics = selected_metrics(selected_rows)
    source_rows = source_metric_summary(source_metrics)
    verdict = verdict_payload(metrics, source_metrics, selected_rows, args.min_train_count)

    selected_fields = [
        "sample_id",
        "eval_fold",
        "selector_name",
        "selector_key",
        "selected_model",
        "gt_angle",
        "gt_distance",
        "pred_angle",
        "pred_distance",
        "pred_angle_norm",
        "gt_angle_norm",
        "angle_abs_error",
        "angle_rel_error",
        "distance_abs_error",
        "distance_rel_error",
        "final_score",
        "distance_valid",
        "angle_valid",
        "gt_distance_bucket",
        "pred_distance_bucket",
        "angle_gt_bucket",
        "b4_pred_distance_bucket",
    ]
    write_csv(output_root / "selected_per_sample.csv", selected_rows, selected_fields)
    write_result_txt(output_root / "result.txt", selected_rows)
    write_csv(
        output_root / "selector_mapping_by_fold.csv",
        mapping_rows,
        [
            "eval_fold",
            "train_count",
            "eval_count",
            "selector_key",
            "train_sample_count",
            "selected_model",
            "selected_train_final",
            "selected_train_valid_count",
            "fallback_model",
            "used_fallback",
            "B3_train_final",
            "B3_train_valid_count",
            "B4_train_final",
            "B4_train_valid_count",
            "E_train_final",
            "E_train_valid_count",
            "E256_train_final",
            "E256_train_valid_count",
        ],
    )
    write_csv(
        output_root / "fold_metrics.csv",
        fold_rows,
        [
            "eval_fold",
            "train_count",
            "eval_count",
            "fallback_model",
            "final_score",
            "sample_final_valid_count",
            "angle_rel_error",
            "angle_valid_count",
            "distance_rel_error",
            "distance_valid_count",
            "num_joined_rows",
        ],
    )
    write_csv(
        output_root / "source_model_metrics.csv",
        source_rows,
        ["model_id", "final_score", "angle_rel_error", "distance_rel_error"],
    )
    write_csv(
        output_root / "source_model_manifest.csv",
        source_manifest,
        [
            "model_id",
            "family",
            "role",
            "eval_dir",
            "metrics_exists",
            "per_sample_exists",
            "result_exists",
            "sample_count",
            "final_score",
            "angle_rel_error",
            "distance_rel_error",
        ],
    )
    (output_root / "official_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "selector_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    write_summary(output_root / "summary.md", verdict, source_rows, fold_rows)

    report_root = args.phase92_root / "reports"
    manifest_root = args.phase92_root / "manifests"
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_summary(report_root / "phase92_g1_heldout_selector_summary.md", verdict, source_rows, fold_rows)
    (manifest_root / "phase92_g1_heldout_selector_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
