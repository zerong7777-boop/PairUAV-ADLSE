#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

DISTANCE_BUCKETS = [
    ("d_00_zero", "abs(distance) == 0"),
    ("d_01_le_1", "0 < abs(distance) <= 1"),
    ("d_02_le_5", "1 < abs(distance) <= 5"),
    ("d_03_le_10", "5 < abs(distance) <= 10"),
    ("d_04_le_25", "10 < abs(distance) <= 25"),
    ("d_05_le_50", "25 < abs(distance) <= 50"),
    ("d_06_le_100", "50 < abs(distance) <= 100"),
    ("d_07_gt_100", "abs(distance) > 100"),
]


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


def load_sources(
    phase91_root: Path,
    phase92_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, str]]], dict[str, dict[str, dict[str, str]]]]:
    manifests: dict[str, dict[str, object]] = {}
    ordered_rows: dict[str, list[dict[str, str]]] = {}
    sample_maps: dict[str, dict[str, dict[str, str]]] = {}
    for model_id, spec in MODEL_SPECS.items():
        eval_dir = resolve_eval_dir(spec, phase91_root, phase92_root)
        metrics_path = eval_dir / "official_metrics.json"
        sample_path = eval_dir / "official_per_sample.csv"
        result_path = eval_dir / "result.txt"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        rows = read_csv(sample_path) if sample_path.exists() else []
        ordered_rows[model_id] = rows
        sample_maps[model_id] = {row["sample_id"]: row for row in rows}
        manifests[model_id] = {
            "model_id": model_id,
            "family": spec["family"],
            "role": spec["role"],
            "eval_dir": str(eval_dir),
            "result_txt": str(result_path),
            "metrics_exists": metrics_path.exists(),
            "per_sample_exists": sample_path.exists(),
            "result_exists": result_path.exists(),
            "sample_count": len(rows),
            **{metric: metrics.get(metric, "") for metric in METRICS},
        }
    return manifests, ordered_rows, sample_maps


def common_ordered_ids(ordered_rows: dict[str, list[dict[str, str]]], sample_maps: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    anchor_rows = ordered_rows[ANCHOR_MODEL_ID]
    ids = [row["sample_id"] for row in anchor_rows]
    return [sid for sid in ids if all(sid in sample_maps[model_id] for model_id in SOURCE_MODEL_IDS)]


def metric_value(sample_maps: dict[str, dict[str, dict[str, str]]], model_id: str, sample_id: str, metric: str) -> float | None:
    return parse_float(sample_maps[model_id][sample_id].get(metric))


def best_model_for_ids(
    sample_maps: dict[str, dict[str, dict[str, str]]],
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
            if (value := metric_value(sample_maps, model_id, sid, "final_score")) is not None
        ]
        value = mean(values)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_model = model_id
            best_value = value
            best_count = len(values)
    return best_model, best_value, best_count


def selector_key(sample_maps: dict[str, dict[str, dict[str, str]]], sample_id: str) -> str:
    return sample_maps[ANCHOR_MODEL_ID][sample_id].get("pred_distance_bucket", "missing") or "missing"


def build_full_mapping(
    sample_maps: dict[str, dict[str, dict[str, str]]],
    sample_ids: list[str],
    min_train_count: int,
) -> tuple[dict[str, str], list[dict[str, object]], str]:
    fallback, fallback_value, fallback_count = best_model_for_ids(sample_maps, SOURCE_MODEL_IDS, sample_ids)
    by_key: dict[str, list[str]] = {}
    for sid in sample_ids:
        by_key.setdefault(selector_key(sample_maps, sid), []).append(sid)

    mapping: dict[str, str] = {}
    mapping_rows: list[dict[str, object]] = []
    for key, ids in sorted(by_key.items()):
        selected_model, selected_value, selected_count = best_model_for_ids(sample_maps, SOURCE_MODEL_IDS, ids, fallback=fallback)
        used_fallback = False
        if len(ids) < min_train_count:
            selected_model = fallback
            selected_value = fallback_value
            selected_count = fallback_count
            used_fallback = True
        mapping[key] = selected_model
        row: dict[str, object] = {
            "selector_key": key,
            "sample_count": len(ids),
            "selected_model": selected_model,
            "selected_final": selected_value,
            "selected_valid_count": selected_count,
            "fallback_model": fallback,
            "used_fallback": used_fallback,
        }
        for model_id in SOURCE_MODEL_IDS:
            values = [
                value
                for sid in ids
                if (value := metric_value(sample_maps, model_id, sid, "final_score")) is not None
            ]
            row[f"{model_id}_final"] = mean(values)
            row[f"{model_id}_valid_count"] = len(values)
        mapping_rows.append(row)
    return mapping, mapping_rows, fallback


def selected_metrics(selected_rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for metric in METRICS:
        values = [value for row in selected_rows if (value := parse_float(row.get(metric))) is not None]
        out[metric] = mean(values)
        out[f"{metric}_valid_count"] = len(values)
    out["angle_valid_count"] = out.pop("angle_rel_error_valid_count")
    out["distance_valid_count"] = out.pop("distance_rel_error_valid_count")
    out["sample_final_valid_count"] = out.pop("final_score_valid_count")
    out["num_joined_rows"] = len(selected_rows)
    out["num_manifest_rows"] = len(selected_rows)
    out["num_prediction_rows"] = len(selected_rows)
    return out


def apply_mapping(
    sample_maps: dict[str, dict[str, dict[str, str]]],
    sample_ids: list[str],
    mapping: dict[str, str],
    fallback_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sid in sample_ids:
        key = selector_key(sample_maps, sid)
        selected_model = mapping.get(key, fallback_model)
        source = sample_maps[selected_model][sid]
        rows.append(
            {
                "sample_id": sid,
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
        )
    return rows


def write_result_txt(path: Path, selected_rows: list[dict[str, object]]) -> None:
    lines = [
        f"{parse_float(row['pred_angle']):.6f} {parse_float(row['pred_distance']):.6f}"
        for row in selected_rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_metric_rows(manifests: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "model_id": model_id,
            "family": manifests[model_id]["family"],
            "role": manifests[model_id]["role"],
            "final_score": manifests[model_id].get("final_score", ""),
            "angle_rel_error": manifests[model_id].get("angle_rel_error", ""),
            "distance_rel_error": manifests[model_id].get("distance_rel_error", ""),
        }
        for model_id in SOURCE_MODEL_IDS
    ]


def verdict_payload(metrics: dict[str, object], manifests: dict[str, dict[str, object]], mapping: dict[str, str], selected_rows: list[dict[str, object]], min_train_count: int) -> dict[str, object]:
    best_single = min(
        SOURCE_MODEL_IDS,
        key=lambda model: parse_float(manifests[model].get("final_score")) or float("inf"),
    )
    best_single_final = parse_float(manifests[best_single].get("final_score"))
    final_score = parse_float(metrics.get("final_score"))
    gain = final_score - best_single_final if final_score is not None and best_single_final is not None else None
    return {
        "phase_id": "phase92-minimum-sufficient-relation-feasibility",
        "stage": "g2_frozen_selector_export",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selector_name": SELECTOR_NAME,
        "candidate_models": SOURCE_MODEL_IDS,
        "trained_on": "fixed_val811_full",
        "min_train_count": min_train_count,
        "verdict": "frozen_selector_export_ready",
        "reason": "full-val811 frozen selector mapping was exported and replayed",
        "authorized_next": "apply_to_unlabeled_same_order_expert_predictions_or_build_minimal_testtime_router",
        "best_single_model": best_single,
        "best_single_final_score": best_single_final,
        "frozen_selector_final_score": final_score,
        "frozen_selector_angle_rel_error": metrics.get("angle_rel_error"),
        "frozen_selector_distance_rel_error": metrics.get("distance_rel_error"),
        "frozen_selector_gain_vs_best_single": gain,
        "selected_model_counts": count_selected(selected_rows),
        "bucket_to_model": mapping,
        "uses_official_test": False,
    }


def count_selected(selected_rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in selected_rows:
        model = str(row["selected_model"])
        counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items()))


def frozen_mapping_payload(
    mapping: dict[str, str],
    mapping_rows: list[dict[str, object]],
    fallback_model: str,
    min_train_count: int,
    manifests: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selector_name": SELECTOR_NAME,
        "selector_key_source": "B4 pred_distance bucket derived from B4 predicted distance",
        "trained_on": "fixed_val811_full",
        "candidate_models": SOURCE_MODEL_IDS,
        "fallback_model": fallback_model,
        "min_train_count": min_train_count,
        "bucket_to_model": mapping,
        "distance_bucket_definition": dict(DISTANCE_BUCKETS),
        "source_result_paths": {model_id: manifests[model_id]["result_txt"] for model_id in SOURCE_MODEL_IDS},
        "bucket_rows": mapping_rows,
        "uses_official_test": False,
    }


def write_summary(path: Path, verdict: dict[str, object], source_rows: list[dict[str, object]], mapping_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Phase92-G2 Frozen Selector Export",
        "",
        f"- created_at: `{verdict.get('created_at')}`",
        f"- selector: `{SELECTOR_NAME}`",
        f"- candidate_models: `{','.join(SOURCE_MODEL_IDS)}`",
        f"- trained_on: `{verdict.get('trained_on')}`",
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
            "## Frozen Selector Replay",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| final_score | {fmt(verdict.get('frozen_selector_final_score'))} |",
            f"| angle_rel_error | {fmt(verdict.get('frozen_selector_angle_rel_error'))} |",
            f"| distance_rel_error | {fmt(verdict.get('frozen_selector_distance_rel_error'))} |",
            f"| gain_vs_best_single | {fmt(verdict.get('frozen_selector_gain_vs_best_single'))} |",
            "",
            "## Frozen Mapping",
            "",
            "| bucket | count | selected | used_fallback | B3 | B4 | E | E256 |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in mapping_rows:
        lines.append(
            "| "
            f"{row['selector_key']} | {row['sample_count']} | {row['selected_model']} | {row['used_fallback']} | "
            f"{fmt(parse_float(row.get('B3_final')))} | {fmt(parse_float(row.get('B4_final')))} | "
            f"{fmt(parse_float(row.get('E_final')))} | {fmt(parse_float(row.get('E256_final')))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This artifact freezes the selector rule learned from full fixed val811.",
            "- The replay score is an in-surface calibration result; the holdout evidence remains Phase92-G1.",
            "- The apply script can run on unlabeled same-order expert `result.txt` files and does not require labels.",
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
    output_root = args.phase92_root / "selector_g2" / "frozen_b4_pred_distance_bucket_val811"
    output_root.mkdir(parents=True, exist_ok=True)

    manifests, ordered_rows, sample_maps = load_sources(args.phase91_root, args.phase92_root)
    sample_ids = common_ordered_ids(ordered_rows, sample_maps)
    if not sample_ids:
        raise SystemExit("no common samples for selector sources")

    mapping, mapping_rows, fallback_model = build_full_mapping(sample_maps, sample_ids, args.min_train_count)
    selected_rows = apply_mapping(sample_maps, sample_ids, mapping, fallback_model)
    metrics = selected_metrics(selected_rows)
    source_rows = source_metric_rows(manifests)
    verdict = verdict_payload(metrics, manifests, mapping, selected_rows, args.min_train_count)
    frozen_mapping = frozen_mapping_payload(mapping, mapping_rows, fallback_model, args.min_train_count, manifests)

    selected_fields = [
        "sample_id",
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
        output_root / "selector_mapping_full.csv",
        mapping_rows,
        [
            "selector_key",
            "sample_count",
            "selected_model",
            "selected_final",
            "selected_valid_count",
            "fallback_model",
            "used_fallback",
            "B3_final",
            "B3_valid_count",
            "B4_final",
            "B4_valid_count",
            "E_final",
            "E_valid_count",
            "E256_final",
            "E256_valid_count",
        ],
    )
    write_csv(
        output_root / "source_model_metrics.csv",
        source_rows,
        ["model_id", "family", "role", "final_score", "angle_rel_error", "distance_rel_error"],
    )
    write_csv(
        output_root / "source_model_manifest.csv",
        [manifests[model_id] for model_id in SOURCE_MODEL_IDS],
        [
            "model_id",
            "family",
            "role",
            "eval_dir",
            "result_txt",
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
    (output_root / "frozen_selector_mapping.json").write_text(json.dumps(frozen_mapping, indent=2) + "\n", encoding="utf-8")
    write_summary(output_root / "summary.md", verdict, source_rows, mapping_rows)

    report_root = args.phase92_root / "reports"
    manifest_root = args.phase92_root / "manifests"
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_summary(report_root / "phase92_g2_frozen_selector_export_summary.md", verdict, source_rows, mapping_rows)
    (manifest_root / "phase92_g2_frozen_selector_export_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
