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

MODEL_SPECS = [
    {
        "model_id": "B0",
        "family": "phase91_baseline",
        "role": "shared_head",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B0_matched_steps2500_val811_lr5e-06",
    },
    {
        "model_id": "B1",
        "family": "phase91_baseline",
        "role": "ordinary_two_head",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B1_matched_steps2500_val811_lr5e-06",
    },
    {
        "model_id": "B2",
        "family": "phase91_baseline",
        "role": "heading_branch",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B2_matched_steps2500_val811_lr5e-06",
    },
    {
        "model_id": "B3",
        "family": "phase91_baseline",
        "role": "distance_side_anchor",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B3_matched_steps2500_val811_lr5e-06",
    },
    {
        "model_id": "B4",
        "family": "phase91_baseline",
        "role": "h8_mid_late_angle_anchor",
        "eval_rel": "phase91:router_smokes/baseline_replay_eval/phase91_B4_matched_steps2500_val811_lr5e-06",
    },
    {
        "model_id": "C",
        "family": "phase92_stage1",
        "role": "two_bottlenecks_all_evidence",
        "eval_rel": "phase92:ablations/stage1_msr_cde_matched_eval/phase92_C_stage1_msr_cde_matched_steps2500_val811_lr5e-6",
    },
    {
        "model_id": "D",
        "family": "phase92_stage1",
        "role": "static_evidence_split",
        "eval_rel": "phase92:ablations/stage1_msr_cde_matched_eval/phase92_D_stage1_msr_cde_matched_steps2500_val811_lr5e-6",
    },
    {
        "model_id": "E",
        "family": "phase92_stage1",
        "role": "two_bottlenecks_static_split",
        "eval_rel": "phase92:ablations/stage1_msr_cde_matched_eval/phase92_E_stage1_msr_cde_matched_steps2500_val811_lr5e-6",
    },
    {
        "model_id": "E64",
        "family": "phase92_stage2_control",
        "role": "compact_bottleneck_control",
        "eval_rel": "phase92:ablations/stage2_msr_e_controls_eval/phase92_E64_stage2_msr_e_controls_steps2500_val811_lr5e-6_seed0_bdim64",
    },
    {
        "model_id": "E256",
        "family": "phase92_stage2_control",
        "role": "capacity_control",
        "eval_rel": "phase92:ablations/stage2_msr_e_controls_eval/phase92_E256_stage2_msr_e_controls_steps2500_val811_lr5e-6_seed0_bdim256",
    },
    {
        "model_id": "E128S1",
        "family": "phase92_stage2_control",
        "role": "seed_repeat_control",
        "eval_rel": "phase92:ablations/stage2_msr_e_controls_eval/phase92_E128S1_stage2_msr_e_controls_steps2500_val811_lr5e-6_seed1_bdim128",
    },
]

CANDIDATE_SETS = {
    "anchors": ["B1", "B3", "B4"],
    "msr_core": ["B1", "B3", "B4", "E"],
    "selector_primary": ["B3", "B4", "E", "E256"],
    "selector_with_controls": ["B3", "B4", "E", "E64", "E256"],
    "all_nonrouter": ["B0", "B1", "B2", "B3", "B4", "C", "D", "E", "E64", "E256", "E128S1"],
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
    raise ValueError(f"unknown eval root prefix: {prefix}")


def load_models(phase91_root: Path, phase92_root: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, str]]]]:
    manifest: list[dict[str, object]] = []
    samples: dict[str, dict[str, dict[str, str]]] = {}
    for spec in MODEL_SPECS:
        eval_dir = resolve_eval_dir(spec, phase91_root, phase92_root)
        metrics_path = eval_dir / "official_metrics.json"
        sample_path = eval_dir / "official_per_sample.csv"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        row: dict[str, object] = {
            "model_id": spec["model_id"],
            "family": spec["family"],
            "role": spec["role"],
            "eval_dir": str(eval_dir),
            "metrics_exists": metrics_path.exists(),
            "per_sample_exists": sample_path.exists(),
            "sample_count": "",
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric, "")
        if sample_path.exists():
            model_rows = {r["sample_id"]: r for r in read_csv(sample_path)}
            samples[spec["model_id"]] = model_rows
            row["sample_count"] = len(model_rows)
        manifest.append(row)
    return manifest, samples


def available_models(samples: dict[str, dict[str, dict[str, str]]], model_ids: list[str]) -> list[str]:
    return [mid for mid in model_ids if mid in samples]


def common_ids(samples: dict[str, dict[str, dict[str, str]]], model_ids: list[str]) -> list[str]:
    present = [samples[mid] for mid in model_ids if mid in samples]
    if len(present) != len(model_ids) or not present:
        return []
    return sorted(set.intersection(*(set(rows) for rows in present)))


def metric_value(samples: dict[str, dict[str, dict[str, str]]], model: str, sid: str, metric: str) -> float | None:
    return parse_float(samples[model][sid].get(metric))


def model_mean(samples: dict[str, dict[str, dict[str, str]]], model: str, ids: list[str], metric: str) -> tuple[float | None, int]:
    vals = [v for sid in ids if (v := metric_value(samples, model, sid, metric)) is not None]
    return mean(vals), len(vals)


def best_single(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    ids: list[str],
    metric: str,
) -> tuple[str, float | None, int]:
    best_id = ""
    best_value: float | None = None
    best_count = 0
    for model in model_ids:
        value, count = model_mean(samples, model, ids, metric)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_id, best_value, best_count = model, value, count
    return best_id, best_value, best_count


def per_sample_best(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    ids: list[str],
    metric: str,
) -> tuple[float | None, int, dict[str, int]]:
    vals: list[float] = []
    counts = {model: 0 for model in model_ids}
    for sid in ids:
        options: list[tuple[float, str]] = []
        for model in model_ids:
            value = metric_value(samples, model, sid, metric)
            if value is not None:
                options.append((value, model))
        if not options:
            continue
        value, model = min(options, key=lambda x: (x[0], x[1]))
        vals.append(value)
        counts[model] += 1
    return mean(vals), len(vals), counts


def selected_metrics(
    samples: dict[str, dict[str, dict[str, str]]],
    selected: dict[str, str],
) -> dict[str, tuple[float | None, int]]:
    out: dict[str, tuple[float | None, int]] = {}
    for metric in METRICS:
        vals: list[float] = []
        for sid, model in selected.items():
            value = metric_value(samples, model, sid, metric)
            if value is not None:
                vals.append(value)
        out[metric] = (mean(vals), len(vals))
    return out


def best_final_model_for_ids(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    ids: list[str],
    fallback: str = "",
) -> str:
    best_id, best_value, _ = best_single(samples, model_ids, ids, "final_score")
    if best_value is None:
        return fallback
    return best_id


def choose_oracle_final(
    samples: dict[str, dict[str, dict[str, str]]],
    model_ids: list[str],
    ids: list[str],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for sid in ids:
        options: list[tuple[float, str]] = []
        for model in model_ids:
            value = metric_value(samples, model, sid, "final_score")
            if value is not None:
                options.append((value, model))
        if options:
            selected[sid] = min(options, key=lambda x: (x[0], x[1]))[1]
    return selected


def count_models(selected: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in selected.values():
        counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items()))


def angle_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def bucket_angle(value: float | None) -> str:
    if value is None:
        return "missing"
    v = value % 360.0
    if abs(v) < 1e-9:
        return "a_00_zero"
    if v <= 45:
        return "a_03_le_45"
    if v <= 90:
        return "a_04_le_90"
    if v <= 180:
        return "a_05_le_180"
    return "a_06_gt_180"


def bucket_gap(value: float | None, thresholds: list[tuple[float, str]], last_name: str) -> str:
    if value is None:
        return "missing"
    v = abs(value)
    for threshold, name in thresholds:
        if v <= threshold:
            return name
    return last_name


def get_pred(samples: dict[str, dict[str, dict[str, str]]], model: str, sid: str, field: str) -> float | None:
    return parse_float(samples[model][sid].get(field))


def proxy_key(
    samples: dict[str, dict[str, dict[str, str]]],
    sid: str,
    proxy_name: str,
    candidate_models: list[str],
) -> str:
    b4 = samples["B4"][sid]
    if proxy_name == "gt_angle_distance_bucket":
        return f"{b4.get('angle_gt_bucket', 'missing')}|{b4.get('gt_distance_bucket', 'missing')}"
    if proxy_name == "b4_pred_distance_bucket":
        return b4.get("pred_distance_bucket", "missing") or "missing"
    if proxy_name == "b4_pred_angle_bucket":
        return bucket_angle(get_pred(samples, "B4", sid, "pred_angle_norm"))
    if proxy_name == "b4_pred_angle_distance_bucket":
        return f"{bucket_angle(get_pred(samples, 'B4', sid, 'pred_angle_norm'))}|{b4.get('pred_distance_bucket', 'missing')}"

    if proxy_name in {"b4_e_disagreement", "b4_e256_disagreement"}:
        other = "E" if proxy_name == "b4_e_disagreement" else "E256"
        if other not in samples:
            return "missing"
        ad = angle_diff(
            get_pred(samples, "B4", sid, "pred_angle_norm"),
            get_pred(samples, other, sid, "pred_angle_norm"),
        )
        dd = None
        b4_d = get_pred(samples, "B4", sid, "pred_distance")
        other_d = get_pred(samples, other, sid, "pred_distance")
        if b4_d is not None and other_d is not None:
            dd = abs(b4_d - other_d)
        a_bucket = bucket_gap(ad, [(1, "a_gap_le_1"), (3, "a_gap_le_3"), (5, "a_gap_le_5"), (10, "a_gap_le_10"), (20, "a_gap_le_20"), (45, "a_gap_le_45")], "a_gap_gt_45")
        d_bucket = bucket_gap(dd, [(1, "d_gap_le_1"), (3, "d_gap_le_3"), (5, "d_gap_le_5"), (10, "d_gap_le_10"), (25, "d_gap_le_25")], "d_gap_gt_25")
        return f"{a_bucket}|{d_bucket}"

    if proxy_name == "multi_model_disagreement":
        angles: list[float] = []
        distances: list[float] = []
        for model in candidate_models:
            a = get_pred(samples, model, sid, "pred_angle_norm")
            d = get_pred(samples, model, sid, "pred_distance")
            if a is not None:
                angles.append(a % 360.0)
            if d is not None:
                distances.append(d)
        max_angle_gap = None
        if len(angles) >= 2:
            gaps = [angle_diff(a, b) for i, a in enumerate(angles) for b in angles[i + 1 :]]
            max_angle_gap = max(g for g in gaps if g is not None)
        distance_span = max(distances) - min(distances) if len(distances) >= 2 else None
        a_bucket = bucket_gap(max_angle_gap, [(3, "a_span_le_3"), (5, "a_span_le_5"), (10, "a_span_le_10"), (20, "a_span_le_20"), (45, "a_span_le_45")], "a_span_gt_45")
        d_bucket = bucket_gap(distance_span, [(1, "d_span_le_1"), (3, "d_span_le_3"), (5, "d_span_le_5"), (10, "d_span_le_10"), (25, "d_span_le_25")], "d_span_gt_25")
        return f"{a_bucket}|{d_bucket}"

    raise ValueError(f"unknown proxy: {proxy_name}")


def fold_id(sid: str) -> int:
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def train_bucket_mapping(
    samples: dict[str, dict[str, dict[str, str]]],
    candidate_models: list[str],
    train_ids: list[str],
    proxy_name: str,
    min_train_count: int,
    fallback_model: str,
) -> tuple[dict[str, str], dict[str, int]]:
    by_key: dict[str, list[str]] = {}
    for sid in train_ids:
        key = proxy_key(samples, sid, proxy_name, candidate_models)
        by_key.setdefault(key, []).append(sid)

    mapping: dict[str, str] = {}
    bucket_sizes: dict[str, int] = {}
    for key, ids in by_key.items():
        bucket_sizes[key] = len(ids)
        if len(ids) < min_train_count:
            continue
        mapping[key] = best_final_model_for_ids(samples, candidate_models, ids, fallback=fallback_model)
    return mapping, bucket_sizes


def apply_bucket_mapping(
    samples: dict[str, dict[str, dict[str, str]]],
    candidate_models: list[str],
    ids: list[str],
    proxy_name: str,
    mapping: dict[str, str],
    fallback_model: str,
) -> tuple[dict[str, str], int]:
    selected: dict[str, str] = {}
    fallback_count = 0
    for sid in ids:
        key = proxy_key(samples, sid, proxy_name, candidate_models)
        model = mapping.get(key, fallback_model)
        if key not in mapping:
            fallback_count += 1
        selected[sid] = model
    return selected, fallback_count


def selector_eval(
    samples: dict[str, dict[str, dict[str, str]]],
    candidate_models: list[str],
    ids: list[str],
    proxy_name: str,
    protocol: str,
    min_train_count: int,
) -> dict[str, object]:
    fallback_model = best_final_model_for_ids(samples, candidate_models, ids)
    if protocol == "insample":
        mapping, bucket_sizes = train_bucket_mapping(samples, candidate_models, ids, proxy_name, min_train_count, fallback_model)
        selected, fallback_count = apply_bucket_mapping(samples, candidate_models, ids, proxy_name, mapping, fallback_model)
        train_bucket_count = len(bucket_sizes)
    elif protocol == "two_fold":
        selected = {}
        fallback_count = 0
        train_bucket_count = 0
        for eval_fold in [0, 1]:
            train_ids = [sid for sid in ids if fold_id(sid) != eval_fold]
            eval_ids = [sid for sid in ids if fold_id(sid) == eval_fold]
            fold_fallback = best_final_model_for_ids(samples, candidate_models, train_ids, fallback=fallback_model)
            mapping, bucket_sizes = train_bucket_mapping(samples, candidate_models, train_ids, proxy_name, min_train_count, fold_fallback)
            fold_selected, fold_fallback_count = apply_bucket_mapping(samples, candidate_models, eval_ids, proxy_name, mapping, fold_fallback)
            selected.update(fold_selected)
            fallback_count += fold_fallback_count
            train_bucket_count += len(bucket_sizes)
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    metrics = selected_metrics(samples, selected)
    counts = count_models(selected)
    return {
        "proxy_name": proxy_name,
        "protocol": protocol,
        "candidate_count": len(candidate_models),
        "candidate_models": ",".join(candidate_models),
        "eval_count": len(selected),
        "fallback_model": fallback_model,
        "fallback_count": fallback_count,
        "train_bucket_count": train_bucket_count,
        "selected_model_counts": json.dumps(counts, sort_keys=True),
        "final_score": metrics["final_score"][0],
        "final_valid_count": metrics["final_score"][1],
        "angle_rel_error": metrics["angle_rel_error"][0],
        "angle_valid_count": metrics["angle_rel_error"][1],
        "distance_rel_error": metrics["distance_rel_error"][0],
        "distance_valid_count": metrics["distance_rel_error"][1],
    }


def oracle_rows(
    samples: dict[str, dict[str, dict[str, str]]],
    set_name: str,
    model_ids: list[str],
    ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    winner_rows: list[dict[str, object]] = []
    for metric in METRICS:
        best_id, best_value, best_count = best_single(samples, model_ids, ids, metric)
        oracle_value, oracle_count, winners = per_sample_best(samples, model_ids, ids, metric)
        rows.append(
            {
                "candidate_set": set_name,
                "metric": metric,
                "candidate_models": ",".join(model_ids),
                "common_sample_count": len(ids),
                "best_single_model": best_id,
                "best_single_value": best_value,
                "best_single_valid_count": best_count,
                "oracle_value": oracle_value,
                "oracle_valid_count": oracle_count,
                "oracle_gain_vs_best_single": (oracle_value - best_value) if oracle_value is not None and best_value is not None else "",
            }
        )
        for model, count in sorted(winners.items()):
            winner_rows.append(
                {
                    "candidate_set": set_name,
                    "metric": metric,
                    "model_id": model,
                    "winner_count": count,
                    "winner_rate": count / oracle_count if oracle_count else "",
                }
            )

    final_selected = choose_oracle_final(samples, model_ids, ids)
    selected = selected_metrics(samples, final_selected)
    best_id, best_value, _ = best_single(samples, model_ids, ids, "final_score")
    rows.append(
        {
            "candidate_set": set_name,
            "metric": "final_score_selector_components",
            "candidate_models": ",".join(model_ids),
            "common_sample_count": len(ids),
            "best_single_model": best_id,
            "best_single_value": best_value,
            "best_single_valid_count": selected["final_score"][1],
            "oracle_value": selected["final_score"][0],
            "oracle_valid_count": selected["final_score"][1],
            "oracle_gain_vs_best_single": (selected["final_score"][0] - best_value) if selected["final_score"][0] is not None and best_value is not None else "",
            "selected_angle_rel_error": selected["angle_rel_error"][0],
            "selected_distance_rel_error": selected["distance_rel_error"][0],
            "selected_model_counts": json.dumps(count_models(final_selected), sort_keys=True),
        }
    )
    return rows, winner_rows


def bucket_winner_rows(
    samples: dict[str, dict[str, dict[str, str]]],
    candidate_set: str,
    candidate_models: list[str],
    ids: list[str],
    min_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key_name in [
        "gt_angle_distance_bucket",
        "b4_pred_distance_bucket",
        "b4_pred_angle_bucket",
        "b4_pred_angle_distance_bucket",
        "b4_e_disagreement",
        "b4_e256_disagreement",
        "multi_model_disagreement",
    ]:
        buckets: dict[str, list[str]] = {}
        for sid in ids:
            key = proxy_key(samples, sid, key_name, candidate_models)
            buckets.setdefault(key, []).append(sid)
        for key, bucket_ids in sorted(buckets.items()):
            if len(bucket_ids) < min_count:
                continue
            best = best_final_model_for_ids(samples, candidate_models, bucket_ids)
            out: dict[str, object] = {
                "candidate_set": candidate_set,
                "bucket_type": key_name,
                "bucket": key,
                "sample_count": len(bucket_ids),
                "best_final_model": best,
            }
            for model in candidate_models:
                value, count = model_mean(samples, model, bucket_ids, "final_score")
                out[f"{model}_final"] = value
                out[f"{model}_valid_count"] = count
            rows.append(out)
    return rows


def make_verdict(
    oracle: list[dict[str, object]],
    selectors: list[dict[str, object]],
    manifest: list[dict[str, object]],
) -> dict[str, object]:
    by_model = {str(row["model_id"]): row for row in manifest}
    b4_final = parse_float(by_model.get("B4", {}).get("final_score"))
    e256_final = parse_float(by_model.get("E256", {}).get("final_score"))

    primary_oracle = next(
        (row for row in oracle if row["candidate_set"] == "selector_primary" and row["metric"] == "final_score_selector_components"),
        {},
    )
    primary_oracle_value = parse_float(primary_oracle.get("oracle_value"))
    primary_best_single = parse_float(primary_oracle.get("best_single_value"))

    proxy_rows = [
        row
        for row in selectors
        if row["candidate_set"] == "selector_primary" and row["protocol"] == "two_fold"
    ]
    best_proxy = min(
        proxy_rows,
        key=lambda row: parse_float(row.get("final_score")) if parse_float(row.get("final_score")) is not None else float("inf"),
        default={},
    )
    best_proxy_final = parse_float(best_proxy.get("final_score"))

    verdict = "analysis_completed_diagnostic_only"
    reason = "proxy selectors do not yet justify a method claim"
    authorized_next = "close_phase92_or_design_new_proxy_after_review"
    oracle_gain = None
    proxy_gain = None
    if primary_oracle_value is not None and primary_best_single is not None:
        oracle_gain = primary_oracle_value - primary_best_single
    if best_proxy_final is not None and primary_best_single is not None:
        proxy_gain = best_proxy_final - primary_best_single

    if oracle_gain is not None and oracle_gain <= -0.005:
        if proxy_gain is not None and proxy_gain <= -0.002:
            verdict = "selector_route_promising_but_needs_holdout"
            reason = "per-sample complementarity is large and proxy two-fold selector beats the best single model"
            authorized_next = "build_holdout_selector_or_minimal_router_smoke"
        else:
            verdict = "oracle_complementarity_without_deployable_proxy"
            reason = "oracle gain is large but two-fold proxy selection does not beat the best single model enough"
            authorized_next = "mine_better_inference_time_proxy_or_stop_selector_route"

    return {
        "phase_id": "phase92-minimum-sufficient-relation-feasibility",
        "stage": "g0_failure_bucket_selector_upper_bound",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verdict": verdict,
        "reason": reason,
        "authorized_next": authorized_next,
        "b4_final_score": b4_final,
        "e256_final_score": e256_final,
        "primary_oracle_final": primary_oracle_value,
        "primary_best_single_final": primary_best_single,
        "primary_oracle_gain_vs_best_single": oracle_gain,
        "best_two_fold_proxy": best_proxy.get("proxy_name", ""),
        "best_two_fold_proxy_final": best_proxy_final,
        "best_two_fold_proxy_gain_vs_best_single": proxy_gain,
    }


def write_summary(
    path: Path,
    verdict: dict[str, object],
    manifest: list[dict[str, object]],
    oracle: list[dict[str, object]],
    selectors: list[dict[str, object]],
) -> None:
    model_rows = [row for row in manifest if row.get("per_sample_exists")]
    primary_oracle_rows = [r for r in oracle if r["candidate_set"] == "selector_primary"]
    selector_primary_2fold = [r for r in selectors if r["candidate_set"] == "selector_primary" and r["protocol"] == "two_fold"]
    selector_primary_2fold = sorted(selector_primary_2fold, key=lambda r: parse_float(r.get("final_score")) or float("inf"))
    selector_primary_insample = [r for r in selectors if r["candidate_set"] == "selector_primary" and r["protocol"] == "insample"]
    selector_primary_insample = sorted(selector_primary_insample, key=lambda r: parse_float(r.get("final_score")) or float("inf"))

    lines = [
        "# Phase92-G0 Failure-Bucket / Selector Upper-Bound Analysis",
        "",
        f"- created_at: `{verdict.get('created_at')}`",
        f"- verdict: `{verdict.get('verdict')}`",
        f"- reason: {verdict.get('reason')}",
        f"- authorized_next: `{verdict.get('authorized_next')}`",
        "",
        "## Model Pool",
        "",
        "| model | family | role | final | angle | distance |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            "| "
            f"{row['model_id']} | {row['family']} | {row['role']} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} |"
        )

    lines.extend(
        [
            "",
            "## Selector-Primary Oracle",
            "",
            "`selector_primary = B3,B4,E,E256`.",
            "",
            "| metric | best_single | best_value | oracle | gain | selected_components |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in primary_oracle_rows:
        lines.append(
            "| "
            f"{row['metric']} | {row.get('best_single_model')} | "
            f"{fmt(parse_float(row.get('best_single_value')))} | "
            f"{fmt(parse_float(row.get('oracle_value')))} | "
            f"{fmt(parse_float(row.get('oracle_gain_vs_best_single')))} | "
            f"{row.get('selected_model_counts', '')} |"
        )

    lines.extend(
        [
            "",
            "## Best Proxy Selectors",
            "",
            "Two-fold rows train bucket-to-model choices on one deterministic split and evaluate on the other. They are still val811 estimates, not official-test evidence.",
            "",
            "| protocol | proxy | final | angle | distance | fallback_count | selected_counts |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in selector_primary_2fold[:5]:
        lines.append(
            "| "
            f"two_fold | {row['proxy_name']} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} | "
            f"{row.get('fallback_count')} | {row.get('selected_model_counts')} |"
        )
    for row in selector_primary_insample[:3]:
        lines.append(
            "| "
            f"insample | {row['proxy_name']} | "
            f"{fmt(parse_float(row.get('final_score')))} | "
            f"{fmt(parse_float(row.get('angle_rel_error')))} | "
            f"{fmt(parse_float(row.get('distance_rel_error')))} | "
            f"{row.get('fallback_count')} | {row.get('selected_model_counts')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The oracle rows estimate whether the existing models contain exploitable per-sample complementarity.",
            "- The proxy selector rows test whether simple inference-time signals can recover that complementarity without using ground truth.",
            "- Any positive proxy result here is still a selector design lead, not a method claim, because the same fixed val811 surface is used for calibration and evaluation.",
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
    parser.add_argument("--min-bucket-count", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase92_root: Path = args.phase92_root
    manifest, samples = load_models(args.phase91_root, phase92_root)

    oracle: list[dict[str, object]] = []
    winners: list[dict[str, object]] = []
    selectors: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []

    proxy_names = [
        "gt_angle_distance_bucket",
        "b4_pred_distance_bucket",
        "b4_pred_angle_bucket",
        "b4_pred_angle_distance_bucket",
        "b4_e_disagreement",
        "b4_e256_disagreement",
        "multi_model_disagreement",
    ]

    for set_name, raw_models in CANDIDATE_SETS.items():
        model_ids = available_models(samples, raw_models)
        ids = common_ids(samples, model_ids)
        if not model_ids or not ids:
            continue
        set_oracle, set_winners = oracle_rows(samples, set_name, model_ids, ids)
        oracle.extend(set_oracle)
        winners.extend(set_winners)
        if set_name in {"selector_primary", "selector_with_controls"}:
            bucket_rows.extend(bucket_winner_rows(samples, set_name, model_ids, ids, args.min_bucket_count))
            for proxy_name in proxy_names:
                if proxy_name == "b4_e_disagreement" and "E" not in model_ids:
                    continue
                if proxy_name == "b4_e256_disagreement" and "E256" not in model_ids:
                    continue
                for protocol in ["insample", "two_fold"]:
                    row = selector_eval(samples, model_ids, ids, proxy_name, protocol, args.min_train_count)
                    row["candidate_set"] = set_name
                    selectors.append(row)

    verdict = make_verdict(oracle, selectors, manifest)

    diag_dir = phase92_root / "diagnostics"
    report_dir = phase92_root / "reports"
    manifest_dir = phase92_root / "manifests"

    write_csv(
        diag_dir / "g0_model_manifest.csv",
        manifest,
        [
            "model_id",
            "family",
            "role",
            "eval_dir",
            "metrics_exists",
            "per_sample_exists",
            "sample_count",
            "final_score",
            "angle_rel_error",
            "distance_rel_error",
        ],
    )
    write_csv(
        diag_dir / "g0_oracle_upper_bounds.csv",
        oracle,
        [
            "candidate_set",
            "metric",
            "candidate_models",
            "common_sample_count",
            "best_single_model",
            "best_single_value",
            "best_single_valid_count",
            "oracle_value",
            "oracle_valid_count",
            "oracle_gain_vs_best_single",
            "selected_angle_rel_error",
            "selected_distance_rel_error",
            "selected_model_counts",
        ],
    )
    write_csv(
        diag_dir / "g0_oracle_winner_counts.csv",
        winners,
        ["candidate_set", "metric", "model_id", "winner_count", "winner_rate"],
    )
    selector_fields = [
        "candidate_set",
        "proxy_name",
        "protocol",
        "candidate_count",
        "candidate_models",
        "eval_count",
        "fallback_model",
        "fallback_count",
        "train_bucket_count",
        "selected_model_counts",
        "final_score",
        "final_valid_count",
        "angle_rel_error",
        "angle_valid_count",
        "distance_rel_error",
        "distance_valid_count",
    ]
    write_csv(diag_dir / "g0_proxy_selector_results.csv", selectors, selector_fields)

    bucket_fields = [
        "candidate_set",
        "bucket_type",
        "bucket",
        "sample_count",
        "best_final_model",
        "B3_final",
        "B3_valid_count",
        "B4_final",
        "B4_valid_count",
        "E_final",
        "E_valid_count",
        "E64_final",
        "E64_valid_count",
        "E256_final",
        "E256_valid_count",
    ]
    write_csv(diag_dir / "g0_bucket_winners.csv", bucket_rows, bucket_fields)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "phase92_g0_selector_upper_bound_verdict.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    write_summary(report_dir / "phase92_g0_selector_upper_bound_summary.md", verdict, manifest, oracle, selectors)
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
