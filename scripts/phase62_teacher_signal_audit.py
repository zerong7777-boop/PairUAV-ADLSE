#!/usr/bin/env python3
"""Audit existing train/val geometry-teacher signals for Phase62.

The script joins the Phase62 rank1 eval spine with an existing row-level
geometry/teacher table. It reports whether teacher outputs or per-pair
observability features explain rank1 hard-angle failures strongly enough to
justify teacher-guided training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_EVAL_CSV = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase62_angle_semantics_partial_unfreeze_v1/eval_spine/rank1_val811_enriched.csv"
)
DEFAULT_HARD_MANIFEST = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase62_angle_semantics_partial_unfreeze_v1/eval_spine/hard_angle_val811_manifest.jsonl"
)
DEFAULT_TEACHER_CSV = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/rich_native_diagnostic_probe/"
    "angle_only_geometry_official_feasibility_v1/f1_quality/geometry_policy_table_val811.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase62_angle_semantics_partial_unfreeze_v1/teacher_signal_audit"
)

TEACHER_ERROR_COLUMNS = [
    "teacher_angle_abs_error",
    "split_fusion_angle_abs_error",
    "candidate_angle_error",
]
BASE_ERROR_COLUMNS = [
    "rank1_angle_abs_error",
    "base_angle_error",
    "reloc3r_angle_abs_error",
]
FEATURE_CANDIDATES = [
    "abs_heading_delta_deg",
    "reliability_abs_heading_delta_deg",
    "angle_field_abs_mean",
    "angle_field_std",
    "angle_field_nonzero_ratio",
    "distance_field_abs_mean",
    "distance_field_std",
    "abs_range_delta",
    "reliability_abs_range_delta",
    "source_reliable",
    "source_conflict",
    "rank1_geometry_helpful",
    "teacher_aux_allowed",
    "angle_policy_weight",
]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def auc_score(values: List[float], labels: List[bool]) -> Optional[float]:
    positives = [v for v, y in zip(values, labels) if y]
    negatives = [v for v, y in zip(values, labels) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for p in positives:
        for n in negatives:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else None


def read_csv_by_pair(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "").strip()
        if pair_id:
            out[pair_id] = row
    return out


def read_hard_pair_ids(path: Path) -> set:
    pair_ids = set()
    if not path.exists():
        return pair_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id") or "").strip()
            if pair_id:
                pair_ids.add(pair_id)
    return pair_ids


def first_float(row: Dict[str, Any], names: List[str]) -> Optional[float]:
    for name in names:
        val = safe_float(row.get(name))
        if val is not None:
            return val
    return None


def join_rows(
    eval_rows: Dict[str, Dict[str, str]],
    teacher_rows: Dict[str, Dict[str, str]],
    hard_pair_ids: set,
) -> List[Dict[str, Any]]:
    joined: List[Dict[str, Any]] = []
    for pair_id, eval_row in sorted(eval_rows.items()):
        teacher_row = teacher_rows.get(pair_id)
        base_error = first_float(eval_row, BASE_ERROR_COLUMNS)
        if base_error is None:
            continue
        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "group_id": eval_row.get("group_id"),
            "is_hard": pair_id in hard_pair_ids,
            "base_angle_error": base_error,
            "base_distance_error": safe_float(eval_row.get("rank1_distance_abs_error")),
            "teacher_covered": teacher_row is not None,
        }
        if teacher_row is not None:
            teacher_error = first_float(teacher_row, TEACHER_ERROR_COLUMNS)
            row["teacher_angle_error"] = teacher_error
            row["teacher_help"] = (
                teacher_error is not None and teacher_error < base_error
            )
            row["teacher_gain"] = (
                base_error - teacher_error if teacher_error is not None else None
            )
            for name in FEATURE_CANDIDATES:
                if name in teacher_row:
                    bool_value = safe_bool(teacher_row.get(name))
                    float_value = safe_float(teacher_row.get(name))
                    if bool_value is not None:
                        row[name] = bool_value
                    elif float_value is not None:
                        row[name] = float_value
        joined.append(row)
    return joined


def summarize_subset(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    base_errors = [r["base_angle_error"] for r in rows if r.get("base_angle_error") is not None]
    covered = [r for r in rows if r.get("teacher_angle_error") is not None]
    teacher_errors = [r["teacher_angle_error"] for r in covered]
    help_rows = [r for r in covered if r.get("teacher_help")]
    oracle_errors = [
        min(float(r["base_angle_error"]), float(r["teacher_angle_error"]))
        for r in covered
        if r.get("teacher_angle_error") is not None
    ]
    base_on_covered = [float(r["base_angle_error"]) for r in covered]
    base_mae = mean(base_errors)
    base_covered_mae = mean(base_on_covered)
    oracle_mae = mean(oracle_errors)
    oracle_gain = (
        base_covered_mae - oracle_mae
        if base_covered_mae is not None and oracle_mae is not None
        else None
    )
    oracle_gain_frac = (
        oracle_gain / base_covered_mae
        if oracle_gain is not None and base_covered_mae not in (None, 0.0)
        else None
    )
    return {
        "name": name,
        "rows": len(rows),
        "base_angle_mae": base_mae,
        "teacher_coverage_rows": len(covered),
        "teacher_coverage_rate": len(covered) / len(rows) if rows else 0.0,
        "teacher_angle_mae_on_covered": mean(teacher_errors),
        "base_angle_mae_on_covered": base_covered_mae,
        "teacher_help_rows": len(help_rows),
        "teacher_help_rate_on_covered": len(help_rows) / len(covered) if covered else 0.0,
        "oracle_selective_angle_mae_on_covered": oracle_mae,
        "oracle_gain_vs_base_on_covered": oracle_gain,
        "oracle_gain_frac_vs_base_on_covered": oracle_gain_frac,
        "base_angle_p80": percentile(base_errors, 0.80),
        "base_angle_p90": percentile(base_errors, 0.90),
        "base_angle_p95": percentile(base_errors, 0.95),
    }


def feature_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hard_labels = [bool(r["is_hard"]) for r in rows]
    base_errors = [float(r["base_angle_error"]) for r in rows]
    out: List[Dict[str, Any]] = []
    for name in FEATURE_CANDIDATES:
        vals: List[float] = []
        ys: List[float] = []
        labels: List[bool] = []
        hard_vals: List[float] = []
        easy_vals: List[float] = []
        for row, label, base_error in zip(rows, hard_labels, base_errors):
            if name not in row:
                continue
            raw = row[name]
            if isinstance(raw, bool):
                value = 1.0 if raw else 0.0
            else:
                value = safe_float(raw)
                if value is None:
                    continue
            vals.append(value)
            ys.append(base_error)
            labels.append(label)
            if label:
                hard_vals.append(value)
            else:
                easy_vals.append(value)
        if not vals:
            continue
        auc = auc_score(vals, labels)
        out.append(
            {
                "feature": name,
                "coverage_rows": len(vals),
                "coverage_rate": len(vals) / len(rows) if rows else 0.0,
                "pearson_with_base_angle_error": pearson(vals, ys),
                "hard_auc_raw": auc,
                "hard_auc_best_direction": max(auc, 1.0 - auc) if auc is not None else None,
                "mean_hard": mean(hard_vals),
                "mean_easy": mean(easy_vals),
                "delta_hard_minus_easy": (
                    mean(hard_vals) - mean(easy_vals)
                    if mean(hard_vals) is not None and mean(easy_vals) is not None
                    else None
                ),
            }
        )
    out.sort(
        key=lambda r: (
            r["hard_auc_best_direction"] if r["hard_auc_best_direction"] is not None else -1.0,
            abs(r["pearson_with_base_angle_error"] or 0.0),
        ),
        reverse=True,
    )
    return out


def bool_group_summaries(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for name in FEATURE_CANDIDATES:
        if not any(isinstance(r.get(name), bool) for r in rows):
            continue
        true_rows = [r for r in rows if r.get(name) is True]
        false_rows = [r for r in rows if r.get(name) is False]
        summaries[name] = {
            "true": summarize_subset(true_rows, f"{name}=true"),
            "false": summarize_subset(false_rows, f"{name}=false"),
        }
    return summaries


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    all_rows = payload["summaries"]["all"]
    hard_rows = payload["summaries"]["hard"]
    decision = payload["decision"]
    lines = [
        "# Phase62 G2 Teacher Signal Audit",
        "",
        "This audit uses existing val811 row-level geometry-teacher artifacts only.",
        "It does not authorize test-time sidecars or test-set graph processing.",
        "",
        "## Coverage And Oracle Ceiling",
        "",
        f"- rows: {all_rows['rows']}",
        f"- teacher coverage: {all_rows['teacher_coverage_rows']} ({all_rows['teacher_coverage_rate']:.4f})",
        f"- base angle MAE: {all_rows['base_angle_mae']:.12f}",
        f"- teacher angle MAE on covered: {all_rows['teacher_angle_mae_on_covered']:.12f}",
        f"- teacher help rate on covered: {all_rows['teacher_help_rate_on_covered']:.4f}",
        f"- oracle selective angle MAE: {all_rows['oracle_selective_angle_mae_on_covered']:.12f}",
        f"- oracle gain fraction: {all_rows['oracle_gain_frac_vs_base_on_covered']:.4f}",
        "",
        "## Hard Top-20%",
        "",
        f"- hard rows: {hard_rows['rows']}",
        f"- hard base angle MAE: {hard_rows['base_angle_mae']:.12f}",
        f"- hard teacher angle MAE: {hard_rows['teacher_angle_mae_on_covered']:.12f}",
        f"- hard teacher help rate: {hard_rows['teacher_help_rate_on_covered']:.4f}",
        f"- hard oracle gain fraction: {hard_rows['oracle_gain_frac_vs_base_on_covered']:.4f}",
        "",
        "## Decision",
        "",
        f"- promote_teacher_guided_training: {decision['promote_teacher_guided_training']}",
        f"- reason: {decision['reason']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_decision(summary: Dict[str, Any], features: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_rows = summary["all"]
    hard_rows = summary["hard"]
    oracle_gain = all_rows.get("oracle_gain_frac_vs_base_on_covered") or 0.0
    hard_oracle_gain = hard_rows.get("oracle_gain_frac_vs_base_on_covered") or 0.0
    best_auc = max(
        [r.get("hard_auc_best_direction") or 0.0 for r in features],
        default=0.0,
    )
    coverage = all_rows.get("teacher_coverage_rate") or 0.0
    if coverage < 0.95:
        return {
            "promote_teacher_guided_training": False,
            "reason": "teacher coverage below 95%",
            "best_hard_auc": best_auc,
        }
    if oracle_gain < 0.08 and hard_oracle_gain < 0.08:
        return {
            "promote_teacher_guided_training": False,
            "reason": "teacher-corrected oracle gain is below the 8% Phase62 floor",
            "best_hard_auc": best_auc,
        }
    if best_auc < 0.65:
        return {
            "promote_teacher_guided_training": False,
            "reason": "observable features do not separate hard angle rows strongly enough",
            "best_hard_auc": best_auc,
        }
    return {
        "promote_teacher_guided_training": True,
        "reason": "teacher oracle and hard-row separation pass the Phase62 floor",
        "best_hard_auc": best_auc,
    }


def run_audit(
    eval_csv: Path,
    hard_manifest: Path,
    teacher_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    eval_rows = read_csv_by_pair(eval_csv)
    teacher_rows = read_csv_by_pair(teacher_csv)
    hard_ids = read_hard_pair_ids(hard_manifest)
    rows = join_rows(eval_rows, teacher_rows, hard_ids)
    hard_rows = [r for r in rows if r.get("is_hard")]
    easy_rows = [r for r in rows if not r.get("is_hard")]
    summaries = {
        "all": summarize_subset(rows, "all"),
        "hard": summarize_subset(hard_rows, "hard"),
        "easy": summarize_subset(easy_rows, "easy"),
    }
    features = feature_rows(rows)
    bool_summaries = bool_group_summaries(rows)
    payload = {
        "eval_csv": str(eval_csv),
        "hard_manifest": str(hard_manifest),
        "teacher_csv": str(teacher_csv),
        "summaries": summaries,
        "feature_audit": features,
        "bool_group_summaries": bool_summaries,
        "decision": make_decision(summaries, features),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "teacher_signal_audit.json"
    feature_csv_path = output_dir / "feature_hard_separation.csv"
    joined_csv_path = output_dir / "joined_teacher_signal_rows.csv"
    report_path = output_dir / "TEACHER_SIGNAL_AUDIT.md"
    write_json(json_path, payload)
    write_csv(feature_csv_path, features)
    write_csv(joined_csv_path, rows)
    write_report(report_path, payload)
    return {
        "json_path": str(json_path),
        "feature_csv_path": str(feature_csv_path),
        "joined_csv_path": str(joined_csv_path),
        "report_path": str(report_path),
        "payload": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--hard-manifest", type=Path, default=DEFAULT_HARD_MANIFEST)
    parser.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        eval_csv=args.eval_csv,
        hard_manifest=args.hard_manifest,
        teacher_csv=args.teacher_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
