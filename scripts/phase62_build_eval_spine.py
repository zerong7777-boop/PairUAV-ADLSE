#!/usr/bin/env python3
"""Build the Phase62 rank1 validation spine and hard-angle subset.

This script is intentionally read-only with respect to model outputs. It
normalizes an existing PairUAV prediction CSV into stable metrics and a
hard-angle manifest that later Phase62 experiments can reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PREDICTION_CSV = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase45_lab_parallel_eval_prep_v1_20260523_2215/"
    "original_rank1_val811/rank1_predictions.csv"
)
DEFAULT_VAL_JSON_ROOT = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase56_reloc3r_geometry_consistent_angle_training_v1/"
    "surfaces/phase54_8192_fixed_val811/val_json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase62_angle_semantics_partial_unfreeze_v1/eval_spine"
)


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


def wrapped_signed_angle_error_deg(pred: float, target: float) -> float:
    return ((pred - target + 180.0) % 360.0) - 180.0


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def read_prediction_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_fixed_json_path(row: Dict[str, Any], val_json_root: Path) -> Optional[str]:
    pair_id = str(row.get("pair_id") or "").strip()
    group_id = str(row.get("group_id") or "").strip()
    if pair_id:
        parts = pair_id.replace("\\", "/").split("/")
        if len(parts) >= 2:
            candidate = val_json_root / parts[-2] / f"{parts[-1]}.json"
            return str(candidate)
        if group_id:
            return str(val_json_root / group_id / f"{pair_id}.json")
    json_path = str(row.get("json_path") or "").strip()
    return json_path or None


def enrich_row(row: Dict[str, str], val_json_root: Path) -> Optional[Dict[str, Any]]:
    target_heading = safe_float(row.get("target_heading"))
    target_distance = safe_float(row.get("target_distance"))
    rank1_heading = safe_float(row.get("rank1_heading"))
    rank1_distance = safe_float(row.get("rank1_distance"))
    angle_error = safe_float(row.get("rank1_angle_abs_error"))
    distance_error = safe_float(row.get("rank1_distance_abs_error"))

    if angle_error is None and target_heading is not None and rank1_heading is not None:
        angle_error = abs(wrapped_signed_angle_error_deg(rank1_heading, target_heading))
    if distance_error is None and target_distance is not None and rank1_distance is not None:
        distance_error = abs(rank1_distance - target_distance)

    if angle_error is None or distance_error is None:
        return None

    out: Dict[str, Any] = dict(row)
    out["target_heading"] = target_heading
    out["target_distance"] = target_distance
    out["rank1_heading"] = rank1_heading
    out["rank1_distance"] = rank1_distance
    out["rank1_angle_abs_error"] = float(angle_error)
    out["rank1_distance_abs_error"] = float(distance_error)
    out["fixed_json_path"] = infer_fixed_json_path(out, val_json_root)
    return out


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_metrics(rows: List[Dict[str, Any]], hard_quantile: float) -> Dict[str, Any]:
    angle_errors = [float(r["rank1_angle_abs_error"]) for r in rows]
    distance_errors = [float(r["rank1_distance_abs_error"]) for r in rows]
    angle_mae = mean(angle_errors)
    distance_mae = mean(distance_errors)
    hard_threshold = percentile(angle_errors, hard_quantile)
    hard_rows = [r for r in rows if hard_threshold is not None and r["rank1_angle_abs_error"] >= hard_threshold]

    return {
        "num_rows": len(rows),
        "hard_quantile": hard_quantile,
        "angle_mae": angle_mae,
        "distance_mae": distance_mae,
        "angle_p50": percentile(angle_errors, 0.50),
        "angle_p80": percentile(angle_errors, 0.80),
        "angle_p90": percentile(angle_errors, 0.90),
        "angle_p95": percentile(angle_errors, 0.95),
        "angle_max": max(angle_errors) if angle_errors else None,
        "distance_p50": percentile(distance_errors, 0.50),
        "distance_p80": percentile(distance_errors, 0.80),
        "distance_p90": percentile(distance_errors, 0.90),
        "distance_p95": percentile(distance_errors, 0.95),
        "distance_max": max(distance_errors) if distance_errors else None,
        "angle_ge_0p5": sum(1 for v in angle_errors if v >= 0.5),
        "angle_ge_1p0": sum(1 for v in angle_errors if v >= 1.0),
        "angle_ge_2p0": sum(1 for v in angle_errors if v >= 2.0),
        "hard_threshold": hard_threshold,
        "hard_rows": len(hard_rows),
        "hard_angle_mae": mean([float(r["rank1_angle_abs_error"]) for r in hard_rows]),
        "hard_distance_mae": mean([float(r["rank1_distance_abs_error"]) for r in hard_rows]),
        "gate_angle_mae_cheap": 0.1297,
        "gate_angle_mae_serious": 0.1253,
        "gate_distance_mae_protect_1p02x": distance_mae * 1.02 if distance_mae is not None else None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_enriched_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "split",
        "group_id",
        "json_path",
        "fixed_json_path",
        "target_heading",
        "target_distance",
        "rank1_heading",
        "rank1_distance",
        "rank1_angle_abs_error",
        "rank1_distance_abs_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, metrics: Dict[str, Any], hard_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase62 G0 Eval Spine",
        "",
        "This report is generated from the fixed rank1 validation prediction CSV.",
        "It is a training-time audit artifact only and does not use official test-set structure.",
        "",
        "## Rank1 Val811 Metrics",
        "",
        f"- rows: {metrics['num_rows']}",
        f"- angle_mae: {metrics['angle_mae']:.12f}",
        f"- distance_mae: {metrics['distance_mae']:.12f}",
        f"- angle_p80: {metrics['angle_p80']:.12f}",
        f"- angle_p90: {metrics['angle_p90']:.12f}",
        f"- angle_p95: {metrics['angle_p95']:.12f}",
        f"- hard_quantile: {metrics['hard_quantile']}",
        f"- hard_threshold: {metrics['hard_threshold']:.12f}",
        f"- hard_rows: {metrics['hard_rows']}",
        f"- hard_manifest: {hard_path}",
        "",
        "## Phase62 Gates",
        "",
        f"- cheap angle gate: <= {metrics['gate_angle_mae_cheap']}",
        f"- serious angle gate: <= {metrics['gate_angle_mae_serious']}",
        f"- distance protect gate: <= {metrics['gate_distance_mae_protect_1p02x']:.12f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_eval_spine(
    prediction_csv: Path,
    val_json_root: Path,
    output_dir: Path,
    hard_quantile: float,
) -> Dict[str, Any]:
    raw_rows = read_prediction_csv(prediction_csv)
    rows = [r for r in (enrich_row(row, val_json_root) for row in raw_rows) if r is not None]
    rows.sort(key=lambda r: str(r.get("pair_id", "")))

    metrics = build_metrics(rows, hard_quantile)
    hard_threshold = metrics["hard_threshold"]
    hard_rows = [
        r for r in rows if hard_threshold is not None and r["rank1_angle_abs_error"] >= hard_threshold
    ]
    hard_rows.sort(key=lambda r: (-float(r["rank1_angle_abs_error"]), str(r.get("pair_id", ""))))
    hard_manifest = []
    for idx, row in enumerate(hard_rows, start=1):
        hard_manifest.append(
            {
                "hard_rank": idx,
                "pair_id": row.get("pair_id"),
                "group_id": row.get("group_id"),
                "json_path": row.get("fixed_json_path") or row.get("json_path"),
                "target_heading": row.get("target_heading"),
                "target_distance": row.get("target_distance"),
                "rank1_heading": row.get("rank1_heading"),
                "rank1_distance": row.get("rank1_distance"),
                "rank1_angle_abs_error": row.get("rank1_angle_abs_error"),
                "rank1_distance_abs_error": row.get("rank1_distance_abs_error"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rank1_val811_metrics.json"
    hard_summary_path = output_dir / "hard_angle_summary.json"
    hard_manifest_path = output_dir / "hard_angle_val811_manifest.jsonl"
    enriched_csv_path = output_dir / "rank1_val811_enriched.csv"
    report_path = output_dir / "EVAL_SPINE_REPORT.md"

    write_json(
        metrics_path,
        {
            "prediction_csv": str(prediction_csv),
            "val_json_root": str(val_json_root),
            "metrics": metrics,
        },
    )
    write_json(
        hard_summary_path,
        {
            "hard_quantile": hard_quantile,
            "hard_threshold": hard_threshold,
            "hard_rows": len(hard_manifest),
            "hard_angle_mae": metrics["hard_angle_mae"],
            "hard_distance_mae": metrics["hard_distance_mae"],
        },
    )
    write_jsonl(hard_manifest_path, hard_manifest)
    write_enriched_csv(enriched_csv_path, rows)
    write_markdown_report(report_path, metrics, hard_manifest_path)

    return {
        "metrics_path": str(metrics_path),
        "hard_summary_path": str(hard_summary_path),
        "hard_manifest_path": str(hard_manifest_path),
        "enriched_csv_path": str(enriched_csv_path),
        "report_path": str(report_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("--val-json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hard-quantile", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_eval_spine(
        prediction_csv=args.prediction_csv,
        val_json_root=args.val_json_root,
        output_dir=args.output_dir,
        hard_quantile=args.hard_quantile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
