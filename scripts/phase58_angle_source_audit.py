#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANGLE_THRESHOLDS = (
    ("0p5", 0.5),
    ("1p0", 1.0),
    ("2p0", 2.0),
    ("5p0", 5.0),
)


@dataclass(frozen=True)
class Prediction:
    pair_id: str
    heading: float
    target_heading: float | None = None
    distance: float | None = None
    target_distance: float | None = None


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    split: str | None = None
    split_col: str = "split"
    pair_col: str = "pair_id"
    heading_col: str | None = None
    target_col: str | None = None
    distance_col: str | None = None
    target_distance_col: str | None = None


def wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_abs_error_deg(pred: float, target: float) -> float:
    return abs(wrap_angle_deg(float(pred) - float(target)))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def choose_column(fieldnames: list[str], explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"column '{explicit}' not found; available={fieldnames}")
        return explicit
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def read_predictions_csv(
    path: Path,
    *,
    pair_col: str = "pair_id",
    heading_col: str | None = None,
    target_col: str | None = None,
    distance_col: str | None = None,
    target_distance_col: str | None = None,
    split_col: str = "split",
    split: str | None = None,
    heading_candidates: list[str] | None = None,
    target_candidates: list[str] | None = None,
    distance_candidates: list[str] | None = None,
    target_distance_candidates: list[str] | None = None,
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if pair_col not in fieldnames:
            raise ValueError(f"pair column '{pair_col}' not found; available={fieldnames}")
        selected_heading = choose_column(
            fieldnames,
            heading_col,
            heading_candidates
            or ["rank1_heading", "pred_heading", "pred_heading_deg", "same_forward_avg_heading", "heading"],
        )
        if selected_heading is None:
            raise ValueError(f"no heading column found in {path}; available={fieldnames}")
        selected_target = choose_column(
            fieldnames,
            target_col,
            target_candidates or ["target_heading", "target_heading_deg"],
        )
        selected_distance = choose_column(
            fieldnames,
            distance_col,
            distance_candidates or ["rank1_distance", "pred_distance", "pred_distance_m", "distance"],
        )
        selected_target_distance = choose_column(
            fieldnames,
            target_distance_col,
            target_distance_candidates or ["target_distance", "target_distance_m"],
        )

        predictions: dict[str, Prediction] = {}
        skipped_rows = 0
        duplicate_rows = 0
        total_rows = 0
        split_filtered_rows = 0
        for row in reader:
            total_rows += 1
            if split is not None and split_col in fieldnames and str(row.get(split_col, "")) != split:
                split_filtered_rows += 1
                continue
            pair_id = str(row.get(pair_col, "")).strip()
            heading = finite_float(row.get(selected_heading))
            if not pair_id or heading is None:
                skipped_rows += 1
                continue
            target_heading = finite_float(row.get(selected_target)) if selected_target else None
            distance = finite_float(row.get(selected_distance)) if selected_distance else None
            target_distance = finite_float(row.get(selected_target_distance)) if selected_target_distance else None
            if pair_id in predictions:
                duplicate_rows += 1
            predictions[pair_id] = Prediction(
                pair_id=pair_id,
                heading=heading,
                target_heading=target_heading,
                distance=distance,
                target_distance=target_distance,
            )

    meta = {
        "path": str(path),
        "rows_total": total_rows,
        "rows_loaded": len(predictions),
        "rows_skipped": skipped_rows,
        "rows_split_filtered": split_filtered_rows,
        "duplicate_rows": duplicate_rows,
        "pair_col": pair_col,
        "heading_col": selected_heading,
        "target_col": selected_target,
        "distance_col": selected_distance,
        "target_distance_col": selected_target_distance,
        "split": split,
        "split_col": split_col,
    }
    return predictions, meta


def read_rank1_csv(path: Path) -> dict[str, Prediction]:
    predictions, _ = read_predictions_csv(
        path,
        heading_candidates=["rank1_heading", "pred_heading", "pred_heading_deg"],
        distance_candidates=["rank1_distance", "pred_distance", "distance"],
    )
    return predictions


def read_source_csv(spec: SourceSpec) -> dict[str, Prediction]:
    predictions, _ = read_predictions_csv(
        spec.path,
        pair_col=spec.pair_col,
        heading_col=spec.heading_col,
        target_col=spec.target_col,
        distance_col=spec.distance_col,
        target_distance_col=spec.target_distance_col,
        split_col=spec.split_col,
        split=spec.split,
    )
    return predictions


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _tail_count(errors: list[float], threshold: float) -> int:
    return sum(error >= threshold for error in errors)


def evaluate_source(
    source_name: str,
    rank1: dict[str, Prediction],
    source: dict[str, Prediction],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_ids = sorted(set(rank1).intersection(source))
    rank1_errors: list[float] = []
    source_errors: list[float] = []
    oracle_errors: list[float] = []
    target_deltas: list[float] = []
    rank1_distance_errors: list[float] = []
    source_distance_errors: list[float] = []
    pair_rows: list[dict[str, Any]] = []

    for pair_id in pair_ids:
        r = rank1[pair_id]
        s = source[pair_id]
        target = r.target_heading if r.target_heading is not None else s.target_heading
        if target is None:
            continue
        rank1_error = angle_abs_error_deg(r.heading, target)
        source_error = angle_abs_error_deg(s.heading, target)
        rank1_errors.append(rank1_error)
        source_errors.append(source_error)
        oracle_errors.append(min(rank1_error, source_error))
        if r.target_heading is not None and s.target_heading is not None:
            target_deltas.append(angle_abs_error_deg(s.target_heading, r.target_heading))
        if r.distance is not None and r.target_distance is not None:
            rank1_distance_errors.append(abs(r.distance - r.target_distance))
        if s.distance is not None and r.target_distance is not None:
            source_distance_errors.append(abs(s.distance - r.target_distance))
        pair_rows.append(
            {
                "source": source_name,
                "pair_id": pair_id,
                "target_heading": _round(target, 9),
                "rank1_heading": _round(r.heading, 9),
                "source_heading": _round(s.heading, 9),
                "rank1_angle_abs_error": _round(rank1_error, 9),
                "source_angle_abs_error": _round(source_error, 9),
                "oracle_angle_abs_error": _round(min(rank1_error, source_error), 9),
                "source_better": int(source_error < rank1_error),
            }
        )

    overlap_rows = len(rank1_errors)
    source_better_count = sum(s < r for s, r in zip(source_errors, rank1_errors))
    source_equal_count = sum(abs(s - r) <= 1e-9 for s, r in zip(source_errors, rank1_errors))
    rank1_mae = _mean(rank1_errors)
    source_mae = _mean(source_errors)
    oracle_mae = _mean(oracle_errors)
    oracle_gain_pct = None
    source_delta_pct = None
    if rank1_mae and oracle_mae is not None:
        oracle_gain_pct = 100.0 * (rank1_mae - oracle_mae) / rank1_mae
    if rank1_mae and source_mae is not None:
        source_delta_pct = 100.0 * (source_mae - rank1_mae) / rank1_mae

    metrics: dict[str, Any] = {
        "source": source_name,
        "rank1_rows": len(rank1),
        "source_rows": len(source),
        "overlap_rows": overlap_rows,
        "rank1_angle_mae": _round(rank1_mae),
        "source_angle_mae": _round(source_mae),
        "source_minus_rank1_angle_mae": _round((source_mae - rank1_mae) if source_mae is not None and rank1_mae is not None else None),
        "source_delta_pct": _round(source_delta_pct),
        "oracle_min_angle_mae": _round(oracle_mae),
        "oracle_gain_pct": _round(oracle_gain_pct),
        "source_better_count": source_better_count,
        "source_equal_count": source_equal_count,
        "source_win_rate": _round(source_better_count / overlap_rows if overlap_rows else None),
        "mean_abs_target_delta": _round(_mean(target_deltas)),
        "rank1_distance_mae": _round(_mean(rank1_distance_errors)),
        "source_distance_mae": _round(_mean(source_distance_errors)),
    }
    for label, threshold in ANGLE_THRESHOLDS:
        metrics[f"rank1_angle_ge_{label}"] = _tail_count(rank1_errors, threshold)
        metrics[f"source_angle_ge_{label}"] = _tail_count(source_errors, threshold)
        metrics[f"oracle_angle_ge_{label}"] = _tail_count(oracle_errors, threshold)
    return metrics, pair_rows


def parse_source_arg(value: str) -> SourceSpec:
    items: dict[str, str] = {}
    for part in value.split(","):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"source part must be key=value: {part!r}")
        key, raw = part.split("=", 1)
        items[key.strip()] = raw.strip()
    missing = [key for key in ("name", "path") if key not in items]
    if missing:
        raise ValueError(f"source spec missing keys: {missing}; got {value!r}")
    return SourceSpec(
        name=items["name"],
        path=Path(items["path"]),
        split=items.get("split"),
        split_col=items.get("split_col", "split"),
        pair_col=items.get("pair_col", "pair_id"),
        heading_col=items.get("heading"),
        target_col=items.get("target"),
        distance_col=items.get("distance"),
        target_distance_col=items.get("target_distance"),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(metrics_rows: list[dict[str, Any]], source_meta: dict[str, Any], rank1_meta: dict[str, Any]) -> str:
    lines = [
        "# Phase58 Angle Source Audit",
        "",
        "This audit compares existing angle-source prediction files against the same rank1 val set on overlapping pair ids.",
        "Distance is reported only for context; the intended splice path keeps rank1 distance unless explicitly stated.",
        "",
        "## Inputs",
        "",
        f"- rank1_csv: `{rank1_meta['path']}`",
        f"- rank1_rows_loaded: `{rank1_meta['rows_loaded']}`",
        "",
        "## Metrics",
        "",
        "| source | overlap | rank1_angle_mae | source_angle_mae | source_delta_pct | win_rate | oracle_min_angle_mae | oracle_gain_pct | source_tail>=1 | oracle_tail>=1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            "| {source} | {overlap_rows} | {rank1_angle_mae} | {source_angle_mae} | {source_delta_pct} | {source_win_rate} | {oracle_min_angle_mae} | {oracle_gain_pct} | {source_angle_ge_1p0} | {oracle_angle_ge_1p0} |".format(
                **row
            )
        )
    lines.extend(["", "## Source Loading"])
    for name, meta in source_meta.items():
        lines.extend(
            [
                "",
                f"### {name}",
                f"- path: `{meta['path']}`",
                f"- rows_total: `{meta['rows_total']}`",
                f"- rows_loaded: `{meta['rows_loaded']}`",
                f"- rows_skipped: `{meta['rows_skipped']}`",
                f"- rows_split_filtered: `{meta['rows_split_filtered']}`",
                f"- duplicate_rows: `{meta['duplicate_rows']}`",
                f"- heading_col: `{meta['heading_col']}`",
                f"- target_col: `{meta['target_col']}`",
            ]
        )
    lines.extend(["", "## Decision Notes", ""])
    if not metrics_rows:
        lines.append("- No source rows were evaluated.")
    else:
        small_direct = [
            row
            for row in metrics_rows
            if row.get("source_angle_mae") is not None
            and row.get("rank1_angle_mae") is not None
            and row["source_angle_mae"] < row["rank1_angle_mae"]
        ]
        material_direct = [
            row for row in small_direct if row.get("source_delta_pct") is not None and row["source_delta_pct"] <= -5.0
        ]
        oracle_candidates = [
            row for row in metrics_rows if row.get("oracle_gain_pct") is not None and row["oracle_gain_pct"] >= 10.0
        ]
        if material_direct:
            lines.append("- At least one source has >=5% direct angle MAE gain over rank1; it is a candidate for splice or teacher construction.")
        elif small_direct:
            names = ", ".join(row["source"] for row in small_direct)
            lines.append(f"- Some sources beat rank1 only marginally ({names}); No audited source has >=5% direct angle MAE gain.")
        else:
            lines.append("- No audited source beats rank1 directly on its overlap; No audited source has >=5% direct angle MAE gain.")
        if oracle_candidates:
            lines.append("- At least one source has >=10% oracle gain, so a selector/policy route may be worth testing; this is an upper bound, not deployable evidence.")
        else:
            lines.append("- Oracle gains are below the 10% decision threshold or unavailable; selector training is not currently justified.")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank1, rank1_meta = read_predictions_csv(
        Path(args.rank1_csv),
        heading_candidates=["rank1_heading", "pred_heading", "pred_heading_deg"],
        distance_candidates=["rank1_distance", "pred_distance", "distance"],
    )
    out_dir = Path(args.out_dir)
    metrics_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    source_meta: dict[str, Any] = {}

    for source_arg in args.source:
        spec = parse_source_arg(source_arg)
        source, meta = read_predictions_csv(
            spec.path,
            pair_col=spec.pair_col,
            heading_col=spec.heading_col,
            target_col=spec.target_col,
            distance_col=spec.distance_col,
            target_distance_col=spec.target_distance_col,
            split_col=spec.split_col,
            split=spec.split,
        )
        metrics, pair_rows = evaluate_source(spec.name, rank1, source)
        metrics_rows.append(metrics)
        all_pair_rows.extend(pair_rows)
        source_meta[spec.name] = meta

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "source_metrics.csv", metrics_rows)
    if args.write_pair_errors:
        write_csv(out_dir / "pair_errors.csv", all_pair_rows)
    payload = {
        "rank1_meta": rank1_meta,
        "source_meta": source_meta,
        "metrics": metrics_rows,
    }
    (out_dir / "source_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(metrics_rows, source_meta, rank1_meta), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank1-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Comma separated key=value spec: name=foo,path=/x.csv,split=val,heading=pred_heading_deg,target=target_heading_deg",
    )
    parser.add_argument("--write-pair-errors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.source:
        parser.error("at least one --source is required")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
