#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


SOURCE_IDS = ["B3", "B4", "E", "E256"]


def parse_float(value: str) -> float:
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        raise ValueError(f"invalid float: {value}")
    return out


def bucket_distance(value: float) -> str:
    distance = abs(float(value))
    if distance == 0:
        return "d_00_zero"
    if distance <= 1:
        return "d_01_le_1"
    if distance <= 5:
        return "d_02_le_5"
    if distance <= 10:
        return "d_03_le_10"
    if distance <= 25:
        return "d_04_le_25"
    if distance <= 50:
        return "d_05_le_50"
    if distance <= 100:
        return "d_06_le_100"
    return "d_07_gt_100"


def read_result(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected two columns, got {len(parts)}")
            rows.append((parse_float(parts[0]), parse_float(parts[1])))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--b3-result", type=Path, required=True)
    parser.add_argument("--b4-result", type=Path, required=True)
    parser.add_argument("--e-result", type=Path, required=True)
    parser.add_argument("--e256-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    bucket_to_model = dict(mapping_payload["bucket_to_model"])
    fallback_model = str(mapping_payload["fallback_model"])

    source_paths = {
        "B3": args.b3_result,
        "B4": args.b4_result,
        "E": args.e_result,
        "E256": args.e256_result,
    }
    source_rows = {model: read_result(path) for model, path in source_paths.items()}
    counts = {model: len(rows) for model, rows in source_rows.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"source result line counts differ: {counts}")
    row_count = next(iter(counts.values()))

    selected_lines: list[str] = []
    trace_rows: list[dict[str, object]] = []
    selected_counts = {model: 0 for model in SOURCE_IDS}
    for idx in range(row_count):
        b4_heading, b4_distance = source_rows["B4"][idx]
        selector_key = bucket_distance(b4_distance)
        selected_model = bucket_to_model.get(selector_key, fallback_model)
        if selected_model not in source_rows:
            selected_model = fallback_model
        pred_heading, pred_distance = source_rows[selected_model][idx]
        selected_counts[selected_model] = selected_counts.get(selected_model, 0) + 1
        selected_lines.append(f"{pred_heading:.6f} {pred_distance:.6f}")
        trace_rows.append(
            {
                "row_index": idx,
                "selector_key": selector_key,
                "selected_model": selected_model,
                "selected_heading": f"{pred_heading:.9f}",
                "selected_distance": f"{pred_distance:.9f}",
                "b4_heading": f"{b4_heading:.9f}",
                "b4_distance": f"{b4_distance:.9f}",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.txt"
    result_path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / "selected_predictions.csv",
        trace_rows,
        [
            "row_index",
            "selector_key",
            "selected_model",
            "selected_heading",
            "selected_distance",
            "b4_heading",
            "b4_distance",
        ],
    )
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mapping": str(args.mapping),
        "source_results": {model: str(path) for model, path in source_paths.items()},
        "output_result": str(result_path),
        "row_count": row_count,
        "selected_counts": selected_counts,
        "uses_official_test_truth": False,
        "requires_same_row_order": True,
    }
    (args.output_dir / "apply_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
