#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


FIELDNAMES = [
    "sample_id",
    "src_image_path",
    "tgt_image_path",
    "json_path",
    "gt_heading",
    "gt_range",
    "teacher_heading_deg",
    "teacher_heading_cos",
    "teacher_heading_sin",
    "teacher_heading_error",
]


def _extract_int(value):
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group())
    return float("inf")


def _json_sort_key(json_path):
    path = Path(json_path)
    group_value = path.parent.name
    json_value = path.stem
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        group_value = data.get("group_id", group_value)
        json_value = data.get("json_id", json_value)
    except Exception:
        pass
    return (_extract_int(group_value), str(group_value), _extract_int(json_value), str(json_value))


def _iter_json_paths(root):
    return sorted(Path(root).rglob("*.json"), key=_json_sort_key)


def _read_predictions(path):
    rows = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no} expected two columns, got {len(parts)}")
        rows.append((float(parts[0]), float(parts[1])))
    return rows


def _coerce_float(data, *keys):
    for key in keys:
        if key in data:
            return float(data[key])
    raise KeyError(f"missing any of {keys}")


def _wrap_angle_diff_deg(pred_deg, target_deg):
    return ((float(pred_deg) - float(target_deg) + 180.0) % 360.0) - 180.0


def _image_value(data, *keys):
    for key in keys:
        if key in data:
            return str(data[key])
    return ""


def build_cache(json_root, prediction_path, output_csv):
    json_paths = _iter_json_paths(json_root)
    predictions = _read_predictions(prediction_path)
    if len(json_paths) != len(predictions):
        raise ValueError(
            f"prediction/json count mismatch: json={len(json_paths)} predictions={len(predictions)}"
        )

    rows = []
    for json_path, (teacher_heading_deg, _teacher_range) in zip(json_paths, predictions):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        group_id = str(data.get("group_id", Path(json_path).parent.name))
        json_id = str(data.get("json_id", Path(json_path).stem))
        sample_id = f"{group_id}/{json_id}"
        gt_heading = _coerce_float(data, "heading_deg", "heading_num")
        gt_range = _coerce_float(data, "range_value", "range_num")
        teacher_rad = math.radians(float(teacher_heading_deg))
        teacher_error = abs(_wrap_angle_diff_deg(teacher_heading_deg, gt_heading))
        rows.append(
            {
                "sample_id": sample_id,
                "src_image_path": _image_value(data, "image_a", "src_image_path", "query_image"),
                "tgt_image_path": _image_value(data, "image_b", "tgt_image_path", "target_image"),
                "json_path": str(json_path),
                "gt_heading": f"{gt_heading:.6f}",
                "gt_range": f"{gt_range:.6f}",
                "teacher_heading_deg": f"{float(teacher_heading_deg):.6f}",
                "teacher_heading_cos": f"{math.cos(teacher_rad):.9f}",
                "teacher_heading_sin": f"{math.sin(teacher_rad):.9f}",
                "teacher_heading_error": f"{teacher_error:.6f}",
            }
        )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_cache(args.json_root, args.prediction, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
