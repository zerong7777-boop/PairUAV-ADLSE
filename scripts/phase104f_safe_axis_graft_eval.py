#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


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


def _coerce_float(data, *keys):
    for key in keys:
        if key in data:
            return float(data[key])
    raise KeyError(f"missing any of {keys}")


def load_manifest(val_json_root):
    rows = []
    paths = sorted(Path(val_json_root).rglob("*.json"), key=_json_sort_key)
    if not paths:
        raise FileNotFoundError(f"No json files under {val_json_root}")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        group_id = str(data.get("group_id", path.parent.name))
        json_id = str(data.get("json_id", path.stem))
        rows.append(
            {
                "sample_id": f"{group_id}/{json_id}",
                "json_path": str(path),
                "gt_heading": _coerce_float(data, "heading_deg", "heading_num"),
                "gt_range": _coerce_float(data, "range_value", "range_num"),
            }
        )
    return rows


def load_prediction(path):
    heading = []
    range_value = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no} expected two columns, got {len(parts)}")
        heading.append(float(parts[0]))
        range_value.append(float(parts[1]))
    return heading, range_value


def validate_prediction_file(path, manifest):
    heading, range_value = load_prediction(path)
    if len(heading) != len(manifest):
        raise ValueError(
            f"prediction line count mismatch for {path}: lines={len(heading)} manifest={len(manifest)}"
        )
    return heading, range_value


def wrap_angle_diff_deg(pred_deg, target_deg):
    return ((pred_deg - target_deg + 180.0) % 360.0) - 180.0


def compute_metrics(heading_pred, range_pred, manifest, range_min=-132.0, range_max=132.0):
    if not (len(heading_pred) == len(range_pred) == len(manifest)):
        raise ValueError("sample count mismatch while computing graft metrics")
    heading_abs_error = [
        abs(wrap_angle_diff_deg(pred, row["gt_heading"]))
        for pred, row in zip(heading_pred, manifest)
    ]
    distance_abs_error = [
        abs(pred - row["gt_range"])
        for pred, row in zip(range_pred, manifest)
    ]
    angle_mae_deg = sum(heading_abs_error) / len(manifest)
    distance_mae = sum(distance_abs_error) / len(manifest)
    angle_rel_error = angle_mae_deg / 180.0
    distance_rel_error = distance_mae / (float(range_max) - float(range_min))
    return {
        "angle_mae_deg": angle_mae_deg,
        "angle_rel_error": angle_rel_error,
        "distance_mae": distance_mae,
        "distance_mode": "range_span",
        "distance_rel_error": distance_rel_error,
        "final_score_proxy": (angle_rel_error + distance_rel_error) / 2.0,
        "samples": len(manifest),
    }


def write_predictions(path, heading_pred, range_pred):
    lines = [
        f"{heading_value:.6f} {range_value:.6f}"
        for heading_value, range_value in zip(heading_pred, range_pred)
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grafts(val_json_root, output_root, combos, range_min=-132.0, range_max=132.0):
    manifest = load_manifest(val_json_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows = []
    for label, heading_path, range_path in combos:
        heading_pred, _ = validate_prediction_file(heading_path, manifest)
        _, range_pred = validate_prediction_file(range_path, manifest)
        combo_dir = output_root / label
        combo_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(combo_dir / "val_predict_output.txt", heading_pred, range_pred)
        payload = compute_metrics(heading_pred, range_pred, manifest, range_min=range_min, range_max=range_max)
        payload.update(
            {
                "note": "Safe axis graft metrics. Inputs validated against one canonical val JSON manifest.",
                "manifest_source": str(val_json_root),
                "heading_prediction_source": str(heading_path),
                "range_prediction_source": str(range_path),
            }
        )
        (combo_dir / "val_metrics_range_span.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append({"label": label, **payload})

    summary_json = output_root / "axis_merge_summary.json"
    summary_csv = output_root / "axis_merge_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-json-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--range-min", type=float, default=-132.0)
    parser.add_argument("--range-max", type=float, default=132.0)
    parser.add_argument(
        "--combo",
        action="append",
        nargs=3,
        metavar=("LABEL", "HEADING_PRED", "RANGE_PRED"),
        required=True,
    )
    args = parser.parse_args()
    rows = run_grafts(
        val_json_root=args.val_json_root,
        output_root=args.output_root,
        combos=args.combo,
        range_min=args.range_min,
        range_max=args.range_max,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
