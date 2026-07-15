import json
from pathlib import Path

import torch


DEFAULT_NOTE = "Local proxy metrics only. Hidden Codabench test scores require hidden truth.txt."


def _as_flat_float_tensor(value):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return tensor.reshape(-1)


def _wrap_angle_diff_deg(pred_deg, target_deg):
    diff = pred_deg - target_deg
    return torch.remainder(diff + 180.0, 360.0) - 180.0


def format_pairuav_prediction_rows(heading_prediction_deg, range_prediction):
    heading_prediction_deg = _as_flat_float_tensor(heading_prediction_deg)
    range_prediction = _as_flat_float_tensor(range_prediction)
    if heading_prediction_deg.numel() != range_prediction.numel():
        raise ValueError("Heading and range predictions must have the same number of samples")
    return [
        f"{heading_value:.6f} {range_value:.6f}"
        for heading_value, range_value in zip(heading_prediction_deg.tolist(), range_prediction.tolist())
    ]


def compute_pairuav_metrics_payload(
    heading_prediction_deg,
    heading_target_deg,
    range_prediction,
    range_target,
    distance_mode,
    range_min,
    range_max,
    note=DEFAULT_NOTE,
):
    heading_prediction_deg = _as_flat_float_tensor(heading_prediction_deg)
    heading_target_deg = _as_flat_float_tensor(heading_target_deg)
    range_prediction = _as_flat_float_tensor(range_prediction)
    range_target = _as_flat_float_tensor(range_target)

    sample_count = int(heading_prediction_deg.numel())
    if sample_count == 0:
        raise ValueError("Metrics payload requires at least one sample")

    heading_abs_error = _wrap_angle_diff_deg(heading_prediction_deg, heading_target_deg).abs()
    distance_abs_error = (range_prediction - range_target).abs()

    angle_mae_deg = float(heading_abs_error.mean().item())
    distance_mae = float(distance_abs_error.mean().item())
    angle_rel_error = angle_mae_deg / 180.0

    if distance_mode == "range_span":
        distance_denominator = float(range_max) - float(range_min)
    elif distance_mode == "endpoint_range_span":
        left_span = (range_target - float(range_min)).abs()
        right_span = (float(range_max) - range_target).abs()
        distance_denominator = float(torch.maximum(left_span, right_span).mean().item())
    else:
        raise ValueError(f"Unsupported distance_mode: {distance_mode}")

    if distance_denominator <= 0:
        raise ValueError("Distance denominator must be positive")

    distance_rel_error = distance_mae / distance_denominator
    final_score_proxy = (angle_rel_error + distance_rel_error) / 2.0

    return {
        "angle_mae_deg": angle_mae_deg,
        "angle_rel_error": angle_rel_error,
        "distance_mae": distance_mae,
        "distance_mode": distance_mode,
        "distance_rel_error": distance_rel_error,
        "final_score_proxy": final_score_proxy,
        "note": note,
        "samples": sample_count,
    }


def write_pairuav_devval_outputs(
    heading_prediction_deg,
    heading_target_deg,
    range_prediction,
    range_target,
    output_dir,
    range_min,
    range_max,
    note=DEFAULT_NOTE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_lines = format_pairuav_prediction_rows(heading_prediction_deg, range_prediction)
    predict_path = output_dir / "val_predict_output.txt"
    predict_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

    metrics_by_mode = {}
    for distance_mode in ("range_span", "endpoint_range_span"):
        payload = compute_pairuav_metrics_payload(
            heading_prediction_deg=heading_prediction_deg,
            heading_target_deg=heading_target_deg,
            range_prediction=range_prediction,
            range_target=range_target,
            distance_mode=distance_mode,
            range_min=range_min,
            range_max=range_max,
            note=note,
        )
        metrics_by_mode[distance_mode] = payload
        metrics_path = output_dir / f"val_metrics_{distance_mode}.json"
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return {
        "val_predict_output": str(predict_path),
        "metrics": metrics_by_mode,
    }
