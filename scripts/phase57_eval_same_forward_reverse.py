#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_pairuav_manifest_predictions import (  # noqa: E402
    DEFAULT_MODEL,
    as_flat_float_list,
    as_string_list,
    build_loader,
    derive_pair_ids,
    load_checkpoint,
    materialize_json_subset,
    read_manifest,
    sha256_file,
)
from reloc3r.reloc3r_relpose import Reloc3rRelpose  # noqa: E402,F401


def wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_abs_error_deg(pred: float, target: float) -> float:
    return abs(wrap_angle_deg(float(pred) - float(target)))


def circular_mean_deg(a: float, b: float) -> float:
    ar = math.radians(float(a))
    br = math.radians(float(b))
    return math.degrees(math.atan2(math.sin(ar) + math.sin(br), math.cos(ar) + math.cos(br)))


def heading_deg_from_vec(heading_vec: Any) -> torch.Tensor:
    tensor = torch.as_tensor(heading_vec, dtype=torch.float32)
    return torch.rad2deg(torch.atan2(tensor[:, 1], tensor[:, 0]))


def select_reverse_prediction_tensors(
    reverse_source: str,
    pred1: dict[str, Any],
    *,
    pred2_swapped: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    if reverse_source == "same_forward_pose1":
        return pred1["heading_vec"], pred1["range_value"]
    if reverse_source == "true_swap_pred2_inverse":
        if pred2_swapped is None:
            raise ValueError("pred2_swapped is required for true_swap_pred2_inverse")
        return pred2_swapped["heading_vec"], pred2_swapped["range_value"]
    raise ValueError(f"unsupported reverse source: {reverse_source}")


def write_same_forward_reverse_csv(
    path: Path,
    *,
    pair_ids: list[str],
    group_ids: list[str],
    json_paths: list[str],
    pred_heading: Any,
    pred_distance: Any,
    reverse_heading: Any,
    reverse_distance: Any,
    target_heading: Any,
    target_distance: Any,
) -> dict[str, Any]:
    pred_heading_values = as_flat_float_list(pred_heading)
    pred_distance_values = as_flat_float_list(pred_distance)
    reverse_heading_values = as_flat_float_list(reverse_heading)
    reverse_distance_values = as_flat_float_list(reverse_distance)
    target_heading_values = as_flat_float_list(target_heading)
    target_distance_values = as_flat_float_list(target_distance)
    count = len(pair_ids)
    lengths = {
        "pair_ids": len(pair_ids),
        "group_ids": len(group_ids),
        "json_paths": len(json_paths),
        "pred_heading": len(pred_heading_values),
        "pred_distance": len(pred_distance_values),
        "reverse_heading": len(reverse_heading_values),
        "reverse_distance": len(reverse_distance_values),
        "target_heading": len(target_heading_values),
        "target_distance": len(target_distance_values),
    }
    if any(value != count for value in lengths.values()):
        raise ValueError(f"prediction output length mismatch: {lengths}")

    path.parent.mkdir(parents=True, exist_ok=True)
    rank1_angle_errors: list[float] = []
    avg_angle_errors: list[float] = []
    distance_errors: list[float] = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "pair_id",
            "split",
            "group_id",
            "json_path",
            "target_heading",
            "target_distance",
            "rank1_heading",
            "rank1_distance",
            "reverse_heading",
            "reverse_distance",
            "reverse_forward_heading",
            "reverse_forward_distance",
            "same_forward_heading_disagreement",
            "same_forward_distance_disagreement",
            "same_forward_avg_heading",
            "same_forward_avg_distance",
            "rank1_angle_abs_error",
            "same_forward_avg_angle_abs_error",
            "rank1_distance_abs_error",
            "same_forward_avg_distance_abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, pair_id in enumerate(pair_ids):
            rank1_heading = pred_heading_values[idx]
            rank1_distance = pred_distance_values[idx]
            inv_heading = reverse_heading_values[idx]
            inv_distance = reverse_distance_values[idx]
            reverse_forward_heading = wrap_angle_deg(-inv_heading)
            reverse_forward_distance = -inv_distance
            avg_heading = circular_mean_deg(rank1_heading, reverse_forward_heading)
            # Distance averaging is diagnostic only; candidate corrections keep rank1 distance.
            avg_distance = 0.5 * (rank1_distance + reverse_forward_distance)
            target_h = target_heading_values[idx]
            target_d = target_distance_values[idx]
            rank1_angle_error = angle_abs_error_deg(rank1_heading, target_h)
            avg_angle_error = angle_abs_error_deg(avg_heading, target_h)
            rank1_dist_error = abs(rank1_distance - target_d)
            avg_dist_error = abs(avg_distance - target_d)
            rank1_angle_errors.append(rank1_angle_error)
            avg_angle_errors.append(avg_angle_error)
            distance_errors.append(rank1_dist_error)
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "split": "train",
                    "group_id": group_ids[idx],
                    "json_path": json_paths[idx],
                    "target_heading": f"{target_h:.9f}",
                    "target_distance": f"{target_d:.9f}",
                    "rank1_heading": f"{rank1_heading:.9f}",
                    "rank1_distance": f"{rank1_distance:.9f}",
                    "reverse_heading": f"{inv_heading:.9f}",
                    "reverse_distance": f"{inv_distance:.9f}",
                    "reverse_forward_heading": f"{reverse_forward_heading:.9f}",
                    "reverse_forward_distance": f"{reverse_forward_distance:.9f}",
                    "same_forward_heading_disagreement": f"{angle_abs_error_deg(rank1_heading, reverse_forward_heading):.9f}",
                    "same_forward_distance_disagreement": f"{abs(rank1_distance - reverse_forward_distance):.9f}",
                    "same_forward_avg_heading": f"{avg_heading:.9f}",
                    "same_forward_avg_distance": f"{avg_distance:.9f}",
                    "rank1_angle_abs_error": f"{rank1_angle_error:.9f}",
                    "same_forward_avg_angle_abs_error": f"{avg_angle_error:.9f}",
                    "rank1_distance_abs_error": f"{rank1_dist_error:.9f}",
                    "same_forward_avg_distance_abs_error": f"{avg_dist_error:.9f}",
                }
            )
    return {
        "rows": count,
        "rank1_angle_mae": sum(rank1_angle_errors) / count if count else None,
        "same_forward_avg_angle_mae": sum(avg_angle_errors) / count if count else None,
        "rank1_distance_mae": sum(distance_errors) / count if count else None,
        "rank1_angle_ge_0p5": sum(err >= 0.5 for err in rank1_angle_errors),
        "same_forward_avg_angle_ge_0p5": sum(err >= 0.5 for err in avg_angle_errors),
        "rank1_angle_ge_1p0": sum(err >= 1.0 for err in rank1_angle_errors),
        "same_forward_avg_angle_ge_1p0": sum(err >= 1.0 for err in avg_angle_errors),
    }


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    checkpoint = Path(args.checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {actual_sha} != {args.expected_checkpoint_sha256}")

    manifest_rows = read_manifest(Path(args.manifest_jsonl))
    materialize_report = materialize_json_subset(manifest_rows, Path(args.json_subset_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model)
    model.to(device)
    model.eval()
    load_checkpoint(model, checkpoint, device)
    loader = build_loader(args)

    all_pair_ids: list[str] = []
    all_group_ids: list[str] = []
    all_json_paths: list[str] = []
    all_pred_heading: list[torch.Tensor] = []
    all_pred_distance: list[torch.Tensor] = []
    all_reverse_heading: list[torch.Tensor] = []
    all_reverse_distance: list[torch.Tensor] = []
    all_target_heading: list[torch.Tensor] = []
    all_target_distance: list[torch.Tensor] = []

    for batch in loader:
        view1, view2 = batch
        for view in batch:
            for name in "img camera_intrinsics camera_pose".split():
                if name in view:
                    view[name] = view[name].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            pred1, pred2 = model(view1, view2)
            pred2_swapped = None
            if args.reverse_source == "true_swap_pred2_inverse":
                _, pred2_swapped = model(view2, view1)
        reverse_heading_vec, reverse_range_value = select_reverse_prediction_tensors(
            args.reverse_source,
            pred1,
            pred2_swapped=pred2_swapped,
        )
        pred_deg = heading_deg_from_vec(pred2["heading_vec"].detach().cpu())
        reverse_deg = heading_deg_from_vec(torch.as_tensor(reverse_heading_vec).detach().cpu())
        all_pair_ids.extend(derive_pair_ids(view2))
        all_group_ids.extend(as_string_list(view2["group_id"]))
        all_json_paths.extend(as_string_list(view2["json_path"]))
        all_pred_heading.append(pred_deg)
        all_pred_distance.append(pred2["range_value"].view(-1).detach().cpu())
        all_reverse_heading.append(reverse_deg)
        all_reverse_distance.append(torch.as_tensor(reverse_range_value).view(-1).detach().cpu())
        all_target_heading.append(torch.as_tensor(view2["heading_deg"]).detach().cpu())
        all_target_distance.append(torch.as_tensor(view2["range_value"]).detach().cpu())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rank1_same_forward_reverse_predictions.csv"
    metrics = write_same_forward_reverse_csv(
        csv_path,
        pair_ids=all_pair_ids,
        group_ids=all_group_ids,
        json_paths=all_json_paths,
        pred_heading=torch.cat(all_pred_heading),
        pred_distance=torch.cat(all_pred_distance),
        reverse_heading=torch.cat(all_reverse_heading),
        reverse_distance=torch.cat(all_reverse_distance),
        target_heading=torch.cat(all_target_heading),
        target_distance=torch.cat(all_target_distance),
    )
    report = {
        "status": "pass",
        "manifest_jsonl": str(args.manifest_jsonl),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "output_dir": str(output_dir),
        "prediction_csv": str(csv_path),
        "manifest_rows": len(manifest_rows),
        "materialize_report": materialize_report,
        "metrics": metrics,
        "elapsed_sec": time.time() - started,
        "device": str(device),
        "reverse_source": args.reverse_source,
        "inverse_policy": {"heading": "neg_deg", "range": "neg"},
    }
    (output_dir / "same_forward_reverse_eval_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "same_forward_reverse_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rank1 Reloc3r pose2 plus same-forward pose1 inverse output.")
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--json-subset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reverse-source",
        choices=["same_forward_pose1", "true_swap_pred2_inverse"],
        default="same_forward_pose1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
