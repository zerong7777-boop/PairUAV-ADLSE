#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.datasets import get_data_loader
from reloc3r.datasets.pairuav import PairUAV
from reloc3r.reloc3r_relpose import Reloc3rRelpose


DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def angle_abs_error_deg(pred: float, target: float) -> float:
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_flat_float_list(value: Any) -> list[float]:
    return [float(x) for x in torch.as_tensor(value, dtype=torch.float32).reshape(-1).tolist()]


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def write_prediction_csv(
    path: Path,
    *,
    pair_ids: list[str],
    group_ids: list[str],
    json_paths: list[str],
    pred_heading: Any,
    pred_distance: Any,
    target_heading: Any,
    target_distance: Any,
) -> dict[str, Any]:
    pred_heading_values = as_flat_float_list(pred_heading)
    pred_distance_values = as_flat_float_list(pred_distance)
    target_heading_values = as_flat_float_list(target_heading)
    target_distance_values = as_flat_float_list(target_distance)
    count = len(pair_ids)
    lengths = {
        "pair_ids": len(pair_ids),
        "group_ids": len(group_ids),
        "json_paths": len(json_paths),
        "pred_heading": len(pred_heading_values),
        "pred_distance": len(pred_distance_values),
        "target_heading": len(target_heading_values),
        "target_distance": len(target_distance_values),
    }
    if any(value != count for value in lengths.values()):
        raise ValueError(f"prediction output length mismatch: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    angle_errors: list[float] = []
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
            "rank1_angle_abs_error",
            "rank1_distance_abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, pair_id in enumerate(pair_ids):
            angle_error = angle_abs_error_deg(pred_heading_values[idx], target_heading_values[idx])
            distance_error = abs(pred_distance_values[idx] - target_distance_values[idx])
            angle_errors.append(angle_error)
            distance_errors.append(distance_error)
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "split": "train",
                    "group_id": group_ids[idx],
                    "json_path": json_paths[idx],
                    "target_heading": f"{target_heading_values[idx]:.9f}",
                    "target_distance": f"{target_distance_values[idx]:.9f}",
                    "rank1_heading": f"{pred_heading_values[idx]:.9f}",
                    "rank1_distance": f"{pred_distance_values[idx]:.9f}",
                    "rank1_angle_abs_error": f"{angle_error:.9f}",
                    "rank1_distance_abs_error": f"{distance_error:.9f}",
                }
            )
    return {
        "rows": count,
        "angle_mae": sum(angle_errors) / count if count else None,
        "distance_mae": sum(distance_errors) / count if count else None,
        "angle_ge_0p5": sum(err >= 0.5 for err in angle_errors),
        "angle_ge_1p0": sum(err >= 1.0 for err in angle_errors),
        "angle_ge_2p0": sum(err >= 2.0 for err in angle_errors),
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def materialize_json_subset(manifest_rows: list[dict[str, Any]], json_subset_root: Path) -> dict[str, Any]:
    json_subset_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in manifest_rows:
        src = Path(row["json_path"])
        dst = json_subset_root / str(row["group_id"]) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written += 1
    return {"json_subset_root": str(json_subset_root), "written": written}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    print(model.load_state_dict(state_dict, strict=False))


def build_loader(args: argparse.Namespace) -> Any:
    dataset = PairUAV(
        json_root=args.json_subset_root,
        image_root=args.image_root,
        split="train",
        resolution=(512, 384),
        seed=777,
        require_labels=True,
    )
    return get_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )


def derive_pair_ids(view2: dict[str, Any]) -> list[str]:
    if "sample_id" in view2:
        return as_string_list(view2["sample_id"])
    group_ids = as_string_list(view2["group_id"])
    json_paths = as_string_list(view2["json_path"])
    return [f"{group_id}/{Path(json_path).stem}" for group_id, json_path in zip(group_ids, json_paths)]


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
    all_target_heading: list[torch.Tensor] = []
    all_target_distance: list[torch.Tensor] = []

    for batch in loader:
        view1, view2 = batch
        for view in batch:
            for name in "img camera_intrinsics camera_pose".split():
                if name in view:
                    view[name] = view[name].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            _, pred2 = model(view1, view2)
        heading_vec = pred2["heading_vec"]
        pred_deg = torch.rad2deg(torch.atan2(heading_vec[:, 1], heading_vec[:, 0])).detach().cpu()
        pred_range = pred2["range_value"].view(-1).detach().cpu()
        all_pair_ids.extend(derive_pair_ids(view2))
        all_group_ids.extend(as_string_list(view2["group_id"]))
        all_json_paths.extend(as_string_list(view2["json_path"]))
        all_pred_heading.append(pred_deg)
        all_pred_distance.append(pred_range)
        all_target_heading.append(torch.as_tensor(view2["heading_deg"]).detach().cpu())
        all_target_distance.append(torch.as_tensor(view2["range_value"]).detach().cpu())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_prediction_csv(
        output_dir / "rank1_predictions.csv",
        pair_ids=all_pair_ids,
        group_ids=all_group_ids,
        json_paths=all_json_paths,
        pred_heading=torch.cat(all_pred_heading),
        pred_distance=torch.cat(all_pred_distance),
        target_heading=torch.cat(all_target_heading),
        target_distance=torch.cat(all_target_distance),
    )
    report = {
        "status": "pass",
        "manifest_jsonl": str(args.manifest_jsonl),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "output_dir": str(output_dir),
        "rank1_predictions_csv": str(output_dir / "rank1_predictions.csv"),
        "manifest_rows": len(manifest_rows),
        "materialize_report": materialize_report,
        "metrics": metrics,
        "elapsed_sec": time.time() - started,
        "device": str(device),
    }
    (output_dir / "rank1_eval_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rank1_eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
