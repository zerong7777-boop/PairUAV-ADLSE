"""Fixed-manifest PairUAV eval runner for reloc3r_route_local_pair_v2.

Additive repair path for A-v3.2c. Does not modify the older Reloc3rRelpose
runner. This runner uses the checkpoint's real model architecture and strict
checkpoint loading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OFFICIAL_ROOT = Path("/media/jgzn/SSD_lexar/RZ/UAVM/official/UAVM_2026")
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

from baseline.reloc3r_route_v2 import (  # noqa: E402
    Reloc3rRouteModelV2,
    denormalize_range_symmetric,
    load_rgb_tensor,
)
from transformers import AutoFeatureExtractor  # noqa: E402


OUTPUT_COLUMNS = [
    "manifest_version",
    "manifest_hash",
    "eval_config_hash",
    "checkpoint_path",
    "variant_id",
    "canonical_pair_id",
    "source_image_key",
    "target_image_key",
    "source_image_path",
    "target_image_path",
    "prediction_heading",
    "prediction_range",
    "gt_heading",
    "gt_range",
    "heading_abs_error",
    "heading_rel_error",
    "range_abs_error",
    "range_rel_error",
    "joint_error",
    "row_status",
    "failure_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any):
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def angle_abs_error(pred_deg: Any, gt_deg: Any):
    pred = safe_float(pred_deg)
    gt = safe_float(gt_deg)
    if pred is None or gt is None:
        return None
    return abs((pred - gt + 180.0) % 360.0 - 180.0)


def relative_error(abs_error: float | None, gt_value: Any):
    gt = safe_float(gt_value)
    if abs_error is None or gt is None or abs(gt) < 1e-12:
        return None
    return abs_error / abs(gt)


def joint_error(heading_abs: float | None, range_abs: float | None):
    if heading_abs is None or range_abs is None:
        return None
    return math.sqrt(heading_abs * heading_abs + range_abs * range_abs)


def resolve_image(image_root: Path, relative: str) -> Path:
    rel = Path(relative)
    candidates = [image_root / rel, image_root / rel.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for {relative}; tried {candidates}")


class FixedManifestRouteV2Dataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path, model_path: Path) -> None:
        self.rows = rows
        self.image_root = image_root
        self.processor = AutoFeatureExtractor.from_pretrained(str(model_path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        source = row.get("source_image_path") or row.get("source_image_key")
        target = row.get("target_image_path") or row.get("target_image_key")
        image_a = load_rgb_tensor(resolve_image(self.image_root, source), self.processor)
        image_b = load_rgb_tensor(resolve_image(self.image_root, target), self.processor)
        return image_a, image_b, row.get("canonical_pair_id", "")


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(isinstance(k, str) and k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint: Path, model_path: Path, device: torch.device, hidden_dim: int, dropout: float):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise TypeError("Expected checkpoint payload with model_state_dict")
    model = Reloc3rRouteModelV2(model_path, hidden_dim=hidden_dim, dropout=dropout)
    result = model.load_state_dict(normalize_state_dict(payload["model_state_dict"]), strict=True)
    model.to(device)
    model.eval()
    return model, {
        "payload_keys": sorted(payload.keys()),
        "model_variant": payload.get("model_variant", ""),
        "epoch": payload.get("epoch", ""),
        "next_epoch": payload.get("next_epoch", ""),
        "missing_key_count": len(result.missing_keys),
        "unexpected_key_count": len(result.unexpected_keys),
    }


def output_row(row, variant_id, variant_config, checkpoint_path, status, reason="", pred_heading=None, pred_range=None):
    heading_abs = angle_abs_error(pred_heading, row.get("gt_heading"))
    pred_range_float = safe_float(pred_range)
    gt_range = safe_float(row.get("gt_range"))
    range_abs = None if pred_range_float is None or gt_range is None else abs(pred_range_float - gt_range)
    heading_rel = relative_error(heading_abs, row.get("gt_heading"))
    range_rel = relative_error(range_abs, row.get("gt_range"))
    return {
        "manifest_version": row.get("manifest_version", ""),
        "manifest_hash": row.get("manifest_hash", ""),
        "eval_config_hash": file_hash(Path(variant_config)),
        "checkpoint_path": str(checkpoint_path),
        "variant_id": variant_id,
        "canonical_pair_id": row.get("canonical_pair_id", ""),
        "source_image_key": row.get("source_image_key", ""),
        "target_image_key": row.get("target_image_key", ""),
        "source_image_path": row.get("source_image_path", ""),
        "target_image_path": row.get("target_image_path", ""),
        "prediction_heading": fmt(safe_float(pred_heading)),
        "prediction_range": fmt(pred_range_float),
        "gt_heading": row.get("gt_heading", ""),
        "gt_range": row.get("gt_range", ""),
        "heading_abs_error": fmt(heading_abs),
        "heading_rel_error": fmt(heading_rel),
        "range_abs_error": fmt(range_abs),
        "range_rel_error": fmt(range_rel),
        "joint_error": fmt(joint_error(heading_abs, range_abs)),
        "row_status": status,
        "failure_reason": reason,
    }


def run(args) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(args.fixed_manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    valid = []
    outputs = []
    for row in rows:
        if not row.get("canonical_pair_id") or not row.get("source_image_key") or not row.get("target_image_key"):
            outputs.append(output_row(row, args.variant_id, args.variant_config, args.checkpoint, "metadata_loss", "missing_identity_fields"))
            continue
        valid.append(row)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    model, diagnostics = load_model(args.checkpoint, args.model_path, device, args.hidden_dim, args.dropout)
    dataset = FixedManifestRouteV2Dataset(valid, args.image_root, args.model_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    by_id = {}
    with torch.inference_mode():
        for image_a, image_b, cids in loader:
            image_a = image_a.to(device, non_blocking=True)
            image_b = image_b.to(device, non_blocking=True)
            pred = model(image_a, image_b)
            pred_xy = F.normalize(pred[:, :2], dim=-1)
            pred_deg = torch.rad2deg(torch.atan2(pred_xy[:, 1], pred_xy[:, 0]))
            pred_range = denormalize_range_symmetric(pred[:, 2])
            for cid, deg, rng in zip(cids, pred_deg.detach().cpu().tolist(), pred_range.detach().cpu().tolist()):
                by_id[str(cid)] = (float(deg), float(rng))
    for row in valid:
        pred = by_id.get(row.get("canonical_pair_id", ""))
        if pred is None:
            outputs.append(output_row(row, args.variant_id, args.variant_config, args.checkpoint, "model_error", "missing_prediction"))
        else:
            outputs.append(output_row(row, args.variant_id, args.variant_config, args.checkpoint, "ok", "", pred[0], pred[1]))
    order = {row.get("canonical_pair_id", ""): i for i, row in enumerate(rows)}
    outputs.sort(key=lambda r: order.get(r.get("canonical_pair_id", ""), len(order)))
    diagnostics.update({
        "row_count": len(rows),
        "valid_row_count": len(valid),
        "output_row_count": len(outputs),
        "device": str(device),
    })
    return outputs, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    outputs, diagnostics = run(args)
    write_csv(args.output_csv, outputs, OUTPUT_COLUMNS)
    write_json(args.diagnostics_json, diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
