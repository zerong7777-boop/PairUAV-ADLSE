"""Fixed-manifest PairUAV eval runner smoke implementation.

The initial A-v3.2c runner supports dry_run_identity_only. It validates and
round-trips identity without loading a checkpoint or changing model behavior.
"""
import argparse
import csv
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np


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

DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def file_hash(path):
    p = Path(path)
    if not path or not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def safe_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_optional_float(value):
    if value is None:
        return ""
    return f"{float(value):.12g}"


def angle_abs_error(pred_deg, gt_deg):
    pred = safe_float(pred_deg)
    gt = safe_float(gt_deg)
    if pred is None or gt is None:
        return None
    diff = abs((pred - gt + 180.0) % 360.0 - 180.0)
    return diff


def relative_error(abs_error, gt_value):
    err = safe_float(abs_error)
    gt = safe_float(gt_value)
    if err is None or gt is None or abs(gt) < 1e-12:
        return None
    return err / abs(gt)


def joint_error(heading_error, range_error):
    heading = safe_float(heading_error)
    range_value = safe_float(range_error)
    if heading is None or range_value is None:
        return None
    return math.sqrt(heading * heading + range_value * range_value)


def load_checkpoint(model, checkpoint_path, device):
    import torch

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return model.load_state_dict(state_dict, strict=False)


def resolve_pairuav_image(image_root, relative_path):
    root = Path(image_root)
    rel = Path(relative_path)
    candidates = [root / rel, root / rel.name, root / rel.parent.name / rel.name]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def output_row_from_manifest(row, variant_id, variant_config, checkpoint_path, status, reason="", prediction_heading=None, prediction_range=None):
    heading_abs = angle_abs_error(prediction_heading, row.get("gt_heading"))
    range_abs = None
    pred_range = safe_float(prediction_range)
    gt_range = safe_float(row.get("gt_range"))
    if pred_range is not None and gt_range is not None:
        range_abs = abs(pred_range - gt_range)
    heading_rel = relative_error(heading_abs, row.get("gt_heading"))
    range_rel = relative_error(range_abs, row.get("gt_range"))
    joint = joint_error(heading_abs, range_abs)
    return {
        "manifest_version": row.get("manifest_version", ""),
        "manifest_hash": row.get("manifest_hash", ""),
        "eval_config_hash": file_hash(variant_config),
        "checkpoint_path": checkpoint_path,
        "variant_id": variant_id,
        "canonical_pair_id": row.get("canonical_pair_id", ""),
        "source_image_key": row.get("source_image_key", ""),
        "target_image_key": row.get("target_image_key", ""),
        "source_image_path": row.get("source_image_path", ""),
        "target_image_path": row.get("target_image_path", ""),
        "prediction_heading": format_optional_float(prediction_heading),
        "prediction_range": format_optional_float(prediction_range),
        "gt_heading": row.get("gt_heading", ""),
        "gt_range": row.get("gt_range", ""),
        "heading_abs_error": format_optional_float(heading_abs),
        "heading_rel_error": format_optional_float(heading_rel),
        "range_abs_error": format_optional_float(range_abs),
        "range_rel_error": format_optional_float(range_rel),
        "joint_error": format_optional_float(joint),
        "row_status": status,
        "failure_reason": reason,
    }


def manifest_row_to_sample(row):
    group_id = row.get("group_id") or row.get("target_key") or "missing"
    json_id = Path(row.get("canonical_pair_id", "missing/missing")).name
    target_group_index = 0
    digits = "".join(ch for ch in str(group_id) if ch.isdigit())
    if digits:
        target_group_index = int(digits) % 4096
    return {
        "group_id": str(group_id),
        "target_group_index": np.int64(target_group_index),
        "scene_id": str(row.get("scene_key") or f"group_{group_id}"),
        "json_id": str(json_id),
        "json_path": row.get("json_path", ""),
        "image_a": row.get("source_image_path") or row.get("source_image_key"),
        "image_b": row.get("target_image_path") or row.get("target_image_key"),
        "heading_deg": np.float32(safe_float(row.get("gt_heading")) or 0.0),
        "range_value": np.float32(safe_float(row.get("gt_range")) or 0.0),
        "canonical_pair_id": row.get("canonical_pair_id", ""),
        "manifest_hash": row.get("manifest_hash", ""),
        "manifest_row_id": row.get("manifest_row_id", ""),
        "source_image_key": row.get("source_image_key", ""),
        "target_image_key": row.get("target_image_key", ""),
    }


def make_fixed_manifest_pairuav_class():
    from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
    from reloc3r.datasets.pairuav import BSCR_GLOBAL_FEATURE_NAMES, MATCHER_FEATURE_NAMES, PairUAV
    from reloc3r.datasets.utils.transforms import ImgNorm

    class FixedManifestPairUAV(PairUAV):
        def __init__(self, manifest_rows, image_root, split="test", resolution=(512, 384), seed=777):
            BaseStereoViewDataset.__init__(self, split=split, resolution=resolution, transform=ImgNorm, aug_crop=False, seed=seed)
            self.image_root = Path(image_root)
            self.require_labels = True
            self.num_target_groups = 4096
            self.matcher_feature_manifest = None
            self.bscr_feature_manifest = None
            self.matcher_feature_dim = len(MATCHER_FEATURE_NAMES)
            self.bscr_global_dim = len(BSCR_GLOBAL_FEATURE_NAMES)
            self.matcher_features_by_id = {}
            self.bscr_features_by_id = {}
            self.samples = [manifest_row_to_sample(row) for row in manifest_rows]

        def _build_view(self, sample, image_key, view_suffix, resolution, rng):
            view = super()._build_view(sample, image_key, view_suffix, resolution, rng)
            view.update(
                {
                    "canonical_pair_id": sample["canonical_pair_id"],
                    "manifest_hash": sample["manifest_hash"],
                    "manifest_row_id": sample["manifest_row_id"],
                    "source_image_key": sample["source_image_key"],
                    "target_image_key": sample["target_image_key"],
                }
            )
            return view

    return FixedManifestPairUAV


def run_dry_identity(manifest_rows, variant_id, variant_config, checkpoint_path=""):
    config_hash = file_hash(variant_config)
    rows = []
    seen = set()
    for row in manifest_rows:
        cid = row.get("canonical_pair_id", "")
        key = (cid, variant_id)
        status = "ok"
        reason = ""
        if not cid:
            status = "missing_identity"
            reason = "missing_canonical_pair_id"
        elif key in seen:
            status = "duplicate_identity"
            reason = "duplicate_canonical_pair_id_variant_id"
        elif not row.get("source_image_key") or not row.get("target_image_key"):
            status = "metadata_loss"
            reason = "missing_source_or_target_identity"
        seen.add(key)
        rows.append(
            {
                "manifest_version": row.get("manifest_version", ""),
                "manifest_hash": row.get("manifest_hash", ""),
                "eval_config_hash": config_hash,
                "checkpoint_path": checkpoint_path,
                "variant_id": variant_id,
                "canonical_pair_id": cid,
                "source_image_key": row.get("source_image_key", ""),
                "target_image_key": row.get("target_image_key", ""),
                "source_image_path": row.get("source_image_path", ""),
                "target_image_path": row.get("target_image_path", ""),
                "prediction_heading": "",
                "prediction_range": "",
                "gt_heading": row.get("gt_heading", ""),
                "gt_range": row.get("gt_range", ""),
                "heading_abs_error": "",
                "heading_rel_error": "",
                "range_abs_error": "",
                "range_rel_error": "",
                "joint_error": "",
                "row_status": status,
                "failure_reason": reason,
            }
        )
    return rows


def _move_batch_to_device(batch, device):
    for view in batch:
        for name in "img camera_intrinsics camera_pose".split():
            if name in view:
                view[name] = view[name].to(device, non_blocking=True)


def _as_list(value, n):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(n)]


def run_model_forward_tiny(manifest_rows, image_root, checkpoint_path, variant_id, variant_config, model_expr=DEFAULT_MODEL, batch_size=4, num_workers=0, device_name="auto"):
    import torch
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    config_hash = file_hash(variant_config)
    outputs = []
    valid_rows = []
    for row in manifest_rows:
        if not row.get("canonical_pair_id") or not row.get("source_image_key") or not row.get("target_image_key"):
            outputs.append(output_row_from_manifest(row, variant_id, variant_config, checkpoint_path, "metadata_loss", "missing_identity_fields"))
            continue
        if not resolve_pairuav_image(image_root, row.get("source_image_path") or row.get("source_image_key")):
            outputs.append(output_row_from_manifest(row, variant_id, variant_config, checkpoint_path, "missing_image", "missing_source_image"))
            continue
        if not resolve_pairuav_image(image_root, row.get("target_image_path") or row.get("target_image_key")):
            outputs.append(output_row_from_manifest(row, variant_id, variant_config, checkpoint_path, "missing_image", "missing_target_image"))
            continue
        valid_rows.append(row)

    if not valid_rows:
        return outputs

    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else ("cpu" if device_name == "auto" else device_name))
    model = eval(model_expr)
    model.to(device)
    model.eval()
    load_checkpoint(model, checkpoint_path, device)
    FixedManifestPairUAV = make_fixed_manifest_pairuav_class()
    dataset = FixedManifestPairUAV(valid_rows, image_root=image_root, split="test", resolution=(512, 384), seed=777)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    by_id = {}
    with torch.no_grad():
        for batch in loader:
            view1, view2 = batch
            try:
                _move_batch_to_device(batch, device)
                _, pred2 = model(view1, view2)
                pred_heading = pred2["heading_vec"]
                pred_range = pred2["range_value"].view(-1)
                pred_deg = torch.rad2deg(torch.atan2(pred_heading[:, 1], pred_heading[:, 0])).detach().cpu().numpy()
                pred_range = pred_range.detach().cpu().numpy()
                cids = _as_list(view2.get("canonical_pair_id", ""), len(pred_deg))
                for i, cid in enumerate(cids):
                    by_id[str(cid)] = (float(pred_deg[i]), float(pred_range[i]), "ok", "")
            except Exception as exc:
                cids = _as_list(view2.get("canonical_pair_id", ""), len(view2.get("img", [])))
                reason = type(exc).__name__ + ":" + str(exc)[:120]
                tb = traceback.format_exc(limit=14).replace("\n", " | ")
                reason = (reason + " | " + tb)[:2400]
                for cid in cids:
                    by_id[str(cid)] = (None, None, "model_error", reason)

    for row in valid_rows:
        pred_heading, pred_range, status, reason = by_id.get(row.get("canonical_pair_id", ""), (None, None, "model_error", "missing_prediction"))
        outputs.append(output_row_from_manifest(row, variant_id, variant_config, checkpoint_path, status, reason, pred_heading, pred_range))
    order = {row.get("canonical_pair_id", ""): i for i, row in enumerate(manifest_rows)}
    return sorted(outputs, key=lambda r: order.get(r.get("canonical_pair_id", ""), len(order)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--variant-config", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", default="dry_run_identity_only", choices=["dry_run_identity_only", "model_forward_tiny"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    rows = read_csv(args.fixed_manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if args.mode == "dry_run_identity_only":
        output = run_dry_identity(rows, args.variant_id, args.variant_config, args.checkpoint)
    else:
        output = run_model_forward_tiny(
            rows,
            image_root=args.image_root,
            checkpoint_path=args.checkpoint,
            variant_id=args.variant_id,
            variant_config=args.variant_config,
            model_expr=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )
    write_csv(args.output_csv, output, OUTPUT_COLUMNS)


if __name__ == "__main__":
    main()
