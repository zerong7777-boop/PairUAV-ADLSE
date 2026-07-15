from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAIN_JSON_ROOT,
    DEFAULT_VAL_JSON_ROOT,
    DEFAULT_WSTRIP_CHECKPOINT,
    circular_abs_error_deg,
    ensure_run_root,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_MODEL_EXPR = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase91 G2 layer-wise ridge probe.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-json-root", type=Path, default=DEFAULT_TRAIN_JSON_ROOT)
    parser.add_argument("--val-json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WSTRIP_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--max-train-pairs", type=int, default=512)
    parser.add_argument("--max-val-pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> str:
    if not checkpoint_path or not checkpoint_path.exists():
        return f"checkpoint_missing_or_not_used: {checkpoint_path}"
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    new_state = dict(state_dict)
    for key, value in list(state_dict.items()):
        if key.startswith("dec_blocks2"):
            new_state[key.replace("dec_blocks2", "dec_blocks")] = value
    return str(model.load_state_dict(new_state, strict=False))


def move_batch_to_device(batch, device: torch.device):
    for view in batch:
        for name in "img camera_intrinsics camera_pose".split():
            if name in view and hasattr(view[name], "to"):
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def target_tensors(view) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heading_deg = torch.as_tensor(view["heading_deg"]).float().view(-1)
    heading_rad = torch.deg2rad(heading_deg)
    heading_vec = torch.stack([torch.cos(heading_rad), torch.sin(heading_rad)], dim=-1)
    range_value = torch.as_tensor(view["range_value"]).float().view(-1, 1)
    return heading_deg, heading_vec, range_value


def extract_features(model, loader, device: torch.device, max_pairs: int, amp: int) -> dict:
    layer_features: dict[int, list[torch.Tensor]] = {}
    headings_deg = []
    heading_vecs = []
    ranges = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            view1, view2 = batch
            with torch.cuda.amp.autocast(enabled=bool(amp)):
                (shape1, shape2), (feat1, feat2), (pos1, pos2) = model._encoder(view1, view2)
                _dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
            dec2 = [tok.detach().float() for tok in dec2]
            current = int(dec2[-1].shape[0])
            take = min(current, max_pairs - processed)
            if take <= 0:
                break
            for layer_id, tokens in enumerate(dec2):
                pooled = tokens[:take].mean(dim=1).detach().cpu()
                layer_features.setdefault(layer_id, []).append(pooled)
            h_deg, h_vec, r = target_tensors(view2)
            headings_deg.append(h_deg[:take].cpu())
            heading_vecs.append(h_vec[:take].cpu())
            ranges.append(r[:take].cpu())
            processed += take
            if processed >= max_pairs:
                break

    return {
        "features": {layer: torch.cat(parts, dim=0) for layer, parts in layer_features.items()},
        "heading_deg": torch.cat(headings_deg, dim=0),
        "heading_vec": torch.cat(heading_vecs, dim=0),
        "range": torch.cat(ranges, dim=0),
        "processed": processed,
    }


def standardize(train_x: torch.Tensor, val_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def ridge_fit(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    lhs = x.T @ x + float(ridge) * eye
    rhs = x.T @ y
    return torch.linalg.solve(lhs, rhs)


def ridge_predict(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    return x @ weight


def heading_metrics(pred_vec: torch.Tensor, target_deg: torch.Tensor) -> tuple[float, torch.Tensor]:
    pred_vec = F.normalize(pred_vec, dim=-1, eps=1e-6)
    pred_deg = torch.rad2deg(torch.atan2(pred_vec[:, 1], pred_vec[:, 0]))
    errors = [circular_abs_error_deg(float(p), float(t)) for p, t in zip(pred_deg, target_deg)]
    return float(sum(errors) / len(errors)), pred_deg


def displacement_epe(pred_deg: torch.Tensor, pred_range: torch.Tensor, target_deg: torch.Tensor, target_range: torch.Tensor) -> float:
    pred_rad = torch.deg2rad(pred_deg.float())
    target_rad = torch.deg2rad(target_deg.float())
    pred_vec = torch.stack([torch.cos(pred_rad), torch.sin(pred_rad)], dim=-1) * pred_range.float().view(-1, 1)
    target_vec = torch.stack([torch.cos(target_rad), torch.sin(target_rad)], dim=-1) * target_range.float().view(-1, 1)
    return float(torch.linalg.norm(pred_vec - target_vec, dim=-1).mean().item())


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    ensure_run_root(run_root)

    from reloc3r.datasets import get_data_loader
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    train_expr = (
        "PairUAV("
        f"json_root={str(args.train_json_root)!r}, "
        f"image_root={str(args.image_root)!r}, "
        "split='train', resolution=(512,384), seed=777, require_labels=True)"
    )
    val_expr = (
        "PairUAV("
        f"json_root={str(args.val_json_root)!r}, "
        f"image_root={str(args.image_root)!r}, "
        "split='dev', resolution=(512,384), seed=777, require_labels=True)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model, {"Reloc3rRelpose": Reloc3rRelpose})
    model.to(device)
    model.eval()
    load_result = load_checkpoint(model, args.checkpoint, device)

    train_loader = get_data_loader(
        train_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )
    val_loader = get_data_loader(
        val_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )

    started = time.time()
    train = extract_features(model, train_loader, device, args.max_train_pairs, args.amp)
    val = extract_features(model, val_loader, device, args.max_val_pairs, args.amp)

    range_min = float(val["range"].min().item())
    range_max = float(val["range"].max().item())
    range_span = max(1e-6, range_max - range_min)
    rows = []

    for layer_id in sorted(train["features"]):
        train_x = train["features"][layer_id].float()
        val_x = val["features"][layer_id].float()
        train_x, val_x = standardize(train_x, val_x)

        heading_w = ridge_fit(train_x, train["heading_vec"].float(), args.ridge)
        range_w = ridge_fit(train_x, train["range"].float(), args.ridge)
        pred_heading_vec = ridge_predict(val_x, heading_w)
        pred_range = ridge_predict(val_x, range_w).view(-1)

        angle_mae, pred_deg = heading_metrics(pred_heading_vec, val["heading_deg"])
        distance_mae = float((pred_range - val["range"].view(-1)).abs().mean().item())
        angle_rel = angle_mae / 180.0
        distance_rel = distance_mae / range_span
        final_proxy = (angle_rel + distance_rel) / 2.0
        epe = displacement_epe(pred_deg, pred_range, val["heading_deg"], val["range"].view(-1))
        rows.append(
            {
                "layer_id": layer_id,
                "feature_dim": int(train["features"][layer_id].shape[1]),
                "train_pairs": train["processed"],
                "val_pairs": val["processed"],
                "angle_mae_deg": angle_mae,
                "distance_mae": distance_mae,
                "angle_rel_error": angle_rel,
                "distance_rel_error": distance_rel,
                "final_score_proxy": final_proxy,
                "displacement_epe": epe,
                "pool": "token_mean",
                "probe": "closed_form_ridge",
                "ridge": args.ridge,
            }
        )

    heading_best = min(rows, key=lambda row: row["angle_mae_deg"])
    range_best = min(rows, key=lambda row: row["distance_mae"])
    proxy_best = min(rows, key=lambda row: row["final_score_proxy"])
    summary = {
        "phase": "phase91_g2_layerwise_probe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_root": str(run_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_result": load_result,
        "device": str(device),
        "train_json_root": str(args.train_json_root),
        "val_json_root": str(args.val_json_root),
        "max_train_pairs": args.max_train_pairs,
        "max_val_pairs": args.max_val_pairs,
        "processed_train_pairs": train["processed"],
        "processed_val_pairs": val["processed"],
        "pool": "token_mean",
        "probe": "closed_form_ridge",
        "ridge": args.ridge,
        "range_min_val": range_min,
        "range_max_val": range_max,
        "heading_best_layer": heading_best["layer_id"],
        "range_best_layer": range_best["layer_id"],
        "proxy_best_layer": proxy_best["layer_id"],
        "evidence_depth_gap": int(heading_best["layer_id"]) - int(range_best["layer_id"]),
        "elapsed_sec": round(time.time() - started, 3),
        "pass": bool(rows),
    }

    write_csv(run_root / "layer_probes" / "layer_probe_metrics.csv", rows)
    write_json(run_root / "layer_probes" / "layer_probe_summary.json", summary)
    md = [
        "# Phase91 G2 Layer-Wise Probe",
        "",
        f"- processed_train_pairs: {train['processed']}",
        f"- processed_val_pairs: {val['processed']}",
        f"- probe: closed-form ridge on token-mean pooled decoder features",
        f"- heading_best_layer: {summary['heading_best_layer']}",
        f"- range_best_layer: {summary['range_best_layer']}",
        f"- proxy_best_layer: {summary['proxy_best_layer']}",
        f"- evidence_depth_gap: {summary['evidence_depth_gap']}",
        f"- elapsed_sec: {summary['elapsed_sec']}",
        "",
        "## Best Rows",
        "",
        "```json",
        json.dumps({"heading_best": heading_best, "range_best": range_best, "proxy_best": proxy_best}, ensure_ascii=False, indent=2),
        "```",
        "",
        "This is a diagnostic probe only. It is not a deployable model and should not be compared to trained H8 as a leaderboard candidate.",
    ]
    write_text(run_root / "layer_probes" / "layer_probe_summary.md", "\n".join(md))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

