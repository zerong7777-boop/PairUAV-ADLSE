from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
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


def parse_layers(raw: str) -> list[int]:
    layers = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    if not layers:
        raise ValueError("--layers must contain at least one layer id")
    return sorted(set(layers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase91 G2b H0/H8-like per-layer head probe.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-json-root", type=Path, default=DEFAULT_TRAIN_JSON_ROOT)
    parser.add_argument("--val-json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WSTRIP_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--layers", default="4,8,11,12")
    parser.add_argument("--max-train-pairs", type=int, default=256)
    parser.add_argument("--max-val-pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--range-scale", type=float, default=100.0)
    parser.add_argument("--num-resconv-block", type=int, default=1)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


class ResConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.res_conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        return res + x


class HeadLikeBranch(nn.Module):
    """A compact copy of the PairUAVHead token->grid readout path."""

    def __init__(self, token_dim: int, output_dim: int = 1024, num_resconv_block: int = 1):
        super().__init__()
        self.proj = nn.Linear(token_dim, output_dim)
        self.res_conv = nn.ModuleList([ResConvBlock(output_dim) for _ in range(int(num_resconv_block))])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        bsz, seq_len, _dim = tokens.shape
        if seq_len != grid_h * grid_w:
            raise ValueError(f"token seq_len={seq_len} does not match grid {grid_h}x{grid_w}")
        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).contiguous().view(bsz, -1, grid_h, grid_w)
        for block in self.res_conv:
            feat = block(feat)
        feat = self.avgpool(feat).view(bsz, -1)
        return self.more_mlps(feat)


class AxisHeadProbe(nn.Module):
    """Separate H0/H8-like readout branches for heading and range."""

    def __init__(self, token_dim: int, num_resconv_block: int = 1):
        super().__init__()
        self.heading_branch = HeadLikeBranch(token_dim, num_resconv_block=num_resconv_block)
        self.range_branch = HeadLikeBranch(token_dim, num_resconv_block=num_resconv_block)
        self.fc_heading = nn.Linear(1024, 2)
        self.fc_range = nn.Linear(1024, 1)

    def forward(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> dict[str, torch.Tensor]:
        heading_feat = self.heading_branch(tokens, grid_h, grid_w)
        range_feat = self.range_branch(tokens, grid_h, grid_w)
        return {
            "heading_vec": F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6),
            "range_value": self.fc_range(range_feat).view(-1),
        }


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


def target_tensors(view, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heading_deg = torch.as_tensor(view["heading_deg"], device=device).float().view(-1)
    heading_rad = torch.deg2rad(heading_deg)
    heading_vec = torch.stack([torch.cos(heading_rad), torch.sin(heading_rad)], dim=-1)
    range_value = torch.as_tensor(view["range_value"], device=device).float().view(-1)
    return heading_deg, heading_vec, range_value


def grid_shape_from_view(view) -> tuple[int, int]:
    img = view["img"]
    height = int(img.shape[-2])
    width = int(img.shape[-1])
    return height // 16, width // 16


def selected_tokens(decout: list[torch.Tensor], layer_id: int) -> torch.Tensor:
    if layer_id < 0 or layer_id >= len(decout):
        raise IndexError(f"layer_id={layer_id} outside decout length {len(decout)}")
    tokens = decout[layer_id]
    if tokens.shape[-1] != 768:
        raise ValueError(
            f"layer_id={layer_id} has dim={tokens.shape[-1]}; head-like probe expects decoder dim 768. "
            "Use layers 1..12."
        )
    return tokens.detach()


def angle_mae_and_pred_deg(pred_vec: torch.Tensor, target_deg: torch.Tensor) -> tuple[float, torch.Tensor]:
    pred_vec = F.normalize(pred_vec, dim=-1, eps=1e-6)
    pred_deg = torch.rad2deg(torch.atan2(pred_vec[:, 1], pred_vec[:, 0]))
    errors = [circular_abs_error_deg(float(p), float(t)) for p, t in zip(pred_deg.detach().cpu(), target_deg.detach().cpu())]
    return float(sum(errors) / len(errors)), pred_deg


def displacement_epe(pred_deg: torch.Tensor, pred_range: torch.Tensor, target_deg: torch.Tensor, target_range: torch.Tensor) -> float:
    pred_rad = torch.deg2rad(pred_deg.float())
    target_rad = torch.deg2rad(target_deg.float())
    pred_xy = torch.stack([torch.cos(pred_rad), torch.sin(pred_rad)], dim=-1) * pred_range.float().view(-1, 1)
    target_xy = torch.stack([torch.cos(target_rad), torch.sin(target_rad)], dim=-1) * target_range.float().view(-1, 1)
    return float(torch.linalg.norm(pred_xy - target_xy, dim=-1).mean().detach().cpu().item())


def build_loader(expr: str, batch_size: int, num_workers: int, shuffle: bool):
    from reloc3r.datasets import get_data_loader

    return get_data_loader(
        expr,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_mem=True,
    )


def run_backbone(model, batch, amp: int):
    view1, view2 = batch
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = model._encoder(view1, view2)
            _dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    return [tok.detach().float() for tok in dec2]


def train_epoch(
    model,
    probes: nn.ModuleDict,
    loader,
    layers: list[int],
    optimizer,
    device: torch.device,
    max_pairs: int,
    range_scale: float,
    amp: int,
) -> dict:
    processed = 0
    loss_sum = 0.0
    steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        view1, view2 = batch
        dec2 = run_backbone(model, batch, amp=amp)
        _heading_deg, heading_vec, range_value = target_tensors(view2, device)
        grid_h, grid_w = grid_shape_from_view(view2)
        take = min(int(range_value.shape[0]), max_pairs - processed)
        if take <= 0:
            break

        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        for layer_id in layers:
            tokens = selected_tokens(dec2, layer_id)[:take]
            outputs = probes[str(layer_id)](tokens, grid_h, grid_w)
            cos = (outputs["heading_vec"] * heading_vec[:take]).sum(dim=-1).clamp(-1.0, 1.0)
            heading_loss = (1.0 - cos).mean()
            range_loss = F.smooth_l1_loss(outputs["range_value"] / range_scale, range_value[:take] / range_scale)
            loss = heading_loss + range_loss
            total_loss = loss if total_loss is None else total_loss + loss
        assert total_loss is not None
        total_loss.backward()
        optimizer.step()

        processed += take
        steps += 1
        loss_sum += float(total_loss.detach().cpu().item())
        if processed >= max_pairs:
            break
    return {"processed_pairs": processed, "steps": steps, "mean_loss": loss_sum / max(1, steps)}


def evaluate(
    model,
    probes: nn.ModuleDict,
    loader,
    layers: list[int],
    device: torch.device,
    max_pairs: int,
    range_scale: float,
    amp: int,
) -> list[dict]:
    by_layer = {
        layer_id: {
            "pred_vec": [],
            "pred_range": [],
            "target_heading_deg": [],
            "target_range": [],
        }
        for layer_id in layers
    }
    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            view1, view2 = batch
            dec2 = run_backbone(model, batch, amp=amp)
            heading_deg, _heading_vec, range_value = target_tensors(view2, device)
            grid_h, grid_w = grid_shape_from_view(view2)
            take = min(int(range_value.shape[0]), max_pairs - processed)
            if take <= 0:
                break
            for layer_id in layers:
                tokens = selected_tokens(dec2, layer_id)[:take]
                outputs = probes[str(layer_id)](tokens, grid_h, grid_w)
                bucket = by_layer[layer_id]
                bucket["pred_vec"].append(outputs["heading_vec"].detach().cpu())
                bucket["pred_range"].append(outputs["range_value"].detach().cpu())
                bucket["target_heading_deg"].append(heading_deg[:take].detach().cpu())
                bucket["target_range"].append(range_value[:take].detach().cpu())
            processed += take
            if processed >= max_pairs:
                break

    rows = []
    for layer_id in layers:
        bucket = by_layer[layer_id]
        pred_vec = torch.cat(bucket["pred_vec"], dim=0)
        pred_range = torch.cat(bucket["pred_range"], dim=0).view(-1)
        target_heading_deg = torch.cat(bucket["target_heading_deg"], dim=0).view(-1)
        target_range = torch.cat(bucket["target_range"], dim=0).view(-1)
        angle_mae, pred_deg = angle_mae_and_pred_deg(pred_vec, target_heading_deg)
        distance_mae = float((pred_range - target_range).abs().mean().item())
        range_span = max(1e-6, float(target_range.max().item() - target_range.min().item()))
        angle_rel = angle_mae / 180.0
        distance_rel = distance_mae / range_span
        rows.append(
            {
                "layer_id": layer_id,
                "val_pairs": int(target_range.shape[0]),
                "angle_mae_deg": angle_mae,
                "distance_mae": distance_mae,
                "angle_rel_error": angle_rel,
                "distance_rel_error": distance_rel,
                "final_score_proxy": (angle_rel + distance_rel) / 2.0,
                "displacement_epe": displacement_epe(pred_deg, pred_range, target_heading_deg, target_range),
                "readout": "h0_h8_like_axis_separate_conv_grid",
                "num_resconv_block": int(next(iter(probes.values())).heading_branch.res_conv.__len__()),
                "range_scale": range_scale,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    ensure_run_root(run_root)
    layers = parse_layers(args.layers)
    if any(layer_id <= 0 for layer_id in layers):
        raise ValueError("Head-like probe expects decoder-dim layers 1..12; layer 0 is encoder-dim.")

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
    for param in model.parameters():
        param.requires_grad = False
    load_result = load_checkpoint(model, args.checkpoint, device)

    probes = nn.ModuleDict({str(layer_id): AxisHeadProbe(768, num_resconv_block=args.num_resconv_block) for layer_id in layers})
    probes.to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(train_expr, args.batch_size, args.num_workers, shuffle=False)
    val_loader = build_loader(val_expr, args.batch_size, args.num_workers, shuffle=False)

    started = time.time()
    train_logs = []
    for epoch in range(int(args.epochs)):
        probes.train()
        train_logs.append(
            {
                "epoch": epoch + 1,
                **train_epoch(
                    model=model,
                    probes=probes,
                    loader=train_loader,
                    layers=layers,
                    optimizer=optimizer,
                    device=device,
                    max_pairs=args.max_train_pairs,
                    range_scale=args.range_scale,
                    amp=args.amp,
                ),
            }
        )

    probes.eval()
    rows = evaluate(
        model=model,
        probes=probes,
        loader=val_loader,
        layers=layers,
        device=device,
        max_pairs=args.max_val_pairs,
        range_scale=args.range_scale,
        amp=args.amp,
    )
    heading_best = min(rows, key=lambda row: row["angle_mae_deg"])
    range_best = min(rows, key=lambda row: row["distance_mae"])
    proxy_best = min(rows, key=lambda row: row["final_score_proxy"])

    tag = args.tag or f"layers_{'-'.join(str(x) for x in layers)}_train{args.max_train_pairs}_val{args.max_val_pairs}"
    metrics_name = f"layer_head_probe_metrics_{tag}.csv"
    summary_name = f"layer_head_probe_summary_{tag}.json"
    md_name = f"layer_head_probe_summary_{tag}.md"

    summary = {
        "phase": "phase91_g2b_head_like_layer_probe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_root": str(run_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_result": load_result,
        "device": str(device),
        "layers": layers,
        "max_train_pairs": args.max_train_pairs,
        "max_val_pairs": args.max_val_pairs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "range_scale": args.range_scale,
        "num_resconv_block": args.num_resconv_block,
        "train_logs": train_logs,
        "heading_best_layer": heading_best["layer_id"],
        "range_best_layer": range_best["layer_id"],
        "proxy_best_layer": proxy_best["layer_id"],
        "evidence_depth_gap": int(heading_best["layer_id"]) - int(range_best["layer_id"]),
        "elapsed_sec": round(time.time() - started, 3),
        "metrics_csv": str(run_root / "layer_probes" / metrics_name),
        "pass": bool(rows),
    }

    write_csv(run_root / "layer_probes" / metrics_name, rows)
    write_json(run_root / "layer_probes" / summary_name, summary)
    # Also update canonical G2b pointers.
    write_csv(run_root / "layer_probes" / "layer_head_probe_metrics.csv", rows)
    write_json(run_root / "layer_probes" / "layer_head_probe_summary.json", summary)

    md = [
        "# Phase91 G2b H0/H8-Like Layer Head Probe",
        "",
        f"- layers: {layers}",
        f"- max_train_pairs: {args.max_train_pairs}",
        f"- max_val_pairs: {args.max_val_pairs}",
        f"- epochs: {args.epochs}",
        f"- readout: H0/H8-like axis-separate conv-grid head",
        f"- num_resconv_block: {args.num_resconv_block}",
        f"- heading_best_layer: {summary['heading_best_layer']}",
        f"- range_best_layer: {summary['range_best_layer']}",
        f"- proxy_best_layer: {summary['proxy_best_layer']}",
        f"- evidence_depth_gap: {summary['evidence_depth_gap']}",
        f"- elapsed_sec: {summary['elapsed_sec']}",
        "",
        "## Train Logs",
        "",
        "```json",
        json.dumps(train_logs, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Best Rows",
        "",
        "```json",
        json.dumps({"heading_best": heading_best, "range_best": range_best, "proxy_best": proxy_best}, ensure_ascii=False, indent=2),
        "```",
        "",
        "This remains a diagnostic probe. It trains only per-layer readout heads on frozen backbone tokens.",
    ]
    write_text(run_root / "layer_probes" / md_name, "\n".join(md))
    write_text(run_root / "layer_probes" / "layer_head_probe_summary.md", "\n".join(md))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

