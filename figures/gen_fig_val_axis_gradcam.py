#!/usr/bin/env python3
"""Generate validation-image axis saliency maps for PairUAV.

This script computes task-gradient activation maps for a PairUAV checkpoint.
It is intended for mechanism visualization: compare where heading and range
gradients flow on the target view for the same validation pair.

The default target is H8/mid-late, whose heading readout uses mid/late decoder
layers while range stays on the H0 late readout. For this head, the script hooks
the spatial readout feature maps for heading and range separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_range_h0_heading_mid_late_heading_range')"


def parse_resolution(text: str) -> tuple[int, int]:
    text = text.strip().strip("()")
    a, b = text.split(",")
    return int(a), int(b)


def read_pairs(path: Path, max_cases: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [row["pair_id"] for row in rows if row.get("pair_id")]
    return pairs[:max_cases]


def to_device_batch(views, device):
    import torch

    batched = []
    for view in views:
        out = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                tensor = value.unsqueeze(0).to(device)
                out[key] = tensor
            elif isinstance(value, np.ndarray):
                out[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            elif isinstance(value, (np.floating, float)):
                out[key] = torch.tensor([float(value)], dtype=torch.float32, device=device)
            elif isinstance(value, (np.integer, int)):
                out[key] = torch.tensor([int(value)], dtype=torch.long, device=device)
            else:
                out[key] = [value]
        batched.append(out)
    return batched


def tensor_to_uint8_image(tensor):
    img = tensor.detach().float().cpu()[0]
    img = (img * 0.5 + 0.5).clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def find_sample_index(dataset, pair_id: str) -> int:
    for idx, sample in enumerate(dataset.samples):
        current = f"{sample['group_id']}/{sample['json_id']}"
        if current == pair_id:
            return idx
    raise KeyError(f"pair_id {pair_id!r} not found in dataset")


class ActivationCapture:
    def __init__(self, modules):
        self.activations = []
        self.handles = [module.register_forward_hook(self._hook) for module in modules]

    def _hook(self, _module, _inputs, output):
        if hasattr(output, "retain_grad"):
            output.retain_grad()
        self.activations.append(output)

    def close(self):
        for handle in self.handles:
            handle.remove()


def normalize_map(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - float(np.nanmin(cam))
    denom = float(np.nanmax(cam))
    if denom > 1e-8:
        cam = cam / denom
    return np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)


def cam_from_activations(activations, target_hw):
    import torch
    import torch.nn.functional as F

    cams = []
    for act in activations:
        grad = act.grad
        if grad is None:
            continue
        weights = grad.detach().abs().mean(dim=(2, 3), keepdim=True)
        cam = (weights * act.detach().abs()).sum(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)[0, 0]
        cams.append(cam)
    if not cams:
        return np.zeros(target_hw, dtype=np.float32)
    merged = torch.stack(cams, dim=0).mean(dim=0)
    return normalize_map(merged.float().cpu().numpy())


def heatmap_stats(heading_map, range_map):
    h = normalize_map(heading_map).reshape(-1)
    r = normalize_map(range_map).reshape(-1)
    if float(h.std()) < 1e-8 or float(r.std()) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(h, r)[0, 1])
    cosine = float(np.dot(h, r) / ((np.linalg.norm(h) * np.linalg.norm(r)) + 1e-8))
    k = max(1, int(0.20 * len(h)))
    h_top = set(np.argpartition(h, -k)[-k:].tolist())
    r_top = set(np.argpartition(r, -k)[-k:].tolist())
    top20_overlap = float(len(h_top.intersection(r_top)) / k)
    diff_mean = float(np.mean(np.abs(h - r)))
    return {
        "heatmap_corr": corr,
        "heatmap_cosine": cosine,
        "top20_overlap": top20_overlap,
        "mean_abs_diff": diff_mean,
    }


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


def axis_modules(model):
    head = model.pose_head
    if hasattr(head, "heading_layer_res_convs"):
        heading_modules = [blocks[-1] for blocks in head.heading_layer_res_convs]
        range_modules = [head.res_conv[-1]]
        return heading_modules, range_modules
    if hasattr(head, "heading_res_conv"):
        return [head.heading_res_conv[-1]], [head.res_conv[-1]]
    if hasattr(head, "heading_res_convs"):
        return [head.heading_res_convs[-1]], [head.range_res_convs[-1]]
    # Shared-head fallback: useful as a negative control, but not an axis split.
    return [head.res_conv[-1]], [head.res_conv[-1]]


def compute_axis_maps(model, views, device, gradient_target: str):
    import torch
    import torch.nn.functional as F

    heading_modules, range_modules = axis_modules(model)
    target_view = views[1]
    target_hw = tuple(int(x) for x in target_view["img"].shape[-2:])

    def run_one(axis: str):
        model.zero_grad(set_to_none=True)
        batch = to_device_batch(views, device)
        for view in batch:
            view["img"] = view["img"].detach().clone().requires_grad_(True)
        modules = heading_modules if axis == "heading" else range_modules
        capture = ActivationCapture(modules)
        try:
            _, pred = model(batch[0], batch[1])
            target_heading = torch.stack(
                [
                    torch.cos(torch.deg2rad(batch[1]["heading_deg"].float())),
                    torch.sin(torch.deg2rad(batch[1]["heading_deg"].float())),
                ],
                dim=-1,
            )
            target_range = batch[1]["range_value"].float().view(-1, 1)
            heading_loss = F.smooth_l1_loss(pred["heading_vec"], target_heading)
            range_loss = F.smooth_l1_loss(pred["range_value"].view(-1, 1), target_range)
            if gradient_target == "loss":
                objective = heading_loss if axis == "heading" else range_loss
            elif gradient_target == "target_aligned_output":
                if axis == "heading":
                    objective = (pred["heading_vec"] * target_heading).sum(dim=-1).mean()
                else:
                    range_sign = torch.where(
                        target_range.abs() > 1e-6,
                        target_range.sign(),
                        torch.ones_like(target_range),
                    )
                    objective = (pred["range_value"].view(-1, 1) * range_sign).mean()
            else:
                raise ValueError(f"unsupported gradient_target: {gradient_target}")
            objective.backward()
            cam = cam_from_activations(capture.activations, target_hw)
            pred_heading_deg = torch.rad2deg(torch.atan2(pred["heading_vec"][:, 1], pred["heading_vec"][:, 0]))
            meta = {
                "pred_heading_deg": float(pred_heading_deg.detach().cpu()[0]),
                "pred_range": float(pred["range_value"].detach().cpu().view(-1)[0]),
                "heading_loss": float(heading_loss.detach().cpu()),
                "range_loss": float(range_loss.detach().cpu()),
                "objective": float(objective.detach().cpu()),
            }
            return cam, meta
        finally:
            capture.close()

    heading_map, heading_meta = run_one("heading")
    range_map, range_meta = run_one("range")
    meta = {
        "pred_heading_deg": heading_meta["pred_heading_deg"],
        "pred_range": heading_meta["pred_range"],
        "heading_loss": heading_meta["heading_loss"],
        "range_loss": heading_meta["range_loss"],
        "heading_objective": heading_meta["objective"],
        "range_objective": range_meta["objective"],
        "gradient_target": gradient_target,
    }
    return heading_map, range_map, meta


def colorize_heatmap(cam):
    import matplotlib.cm as cm

    colored = cm.get_cmap("magma")(normalize_map(cam))[..., :3]
    return (colored * 255.0).astype(np.uint8)


def overlay(image, cam, alpha=0.45):
    heat = colorize_heatmap(cam)
    return np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)


def render_figure(cases, output_png: Path, output_svg: Path | None, gradient_target: str):
    import matplotlib.pyplot as plt

    rows = len(cases)
    fig, axes = plt.subplots(rows, 5, figsize=(13.5, 2.45 * rows), squeeze=False)
    if gradient_target == "loss":
        heading_title = "heading loss grad"
        range_title = "range loss grad"
        figure_title = "Validation loss-gradient maps: heading and range use different image evidence"
    else:
        heading_title = "heading output grad"
        range_title = "range output grad"
        figure_title = "Validation target-aligned output gradients: heading and range use different image evidence"
    for row_idx, case in enumerate(cases):
        panels = [
            ("source view", case["source_image"]),
            ("target view", case["target_image"]),
            (heading_title, overlay(case["target_image"], case["heading_map"])),
            (range_title, overlay(case["target_image"], case["range_map"])),
            ("|heading-range|", colorize_heatmap(np.abs(case["heading_map"] - case["range_map"]))),
        ]
        for col_idx, (title, img) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if col_idx == 0:
                stats = case["stats"]
                ylabel = (
                    f"{case['pair_id']}\n"
                    f"corr={stats['heatmap_corr']:.2f}, "
                    f"top20={stats['top20_overlap']:.2f}"
                )
                ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(
        figure_title,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, default=Path("figures/val_qualitative_cases.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resolution", default="(512,384)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument(
        "--gradient-target",
        choices=["target_aligned_output", "loss"],
        default="target_aligned_output",
        help="Backprop target-aligned task output by default; use loss for error-sensitivity maps.",
    )
    parser.add_argument("--output-png", type=Path, default=Path("figures/fig_val_axis_gradcam.png"))
    parser.add_argument("--output-svg", type=Path, default=Path("figures/fig_val_axis_gradcam.svg"))
    parser.add_argument("--summary-csv", type=Path, default=Path("figures/val_axis_gradcam_summary.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("figures/val_axis_gradcam_summary.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))

    import torch
    from reloc3r.datasets.pairuav import PairUAV
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairUAV(
        json_root=str(args.json_root),
        image_root=str(args.image_root),
        split="val",
        resolution=parse_resolution(args.resolution),
        seed=args.seed,
        require_labels=True,
    )
    model = eval(args.model)
    load_result = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    pair_ids = read_pairs(args.case_manifest, args.max_cases)
    cases = []
    summary_rows = []
    for pair_id in pair_ids:
        idx = find_sample_index(dataset, pair_id)
        views = dataset[idx]
        heading_map, range_map, meta = compute_axis_maps(model, views, device, args.gradient_target)
        source_img = tensor_to_uint8_image(views[0]["img"].unsqueeze(0))
        target_img = tensor_to_uint8_image(views[1]["img"].unsqueeze(0))
        stats = heatmap_stats(heading_map, range_map)
        sample = dataset.samples[idx]
        row = {
            "pair_id": pair_id,
            "target_heading_deg": float(sample["heading_deg"]),
            "target_range": float(sample["range_value"]),
            **meta,
            **stats,
        }
        cases.append(
            {
                "pair_id": pair_id,
                "source_image": source_img,
                "target_image": target_img,
                "heading_map": heading_map,
                "range_map": range_map,
                "stats": stats,
            }
        )
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    render_figure(cases, args.output_png, args.output_svg, args.gradient_target)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    args.summary_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "model": args.model,
                "json_root": str(args.json_root),
                "image_root": str(args.image_root),
                "load_state_dict": str(load_result),
                "cases": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_png}")
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
