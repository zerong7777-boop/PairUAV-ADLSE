#!/usr/bin/env python3
"""Generate validation-image patch-occlusion axis sensitivity maps for PairUAV.

Unlike gradient maps, this figure uses input perturbations. For each selected
validation pair, the script masks one target-view patch at a time and measures
how much the model's heading output and range output change. The result is a
causal-style sensitivity visualization for heading/range evidence heterogeneity.
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


def read_pairs(path: Path, max_cases: int, groups: set[str] | None) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = []
    for row in rows:
        if groups and row.get("case_group") not in groups:
            continue
        pair_id = row.get("pair_id")
        if pair_id:
            pairs.append(pair_id)
        if len(pairs) >= max_cases:
            break
    return pairs


def tensor_to_uint8_image(tensor):
    img = tensor.detach().float().cpu()[0]
    img = (img * 0.5 + 0.5).clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def to_device_batch(views, device):
    import torch

    batched = []
    for view in views:
        out = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.unsqueeze(0).to(device)
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


def clone_views_with_mask(views, iy: int, ix: int, grid: int, mask_mode: str):
    import torch

    cloned = []
    for view_idx, view in enumerate(views):
        out = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.clone()
            else:
                out[key] = value
        if view_idx == 1:
            img = out["img"]
            _, h, w = img.shape
            y0 = round(iy * h / grid)
            y1 = round((iy + 1) * h / grid)
            x0 = round(ix * w / grid)
            x1 = round((ix + 1) * w / grid)
            if mask_mode == "zero":
                fill = torch.zeros((img.shape[0], 1, 1), dtype=img.dtype, device=img.device)
            elif mask_mode == "image_mean":
                fill = img.mean(dim=(1, 2), keepdim=True)
            else:
                raise ValueError(f"unsupported mask_mode: {mask_mode}")
            img[:, y0:y1, x0:x1] = fill
            out["img"] = img
        cloned.append(out)
    return cloned


def find_sample_index(dataset, pair_id: str) -> int:
    for idx, sample in enumerate(dataset.samples):
        current = f"{sample['group_id']}/{sample['json_id']}"
        if current == pair_id:
            return idx
    raise KeyError(f"pair_id {pair_id!r} not found in dataset")


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


def pred_pose(model, views, device):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        batch = to_device_batch(views, device)
        _, pred = model(batch[0], batch[1])
        heading_vec = F.normalize(pred["heading_vec"].float(), dim=-1)
        heading_deg = torch.rad2deg(torch.atan2(heading_vec[:, 1], heading_vec[:, 0]))
        range_value = pred["range_value"].float().view(-1)
    return heading_vec.detach().cpu()[0], float(heading_deg.detach().cpu()[0]), float(range_value.detach().cpu()[0])


def angle_delta_deg(vec_a, vec_b) -> float:
    import torch

    dot = torch.clamp(torch.dot(vec_a.float(), vec_b.float()), -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)))


def normalize_map(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - float(np.nanmin(cam))
    denom = float(np.nanmax(cam))
    if denom > 1e-8:
        cam = cam / denom
    return np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)


def resize_map(cam, target_hw):
    from PIL import Image

    norm = normalize_map(cam)
    image = Image.fromarray((norm * 255.0).round().astype(np.uint8), mode="L")
    image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


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
    return {
        "heatmap_corr": corr,
        "heatmap_cosine": cosine,
        "top20_overlap": float(len(h_top.intersection(r_top)) / k),
        "mean_abs_diff": float(np.mean(np.abs(h - r))),
    }


def compute_occlusion_maps(model, views, device, grid: int, mask_mode: str):
    base_vec, base_heading, base_range = pred_pose(model, views, device)
    heading_grid = np.zeros((grid, grid), dtype=np.float32)
    range_grid = np.zeros((grid, grid), dtype=np.float32)
    for iy in range(grid):
        for ix in range(grid):
            masked = clone_views_with_mask(views, iy, ix, grid, mask_mode)
            vec, _heading_deg, range_value = pred_pose(model, masked, device)
            heading_grid[iy, ix] = angle_delta_deg(base_vec, vec)
            range_grid[iy, ix] = abs(range_value - base_range)
    target_hw = tuple(int(x) for x in views[1]["img"].shape[-2:])
    heading_map = resize_map(heading_grid, target_hw)
    range_map = resize_map(range_grid, target_hw)
    return heading_map, range_map, {
        "baseline_pred_heading_deg": base_heading,
        "baseline_pred_range": base_range,
        "max_heading_delta_deg": float(heading_grid.max()),
        "max_range_delta": float(range_grid.max()),
    }


def colorize_heatmap(cam):
    import matplotlib.cm as cm

    colored = cm.get_cmap("magma")(normalize_map(cam))[..., :3]
    return (colored * 255.0).astype(np.uint8)


def overlay(image, cam, alpha=0.45):
    heat = colorize_heatmap(cam)
    return np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)


def render_figure(cases, output_png: Path, output_svg: Path | None):
    import matplotlib.pyplot as plt

    rows = len(cases)
    fig, axes = plt.subplots(rows, 5, figsize=(13.5, 2.45 * rows), squeeze=False)
    for row_idx, case in enumerate(cases):
        panels = [
            ("source view", case["source_image"]),
            ("target view", case["target_image"]),
            ("heading sensitivity", overlay(case["target_image"], case["heading_map"])),
            ("range sensitivity", overlay(case["target_image"], case["range_map"])),
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
        "Validation patch-occlusion sensitivity: heading and range react to different regions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
    parser.add_argument("--case-groups", default="axis_conflict", help="Comma-separated case groups to include, or empty for all.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resolution", default="(512,384)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--mask-mode", choices=["image_mean", "zero"], default="image_mean")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-png", type=Path, default=Path("figures/fig_val_patch_occlusion_axis_sensitivity.png"))
    parser.add_argument("--output-svg", type=Path, default=Path("figures/fig_val_patch_occlusion_axis_sensitivity.svg"))
    parser.add_argument("--summary-csv", type=Path, default=Path("figures/val_patch_occlusion_axis_sensitivity_summary.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("figures/val_patch_occlusion_axis_sensitivity_summary.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))

    import torch
    from reloc3r.datasets.pairuav import PairUAV
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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

    groups = {g.strip() for g in args.case_groups.split(",") if g.strip()} if args.case_groups else None
    pair_ids = read_pairs(args.case_manifest, args.max_cases, groups)
    cases = []
    summary_rows = []
    for pair_id in pair_ids:
        idx = find_sample_index(dataset, pair_id)
        views = dataset[idx]
        heading_map, range_map, meta = compute_occlusion_maps(model, views, device, args.grid, args.mask_mode)
        source_img = tensor_to_uint8_image(views[0]["img"].unsqueeze(0))
        target_img = tensor_to_uint8_image(views[1]["img"].unsqueeze(0))
        stats = heatmap_stats(heading_map, range_map)
        sample = dataset.samples[idx]
        row = {
            "pair_id": pair_id,
            "target_heading_deg": float(sample["heading_deg"]),
            "target_range": float(sample["range_value"]),
            "grid": args.grid,
            "mask_mode": args.mask_mode,
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

    render_figure(cases, args.output_png, args.output_svg)

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
