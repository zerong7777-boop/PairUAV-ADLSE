#!/usr/bin/env python3
"""Phase95-R0 pose-regime representation geometry audit.

This is a diagnostic-only script. It reads fixed labeled train/val JSON splits
and model checkpoints, extracts decoder token-mean features, and measures
whether heading/range regimes form distinguishable regions in representation
space. It does not read official hidden test labels or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_MODEL_EXPR = "Reloc3rRelpose(img_size=512, output_mode='pairuav_range_h0_heading_mid_late_heading_range')"
RANGE_BUCKETS = [
    ("d_01_le_1", 1.0),
    ("d_02_le_5", 5.0),
    ("d_03_le_10", 10.0),
    ("d_04_le_25", 25.0),
    ("d_05_le_50", 50.0),
    ("d_06_le_100", 100.0),
    ("d_07_gt_100", math.inf),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--layers", default="6,11,12")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> str:
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


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def heading_bin(value: float, bins: int = 8) -> str:
    value = value % 360.0
    width = 360.0 / bins
    idx = min(int(value // width), bins - 1)
    lo = int(idx * width)
    hi = int((idx + 1) * width)
    return f"h_{idx:02d}_{lo}_{hi}"


def range_bucket(value: float) -> str:
    abs_value = abs(float(value))
    for name, limit in RANGE_BUCKETS:
        if abs_value <= limit:
            return name
    return RANGE_BUCKETS[-1][0]


def extract_features(model, loader, device: torch.device, layers: list[int], max_pairs: int, amp: int) -> dict:
    features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    headings = []
    ranges = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            view1, view2 = batch
            with torch.cuda.amp.autocast(enabled=bool(amp)):
                (_shape1, _shape2), (feat1, feat2), (pos1, pos2) = model._encoder(view1, view2)
                _dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
            current = int(dec2[-1].shape[0])
            take = min(current, max_pairs - processed)
            if take <= 0:
                break
            for layer in layers:
                pooled = dec2[layer][:take].detach().float().mean(dim=1).cpu()
                features[layer].append(pooled)
            headings.append(torch.as_tensor(view2["heading_deg"]).float().view(-1)[:take].cpu())
            ranges.append(torch.as_tensor(view2["range_value"]).float().view(-1)[:take].cpu())
            processed += take
            if processed >= max_pairs:
                break
    return {
        "features": {layer: torch.cat(parts, dim=0) for layer, parts in features.items()},
        "heading_deg": torch.cat(headings, dim=0),
        "range": torch.cat(ranges, dim=0),
        "processed": processed,
    }


def pairwise_group_stats(x: torch.Tensor, labels: list[str]) -> dict[str, float]:
    x = F.normalize(x.float(), dim=1, eps=1e-6)
    sim = x @ x.T
    n = sim.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    same = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            same[i, j] = labels[i] == labels[j]
    same = same & ~eye
    diff = (~same) & ~eye
    mean_same = float(sim[same].mean().item()) if same.any() else math.nan
    mean_diff = float(sim[diff].mean().item()) if diff.any() else math.nan

    mean = x.mean(dim=0, keepdim=True)
    total_var = float(((x - mean) ** 2).sum(dim=1).mean().item())
    between = 0.0
    for label in sorted(set(labels)):
        idx = torch.tensor([k for k, item in enumerate(labels) if item == label], dtype=torch.long)
        centroid = x[idx].mean(dim=0, keepdim=True)
        between += len(idx) / n * float(((centroid - mean) ** 2).sum().item())

    nn_sim = sim.clone()
    nn_sim[eye] = -float("inf")
    nn = nn_sim.argmax(dim=1).tolist()
    nn_purity = sum(int(labels[i] == labels[nn[i]]) for i in range(n)) / n
    return {
        "same_cosine": mean_same,
        "diff_cosine": mean_diff,
        "same_minus_diff_cosine": mean_same - mean_diff,
        "same_cosine_distance": 1.0 - mean_same,
        "diff_cosine_distance": 1.0 - mean_diff,
        "between_total_var_ratio": between / max(total_var, 1e-12),
        "nearest_neighbor_same_regime_rate": nn_purity,
    }


def nearest_continuous_stats(x: torch.Tensor, headings: torch.Tensor, ranges: torch.Tensor) -> dict[str, float]:
    x = F.normalize(x.float(), dim=1, eps=1e-6)
    sim = x @ x.T
    n = sim.shape[0]
    sim[torch.eye(n, dtype=torch.bool)] = -float("inf")
    nn = sim.argmax(dim=1).tolist()
    heading_diffs = [
        abs(wrap_angle_diff_deg(float(headings[i]), float(headings[nn[i]]))) for i in range(n)
    ]
    range_diffs = [abs(float(ranges[i]) - float(ranges[nn[i]])) for i in range(n)]
    return {
        "nn_heading_abs_diff_deg": sum(heading_diffs) / n,
        "nn_range_abs_diff": sum(range_diffs) / n,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    started = time.time()

    from reloc3r.datasets import get_data_loader
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    dataset_expr = (
        "PairUAV("
        f"json_root={str(args.json_root)!r}, "
        f"image_root={str(args.image_root)!r}, "
        f"split={args.split!r}, resolution=(512,384), seed={args.seed}, require_labels=True)"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model, {"Reloc3rRelpose": Reloc3rRelpose})
    model.to(device)
    model.eval()
    load_result = load_checkpoint(model, Path(args.checkpoint), device)
    loader = get_data_loader(
        dataset_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )
    data = extract_features(model, loader, device, layers, args.max_pairs, args.amp)
    headings = data["heading_deg"]
    ranges = data["range"]
    heading_bins = [heading_bin(float(v)) for v in headings]
    range_bins = [range_bucket(float(v)) for v in ranges]
    signed_range_bins = [("neg_" if float(v) < 0 else "pos_") + range_bucket(float(v)) for v in ranges]

    rows: list[dict[str, object]] = []
    for layer in layers:
        x = data["features"][layer]
        for regime_name, labels in [
            ("heading_8bin", heading_bins),
            ("range_abs_bucket", range_bins),
            ("range_signed_bucket", signed_range_bins),
        ]:
            row = {
                "layer_id": layer,
                "feature_dim": int(x.shape[1]),
                "pairs": int(x.shape[0]),
                "regime": regime_name,
                "unique_regimes": len(set(labels)),
                **pairwise_group_stats(x, labels),
                **nearest_continuous_stats(x, headings, ranges),
            }
            rows.append(row)

    write_csv(output_dir / "phase95_r0_geometry_rows.csv", rows)
    best_rows = {}
    for regime in sorted({str(row["regime"]) for row in rows}):
        subset = [row for row in rows if row["regime"] == regime]
        best_rows[regime] = max(subset, key=lambda row: float(row["same_minus_diff_cosine"]))

    summary = {
        "phase": "phase95_r0_representation_geometry_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "json_root": str(args.json_root),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_result": load_result,
        "model": args.model,
        "split": args.split,
        "layers": layers,
        "processed_pairs": int(data["processed"]),
        "device": str(device),
        "amp": args.amp,
        "elapsed_sec": round(time.time() - started, 3),
        "best_by_regime": best_rows,
        "interpretation": (
            "Positive same_minus_diff_cosine / between_total_var_ratio indicates pose-regime structure, "
            "not a deployable leaderboard method by itself."
        ),
    }
    (output_dir / "phase95_r0_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Phase95-R0 Representation Geometry Audit",
        "",
        f"- processed_pairs: `{summary['processed_pairs']}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- model: `{args.model}`",
        f"- layers: `{args.layers}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "",
        "## Best Regime Separations",
        "",
        "| regime | layer | same_minus_diff_cosine | between_total_var_ratio | nn_same_regime_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for regime, row in best_rows.items():
        md.append(
            f"| `{regime}` | {row['layer_id']} | {float(row['same_minus_diff_cosine']):.6g} | "
            f"{float(row['between_total_var_ratio']):.6g} | {float(row['nearest_neighbor_same_regime_rate']):.6g} |"
        )
    md.extend(
        [
            "",
            "This is a labeled local validation diagnostic only. It does not use official hidden-test labels.",
        ]
    )
    (output_dir / "phase95_r0_geometry_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "processed_pairs": data["processed"], "best_by_regime": best_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
