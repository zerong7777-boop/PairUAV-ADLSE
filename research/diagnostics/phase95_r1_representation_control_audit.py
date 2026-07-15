#!/usr/bin/env python3
"""Phase95-R1 representation-control audit.

This is a diagnostic-only script. It extends Phase95-R0 by running label-shuffle
and sample-slice controls on fixed local train/val JSON splits. It does not read
official hidden-test labels or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-pairs", type=int, default=811)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--layers", default="6,11,12")
    parser.add_argument("--shuffle-seeds", default="777,778,779")
    parser.add_argument("--sample-sizes", default="256,811")
    parser.add_argument("--sample-seeds", default="777,778,779")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def select_items(values: list[str], indices: torch.Tensor) -> list[str]:
    return [values[int(idx)] for idx in indices.tolist()]


def shuffled(values: list[str], seed: int) -> list[str]:
    result = list(values)
    rng = random.Random(seed)
    rng.shuffle(result)
    return result


def sample_index_sets(n: int, sample_sizes: list[int], sample_seeds: list[int]) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = [("full", torch.arange(n, dtype=torch.long))]
    seen = {"full"}
    for size in sample_sizes:
        if size >= n:
            continue
        prefix_name = f"prefix_{size}"
        if prefix_name not in seen:
            out.append((prefix_name, torch.arange(size, dtype=torch.long)))
            seen.add(prefix_name)
        for seed in sample_seeds:
            name = f"random_{size}_seed{seed}"
            if name in seen:
                continue
            rng = random.Random(seed)
            idx = list(range(n))
            rng.shuffle(idx)
            out.append((name, torch.tensor(sorted(idx[:size]), dtype=torch.long)))
            seen.add(name)
    return out


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


def make_label_modes(base_labels: dict[str, list[str]], shuffle_seeds: list[int]) -> list[dict[str, object]]:
    modes: list[dict[str, object]] = []
    for regime_name, labels in base_labels.items():
        modes.append(
            {
                "regime": regime_name,
                "label_mode": "true",
                "label_seed": "",
                "labels": labels,
                "control_kind": "true",
            }
        )
        for seed in shuffle_seeds:
            modes.append(
                {
                    "regime": regime_name,
                    "label_mode": "label_shuffle",
                    "label_seed": seed,
                    "labels": shuffled(labels, seed),
                    "control_kind": "shuffle",
                }
            )
    return modes


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    full_rows = [row for row in rows if row["sample_mode"] == "full"]
    true_rows = [row for row in full_rows if row["control_kind"] == "true"]
    best_true_by_regime = {}
    control_summary = {}
    for regime in sorted({str(row["regime"]) for row in true_rows}):
        true_subset = [row for row in true_rows if row["regime"] == regime]
        best = max(true_subset, key=lambda row: float(row["same_minus_diff_cosine"]))
        best_true_by_regime[regime] = best
        same_layer_shuffle = [
            row
            for row in full_rows
            if row["regime"] == regime
            and row["control_kind"] == "shuffle"
            and row["layer_id"] == best["layer_id"]
        ]
        shuffle_values = [float(row["same_minus_diff_cosine"]) for row in same_layer_shuffle]
        if shuffle_values:
            mean_shuffle = sum(shuffle_values) / len(shuffle_values)
            max_shuffle = max(shuffle_values)
        else:
            mean_shuffle = math.nan
            max_shuffle = math.nan
        control_summary[regime] = {
            "best_true_layer": best["layer_id"],
            "best_true_same_minus_diff": best["same_minus_diff_cosine"],
            "shuffle_mean_same_minus_diff_same_layer": mean_shuffle,
            "shuffle_max_same_minus_diff_same_layer": max_shuffle,
            "true_minus_shuffle_mean": float(best["same_minus_diff_cosine"]) - mean_shuffle,
            "true_minus_shuffle_max": float(best["same_minus_diff_cosine"]) - max_shuffle,
            "best_true_between_total_var_ratio": best["between_total_var_ratio"],
            "best_true_nn_same_regime_rate": best["nearest_neighbor_same_regime_rate"],
        }
    return {
        "best_true_full_by_regime": best_true_by_regime,
        "shuffle_control_by_regime": control_summary,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_list(args.layers)
    shuffle_seeds = parse_int_list(args.shuffle_seeds)
    sample_sizes = parse_int_list(args.sample_sizes)
    sample_seeds = parse_int_list(args.sample_seeds)
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
    base_labels = {
        "heading_8bin": heading_bins,
        "range_abs_bucket": range_bins,
        "range_signed_bucket": signed_range_bins,
    }
    label_modes = make_label_modes(base_labels, shuffle_seeds)
    samples = sample_index_sets(int(data["processed"]), sample_sizes, sample_seeds)

    rows: list[dict[str, object]] = []
    for sample_mode, indices in samples:
        sample_headings = headings[indices]
        sample_ranges = ranges[indices]
        for layer in layers:
            x = data["features"][layer][indices]
            for mode in label_modes:
                labels = select_items(mode["labels"], indices)
                row = {
                    "case_name": args.case_name,
                    "split": args.split,
                    "sample_mode": sample_mode,
                    "sample_size": int(indices.numel()),
                    "layer_id": layer,
                    "feature_dim": int(x.shape[1]),
                    "regime": mode["regime"],
                    "label_mode": mode["label_mode"],
                    "label_seed": mode["label_seed"],
                    "control_kind": mode["control_kind"],
                    "unique_regimes": len(set(labels)),
                    **pairwise_group_stats(x, labels),
                    **nearest_continuous_stats(x, sample_headings, sample_ranges),
                }
                rows.append(row)

    write_csv(output_dir / "phase95_r1_control_rows.csv", rows)
    summary = {
        "phase": "phase95_r1_representation_control_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_name": args.case_name,
        "output_dir": str(output_dir),
        "json_root": str(args.json_root),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_result": load_result,
        "model": args.model,
        "split": args.split,
        "layers": layers,
        "shuffle_seeds": shuffle_seeds,
        "sample_sizes": sample_sizes,
        "sample_seeds": sample_seeds,
        "processed_pairs": int(data["processed"]),
        "device": str(device),
        "amp": args.amp,
        "elapsed_sec": round(time.time() - started, 3),
        **summarize_rows(rows),
        "interpretation": (
            "A true-vs-shuffle gap supports real pose-regime structure in the representation. "
            "It is diagnostic evidence only and is not a deployable leaderboard method."
        ),
    }
    (output_dir / "phase95_r1_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Phase95-R1 Representation Control Audit",
        "",
        f"- case_name: `{args.case_name}`",
        f"- processed_pairs: `{summary['processed_pairs']}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- model: `{args.model}`",
        f"- layers: `{args.layers}`",
        f"- shuffle_seeds: `{args.shuffle_seeds}`",
        f"- sample_sizes: `{args.sample_sizes}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "",
        "## Full-Sample True vs Shuffle Controls",
        "",
        "| regime | best true layer | true same-minus-diff | shuffle mean | shuffle max | true - shuffle mean | true - shuffle max | true between/total | true NN purity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, row in summary["shuffle_control_by_regime"].items():
        md.append(
            f"| `{regime}` | {row['best_true_layer']} | "
            f"{float(row['best_true_same_minus_diff']):.6g} | "
            f"{float(row['shuffle_mean_same_minus_diff_same_layer']):.6g} | "
            f"{float(row['shuffle_max_same_minus_diff_same_layer']):.6g} | "
            f"{float(row['true_minus_shuffle_mean']):.6g} | "
            f"{float(row['true_minus_shuffle_max']):.6g} | "
            f"{float(row['best_true_between_total_var_ratio']):.6g} | "
            f"{float(row['best_true_nn_same_regime_rate']):.6g} |"
        )
    md.extend(
        [
            "",
            "This is a labeled local validation diagnostic only. It does not use official hidden-test labels.",
        ]
    )
    (output_dir / "phase95_r1_control_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "processed_pairs": data["processed"],
                "shuffle_control_by_regime": summary["shuffle_control_by_regime"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
