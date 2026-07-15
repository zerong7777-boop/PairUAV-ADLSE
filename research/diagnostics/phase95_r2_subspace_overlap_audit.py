#!/usr/bin/env python3
"""Phase95-R2 heading/range linear-subspace overlap audit.

This is a diagnostic-only script. It extracts decoder token-mean features, fits
ridge linear probes for heading and range regimes on deterministic half splits,
and compares self-vs-cross subspace alignment. It does not read official hidden
test labels or leaderboard feedback.
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
    parser.add_argument("--split-seeds", default="777,778,779")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--sv-threshold", type=float, default=1e-5)
    parser.add_argument("--include-shuffle", type=int, choices=[0, 1], default=1)
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


def one_hot(labels: list[str]) -> tuple[torch.Tensor, list[str]]:
    classes = sorted(set(labels))
    index = {label: i for i, label in enumerate(classes)}
    y = torch.zeros((len(labels), len(classes)), dtype=torch.float64)
    for row, label in enumerate(labels):
        y[row, index[label]] = 1.0
    return y, classes


def subset_labels(labels: list[str], indices: torch.Tensor) -> list[str]:
    return [labels[int(i)] for i in indices.tolist()]


def shuffled_labels(labels: list[str], seed: int) -> list[str]:
    out = list(labels)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def split_indices(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    mid = n // 2
    return torch.tensor(sorted(idx[:mid]), dtype=torch.long), torch.tensor(sorted(idx[mid:]), dtype=torch.long)


def ridge_probe(
    x: torch.Tensor,
    labels: list[str],
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    ridge: float,
    sv_threshold: float,
) -> dict[str, object]:
    x = x.float().double()
    x_train = x[train_idx]
    x_test = x[test_idx]
    train_labels = subset_labels(labels, train_idx)
    test_labels = subset_labels(labels, test_idx)
    y_train, classes = one_hot(train_labels)
    class_index = {label: i for i, label in enumerate(classes)}

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    z_train = (x_train - mean) / std
    z_test = (x_test - mean) / std
    z_train_aug = torch.cat([z_train, torch.ones((z_train.shape[0], 1), dtype=torch.float64)], dim=1)
    z_test_aug = torch.cat([z_test, torch.ones((z_test.shape[0], 1), dtype=torch.float64)], dim=1)

    xtx = z_train_aug.T @ z_train_aug
    reg = torch.eye(xtx.shape[0], dtype=torch.float64) * ridge
    reg[-1, -1] = 0.0
    xty = z_train_aug.T @ y_train
    weights = torch.linalg.solve(xtx + reg, xty)
    train_scores = z_train_aug @ weights
    test_scores = z_test_aug @ weights
    train_pred = train_scores.argmax(dim=1).tolist()
    test_pred = test_scores.argmax(dim=1).tolist()
    train_true = [class_index[label] for label in train_labels]
    test_true = [class_index.get(label, -1) for label in test_labels]
    test_valid = [i for i, val in enumerate(test_true) if val >= 0]
    train_acc = sum(int(a == b) for a, b in zip(train_pred, train_true)) / max(len(train_true), 1)
    test_acc = (
        sum(int(test_pred[i] == test_true[i]) for i in test_valid) / max(len(test_valid), 1)
        if test_valid
        else math.nan
    )

    feature_weights = weights[:-1, :]
    u, s, _vh = torch.linalg.svd(feature_weights, full_matrices=False)
    if s.numel() == 0 or float(s.max()) <= 0.0:
        basis = torch.empty((x.shape[1], 0), dtype=torch.float64)
        rank = 0
    else:
        keep = s > float(s.max()) * sv_threshold
        basis = u[:, keep]
        rank = int(keep.sum().item())
    return {
        "basis": basis,
        "rank": rank,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "class_count": len(classes),
        "classes": classes,
    }


def subspace_alignment(q1: torch.Tensor, q2: torch.Tensor) -> dict[str, float]:
    if q1.shape[1] == 0 or q2.shape[1] == 0:
        return {
            "rank_a": int(q1.shape[1]),
            "rank_b": int(q2.shape[1]),
            "max_cos": math.nan,
            "mean_sq_cos": math.nan,
            "sum_sq_cos": math.nan,
        }
    s = torch.linalg.svdvals(q1.T @ q2).clamp(0.0, 1.0)
    return {
        "rank_a": int(q1.shape[1]),
        "rank_b": int(q2.shape[1]),
        "max_cos": float(s.max().item()),
        "mean_sq_cos": float((s.square().sum() / max(min(q1.shape[1], q2.shape[1]), 1)).item()),
        "sum_sq_cos": float(s.square().sum().item()),
    }


def avg(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_list(args.layers)
    split_seeds = parse_int_list(args.split_seeds)
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
    base_labels = {
        "heading_8bin": [heading_bin(float(v)) for v in headings],
        "range_abs_bucket": [range_bucket(float(v)) for v in ranges],
        "range_signed_bucket": [
            ("neg_" if float(v) < 0 else "pos_") + range_bucket(float(v)) for v in ranges
        ],
    }

    rows: list[dict[str, object]] = []
    label_modes = [("true", base_labels)]
    if args.include_shuffle:
        label_modes.append(
            (
                "label_shuffle",
                {
                    name: shuffled_labels(labels, args.seed + 1000 + i)
                    for i, (name, labels) in enumerate(base_labels.items())
                },
            )
        )

    for layer in layers:
        x = data["features"][layer]
        n = x.shape[0]
        for control_kind, labels_by_regime in label_modes:
            for split_seed in split_seeds:
                idx_a, idx_b = split_indices(n, split_seed)
                probes = {}
                for regime, labels in labels_by_regime.items():
                    probes[(regime, "a")] = ridge_probe(x, labels, idx_a, idx_b, args.ridge, args.sv_threshold)
                    probes[(regime, "b")] = ridge_probe(x, labels, idx_b, idx_a, args.ridge, args.sv_threshold)

                for range_regime in ["range_abs_bucket", "range_signed_bucket"]:
                    h_a = probes[("heading_8bin", "a")]
                    h_b = probes[("heading_8bin", "b")]
                    r_a = probes[(range_regime, "a")]
                    r_b = probes[(range_regime, "b")]
                    h_self = subspace_alignment(h_a["basis"], h_b["basis"])
                    r_self = subspace_alignment(r_a["basis"], r_b["basis"])
                    cross_alignments = [
                        subspace_alignment(h_a["basis"], r_a["basis"]),
                        subspace_alignment(h_a["basis"], r_b["basis"]),
                        subspace_alignment(h_b["basis"], r_a["basis"]),
                        subspace_alignment(h_b["basis"], r_b["basis"]),
                    ]
                    cross_mean_sq = avg([item["mean_sq_cos"] for item in cross_alignments])
                    cross_max = max(item["max_cos"] for item in cross_alignments)
                    self_mean_sq = avg([h_self["mean_sq_cos"], r_self["mean_sq_cos"]])
                    rows.append(
                        {
                            "case_name": args.case_name,
                            "split": args.split,
                            "control_kind": control_kind,
                            "layer_id": layer,
                            "split_seed": split_seed,
                            "regime_pair": f"heading_8bin__vs__{range_regime}",
                            "pairs": int(n),
                            "heading_class_count": h_a["class_count"],
                            "range_class_count": r_a["class_count"],
                            "heading_rank_a": h_a["rank"],
                            "heading_rank_b": h_b["rank"],
                            "range_rank_a": r_a["rank"],
                            "range_rank_b": r_b["rank"],
                            "heading_train_acc_ab": h_a["train_acc"],
                            "heading_test_acc_ab": h_a["test_acc"],
                            "heading_train_acc_ba": h_b["train_acc"],
                            "heading_test_acc_ba": h_b["test_acc"],
                            "range_train_acc_ab": r_a["train_acc"],
                            "range_test_acc_ab": r_a["test_acc"],
                            "range_train_acc_ba": r_b["train_acc"],
                            "range_test_acc_ba": r_b["test_acc"],
                            "heading_self_mean_sq_cos": h_self["mean_sq_cos"],
                            "heading_self_max_cos": h_self["max_cos"],
                            "range_self_mean_sq_cos": r_self["mean_sq_cos"],
                            "range_self_max_cos": r_self["max_cos"],
                            "cross_mean_sq_cos": cross_mean_sq,
                            "cross_max_cos": cross_max,
                            "self_mean_sq_cos": self_mean_sq,
                            "cross_to_self_ratio": cross_mean_sq / max(self_mean_sq, 1e-12),
                            "non_overlap_score": 1.0 - (cross_mean_sq / max(self_mean_sq, 1e-12)),
                        }
                    )

    write_csv(output_dir / "phase95_r2_subspace_rows.csv", rows)
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["control_kind"]), int(row["layer_id"]), str(row["regime_pair"]))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (control_kind, layer, regime_pair), items in sorted(grouped.items()):
        summary_rows.append(
            {
                "control_kind": control_kind,
                "layer_id": layer,
                "regime_pair": regime_pair,
                "heading_test_acc_mean": avg(
                    [float(item["heading_test_acc_ab"]) for item in items]
                    + [float(item["heading_test_acc_ba"]) for item in items]
                ),
                "range_test_acc_mean": avg(
                    [float(item["range_test_acc_ab"]) for item in items]
                    + [float(item["range_test_acc_ba"]) for item in items]
                ),
                "heading_self_mean_sq_cos": avg([float(item["heading_self_mean_sq_cos"]) for item in items]),
                "range_self_mean_sq_cos": avg([float(item["range_self_mean_sq_cos"]) for item in items]),
                "cross_mean_sq_cos": avg([float(item["cross_mean_sq_cos"]) for item in items]),
                "cross_max_cos": avg([float(item["cross_max_cos"]) for item in items]),
                "cross_to_self_ratio": avg([float(item["cross_to_self_ratio"]) for item in items]),
                "non_overlap_score": avg([float(item["non_overlap_score"]) for item in items]),
            }
        )
    write_csv(output_dir / "phase95_r2_subspace_summary_rows.csv", summary_rows)
    true_rows = [row for row in summary_rows if row["control_kind"] == "true"]
    best_by_pair = {}
    for pair in sorted({str(row["regime_pair"]) for row in true_rows}):
        subset = [row for row in true_rows if row["regime_pair"] == pair]
        best_by_pair[pair] = max(
            subset,
            key=lambda row: (
                float(row["non_overlap_score"]),
                float(row["heading_test_acc_mean"]) + float(row["range_test_acc_mean"]),
            ),
        )

    summary = {
        "phase": "phase95_r2_subspace_overlap_audit",
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
        "split_seeds": split_seeds,
        "ridge": args.ridge,
        "sv_threshold": args.sv_threshold,
        "processed_pairs": int(data["processed"]),
        "device": str(device),
        "amp": args.amp,
        "elapsed_sec": round(time.time() - started, 3),
        "best_true_by_pair": best_by_pair,
        "interpretation": (
            "Low cross_to_self_ratio with strong heading/range probe accuracy supports partially "
            "separable heading and range linear subspaces. This is diagnostic evidence only."
        ),
    }
    (output_dir / "phase95_r2_subspace_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Phase95-R2 Subspace Overlap Audit",
        "",
        f"- case_name: `{args.case_name}`",
        f"- processed_pairs: `{summary['processed_pairs']}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- model: `{args.model}`",
        f"- layers: `{args.layers}`",
        f"- split_seeds: `{args.split_seeds}`",
        f"- elapsed_sec: `{summary['elapsed_sec']}`",
        "",
        "## Best True Subspace Separations",
        "",
        "| regime pair | layer | heading acc | range acc | heading self | range self | cross mean | cross/self | non-overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, row in best_by_pair.items():
        md.append(
            f"| `{pair}` | {row['layer_id']} | "
            f"{float(row['heading_test_acc_mean']):.6g} | "
            f"{float(row['range_test_acc_mean']):.6g} | "
            f"{float(row['heading_self_mean_sq_cos']):.6g} | "
            f"{float(row['range_self_mean_sq_cos']):.6g} | "
            f"{float(row['cross_mean_sq_cos']):.6g} | "
            f"{float(row['cross_to_self_ratio']):.6g} | "
            f"{float(row['non_overlap_score']):.6g} |"
        )
    md.extend(
        [
            "",
            "This is a labeled local validation diagnostic only. It does not use official hidden-test labels.",
        ]
    )
    (output_dir / "phase95_r2_subspace_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "processed_pairs": data["processed"],
                "best_true_by_pair": best_by_pair,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
