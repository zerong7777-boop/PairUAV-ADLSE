#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.phase64_token_shards import Phase64TokenShardDataset
from reloc3r.phase65_matching_angle_model import (
    Phase65MatchingAngleBranch,
    angle_abs_error_deg,
    phase65_angle_loss,
)


class TorchPhase65Dataset(Dataset):
    def __init__(self, manifest_path, preload=False):
        self.dataset = Phase64TokenShardDataset(manifest_path, to_torch=True, preload=preload)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def collate(samples):
    keys = [
        "target_heading",
        "target_distance",
        "rank1_heading",
        "rank1_distance",
        "rank1_angle_abs_error",
        "tokens",
        "token_mask",
        "hypothesis_features",
        "global_stats",
        "fallback_used",
        "valid_matches",
        "residual_target",
    ]
    batch = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "match_path": [sample["match_path"] for sample in samples],
    }
    for key in keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return batch


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_model(model, batch):
    return model(
        tokens=batch["tokens"],
        token_mask=batch["token_mask"],
        hypothesis_features=batch["hypothesis_features"],
        global_stats=batch["global_stats"],
        rank1_heading=batch["rank1_heading"],
        rank1_distance=batch["rank1_distance"],
    )


def load_indices(split_json, overfit_rows=None):
    if overfit_rows is not None and int(overfit_rows) > 0:
        indices = list(range(int(overfit_rows)))
        return indices, indices, {"mode": "overfit", "rows": int(overfit_rows)}
    if not split_json:
        raise ValueError("--split-json is required unless --overfit-rows is set")
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    return list(split["train_indices"]), list(split["val_indices"]), split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    angle_errors = []
    rank1_errors = []
    losses = []
    hard_records = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_model(model, batch)
        loss = phase65_angle_loss(outputs, batch["target_heading"])
        err = angle_abs_error_deg(outputs["corrected_heading"], batch["target_heading"])
        rank1_err = batch["rank1_angle_abs_error"].view(-1)
        angle_errors.extend([float(x) for x in err.detach().cpu().reshape(-1)])
        rank1_errors.extend([float(x) for x in rank1_err.detach().cpu().reshape(-1)])
        losses.append(float(loss.detach().cpu()))

        corrected = outputs["corrected_heading"].detach().cpu().reshape(-1).tolist()
        direct = outputs["direct_heading"].detach().cpu().reshape(-1).tolist()
        residual = outputs["residual"].detach().cpu().reshape(-1).tolist()
        target = batch["target_heading"].detach().cpu().reshape(-1).tolist()
        rank1 = batch["rank1_heading"].detach().cpu().reshape(-1).tolist()
        rank1_dist = batch["rank1_distance"].detach().cpu().reshape(-1).tolist()
        valid_matches = batch["valid_matches"].detach().cpu().reshape(-1).tolist()
        weights = outputs["candidate_weights"].detach().cpu().tolist()
        for idx, sample_id in enumerate(batch["sample_id"]):
            row = {
                "sample_id": sample_id,
                "target_heading": target[idx],
                "rank1_heading": rank1[idx],
                "rank1_distance": rank1_dist[idx],
                "corrected_heading": corrected[idx],
                "direct_heading": direct[idx],
                "residual": residual[idx],
                "rank1_angle_abs_error": float(rank1_err[idx].detach().cpu()),
                "corrected_angle_abs_error": float(err[idx].detach().cpu()),
                "valid_matches": int(valid_matches[idx]),
                "candidate_weight_rank1": float(weights[idx][0]),
                "candidate_weight_direct": float(weights[idx][1]),
            }
            rows.append(row)
            hard_records.append((row["rank1_angle_abs_error"], row["corrected_angle_abs_error"]))
    rank1_mean = float(np.mean(rank1_errors)) if rank1_errors else None
    corrected_mean = float(np.mean(angle_errors)) if angle_errors else None
    hard_gain = None
    if hard_records:
        threshold = np.quantile([item[0] for item in hard_records], 0.75)
        hard = [item for item in hard_records if item[0] >= threshold]
        if hard:
            hard_rank1 = float(np.mean([item[0] for item in hard]))
            hard_corrected = float(np.mean([item[1] for item in hard]))
            hard_gain = {
                "threshold": float(threshold),
                "rows": len(hard),
                "rank1_angle_mae": hard_rank1,
                "corrected_angle_mae": hard_corrected,
                "delta_vs_rank1": hard_corrected - hard_rank1,
                "relative_gain": (hard_rank1 - hard_corrected) / hard_rank1 if hard_rank1 > 0 else None,
            }
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "angle_mae": corrected_mean,
        "rank1_angle_mae": rank1_mean,
        "delta_vs_rank1": corrected_mean - rank1_mean if corrected_mean is not None else None,
        "relative_gain": (rank1_mean - corrected_mean) / rank1_mean if rank1_mean and rank1_mean > 0 else None,
        "rows": len(angle_errors),
        "hard_top25": hard_gain,
        "pred_rows": rows,
    }


def write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-residual-candidates", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-residual-deg", type=float, default=5.0)
    parser.add_argument("--candidate-min-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=65065)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-shards", type=int, default=1)
    parser.add_argument("--overfit-rows", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ds = TorchPhase65Dataset(args.manifest, preload=bool(args.preload_shards))
    train_indices, val_indices, split_report = load_indices(args.split_json, args.overfit_rows if args.overfit_rows > 0 else None)
    train_ds = Subset(base_ds, train_indices)
    val_ds = Subset(base_ds, val_indices)
    first = base_ds[0]
    model = Phase65MatchingAngleBranch(
        token_dim=first["tokens"].shape[-1],
        hypothesis_dim=first["hypothesis_features"].shape[-1],
        global_dim=first["global_stats"].shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_residual_candidates=args.num_residual_candidates,
        dropout=args.dropout,
        max_residual_deg=args.max_residual_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    history = []
    best = None
    for epoch in range(int(args.epochs)):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(model, batch)
            loss = phase65_angle_loss(
                outputs,
                batch["target_heading"],
                candidate_min_weight=args.candidate_min_weight,
                entropy_weight=args.entropy_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": val["loss"],
            "val_angle_mae": val["angle_mae"],
            "val_rank1_angle_mae": val["rank1_angle_mae"],
            "val_delta_vs_rank1": val["delta_vs_rank1"],
            "val_relative_gain": val["relative_gain"],
            "val_hard_top25": val["hard_top25"],
            "val_rows": val["rows"],
        }
        history.append(record)
        if best is None or record["val_angle_mae"] < best["val_angle_mae"]:
            best = dict(record)
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, output_dir / "checkpoint-best.pt")
        print(json.dumps(record, ensure_ascii=False))

    final_eval = evaluate(model, val_loader, device)
    write_predictions(output_dir / "val_predictions.csv", final_eval["pred_rows"])
    summary = {
        "status": "pass",
        "device": str(device),
        "manifest": str(Path(args.manifest)),
        "split_json": str(Path(args.split_json)) if args.split_json else None,
        "split_report": {k: split_report[k] for k in split_report if k not in ("train_indices", "val_indices")},
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "history": history,
        "best": best,
        "final": {key: value for key, value in final_eval.items() if key != "pred_rows"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
