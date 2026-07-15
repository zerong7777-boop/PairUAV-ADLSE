#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.phase64_token_angle_model import Phase64TokenAngleSpecialist, phase64_angle_loss, wrap_deg_tensor
from reloc3r.phase64_token_shards import Phase64TokenShardDataset


def angle_abs_error(pred, target):
    return torch.abs(torch.remainder(pred - target + 180.0, 360.0) - 180.0)


class TorchPhase64Dataset(Dataset):
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
        "residual_target",
    ]
    batch = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "match_path": [sample["match_path"] for sample in samples],
    }
    for key in keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return batch


def split_indices(n, val_fraction, seed):
    indices = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_count = max(1, int(round(n * float(val_fraction))))
    val = sorted(indices[:val_count])
    train = sorted(indices[val_count:])
    return train, val


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    angle_errors = []
    rank1_errors = []
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_model(model, batch)
        loss = phase64_angle_loss(outputs, batch["residual_target"], batch["fallback_used"])
        err = angle_abs_error(outputs["corrected_heading"], batch["target_heading"])
        angle_errors.extend([float(x) for x in err.detach().cpu().reshape(-1)])
        rank1_errors.extend([float(x) for x in batch["rank1_angle_abs_error"].detach().cpu().reshape(-1)])
        losses.append(float(loss.detach().cpu()))
        corrected = outputs["corrected_heading"].detach().cpu().reshape(-1).tolist()
        residual = outputs["residual"].detach().cpu().reshape(-1).tolist()
        gate = outputs["gate"].detach().cpu().reshape(-1).tolist()
        target = batch["target_heading"].detach().cpu().reshape(-1).tolist()
        rank1 = batch["rank1_heading"].detach().cpu().reshape(-1).tolist()
        rank1_dist = batch["rank1_distance"].detach().cpu().reshape(-1).tolist()
        for idx, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target_heading": target[idx],
                    "rank1_heading": rank1[idx],
                    "rank1_distance": rank1_dist[idx],
                    "corrected_heading": corrected[idx],
                    "residual": residual[idx],
                    "gate": gate[idx],
                    "corrected_angle_abs_error": angle_errors[-len(batch["sample_id"]) + idx],
                    "rank1_angle_abs_error": rank1_errors[-len(batch["sample_id"]) + idx],
                }
            )
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "angle_mae": float(np.mean(angle_errors)) if angle_errors else None,
        "rank1_angle_mae": float(np.mean(rank1_errors)) if rank1_errors else None,
        "delta_vs_rank1": float(np.mean(angle_errors) - np.mean(rank1_errors)) if angle_errors else None,
        "rows": len(angle_errors),
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
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-residual-deg", type=float, default=0.30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=64064)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-shards", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = TorchPhase64Dataset(args.train_manifest, preload=bool(args.preload_shards))
    if args.val_manifest:
        train_ds = train_base
        val_ds = TorchPhase64Dataset(args.val_manifest, preload=bool(args.preload_shards))
    else:
        train_idx, val_idx = split_indices(len(train_base), args.val_fraction, args.seed)
        train_ds = Subset(train_base, train_idx)
        val_ds = Subset(train_base, val_idx)

    first = train_base[0]
    model = Phase64TokenAngleSpecialist(
        token_dim=first["tokens"].shape[-1],
        hypothesis_dim=first["hypothesis_features"].shape[-1],
        global_dim=first["global_stats"].shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
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
            loss = phase64_angle_loss(outputs, batch["residual_target"], batch["fallback_used"])
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
        "train_manifest": str(Path(args.train_manifest)),
        "val_manifest": str(Path(args.val_manifest)) if args.val_manifest else None,
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
