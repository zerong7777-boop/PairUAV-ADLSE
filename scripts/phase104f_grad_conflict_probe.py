#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_pairuav import build_dataset, load_checkpoint
from reloc3r.loss import (
    PairUAVHeadingOnlyMetricAwareLoss,
    PairUAVHeadingTeacherCacheLoss,
    PairUAVRangeOnlyMetricAwareLoss,
)
from reloc3r.reloc3r_relpose import Reloc3rRelpose
from reloc3r.trainable_policy import apply_trainable_policy


def _move_batch(batch, device):
    for view in batch:
        for name in "img camera_intrinsics camera_pose".split():
            if name in view:
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def _flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def _cosine(a, b):
    a = a.float()
    b = b.float()
    denom = a.norm() * b.norm()
    if float(denom) <= 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _mean(values):
    clean = [float(v) for v in values if v == v]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reloc3rRelpose(img_size=512, output_mode=args.output_mode).to(device)
    load_checkpoint(model, args.checkpoint, device)
    if args.trainable_policy:
        summary = apply_trainable_policy(model, args.trainable_policy)
        print(json.dumps({k: v for k, v in summary.items() if k != "trainable_names"}, indent=2))
    model.eval()

    loader = build_dataset(args.dataset, args.batch_size, args.num_workers, test=False)
    heading_loss_fn = PairUAVHeadingOnlyMetricAwareLoss()
    range_loss_fn = PairUAVRangeOnlyMetricAwareLoss()
    kd_loss_fn = (
        PairUAVHeadingTeacherCacheLoss(args.teacher_csv, heading_distill_weight=args.heading_distill_weight)
        if args.teacher_csv
        else None
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected for gradient probe")

    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        view1, view2 = _move_batch(batch, device)
        pose1, pose2 = model(view1, view2)
        heading_loss, _ = heading_loss_fn(view1, view2, pose1, pose2)
        range_loss, _ = range_loss_fn(view1, view2, pose1, pose2)
        grad_heading = _flat_grad(heading_loss, params)
        grad_range = _flat_grad(range_loss, params)
        row = {
            "batch": batch_idx,
            "cos_grad_heading_range": _cosine(grad_heading, grad_range),
            "grad_norm_heading": float(grad_heading.float().norm()),
            "grad_norm_range": float(grad_range.float().norm()),
        }
        if kd_loss_fn is not None:
            kd_total, _ = kd_loss_fn(view1, view2, pose1, pose2)
            kd_component = kd_total - heading_loss
            grad_kd = _flat_grad(kd_component, params)
            row.update(
                {
                    "cos_grad_heading_gt_kd": _cosine(grad_heading, grad_kd),
                    "grad_norm_heading_gt": float(grad_heading.float().norm()),
                    "grad_norm_heading_kd": float(grad_kd.float().norm()),
                }
            )
        rows.append(row)

    summary = {
        "batches": len(rows),
        "mean_cos_grad_heading_range": _mean([row["cos_grad_heading_range"] for row in rows]),
        "mean_grad_norm_heading": _mean([row["grad_norm_heading"] for row in rows]),
        "mean_grad_norm_range": _mean([row["grad_norm_range"] for row in rows]),
        "rows": rows,
    }
    if rows and "cos_grad_heading_gt_kd" in rows[0]:
        summary["mean_cos_grad_heading_gt_kd"] = _mean([row["cos_grad_heading_gt_kd"] for row in rows])
        summary["mean_grad_norm_heading_kd"] = _mean([row["grad_norm_heading_kd"] for row in rows])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-mode", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trainable-policy", default="paaer_heading_only")
    parser.add_argument("--teacher-csv", default="")
    parser.add_argument("--heading-distill-weight", type=float, default=0.05)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
