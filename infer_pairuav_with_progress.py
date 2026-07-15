import argparse
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from reloc3r.datasets.pairuav import PairUAV
from reloc3r.datasets import get_data_loader
from reloc3r.reloc3r_relpose import Reloc3rRelpose


DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def parse_args():
    parser = argparse.ArgumentParser(description="PairUAV Reloc3r inference with progress and ETA.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--resolution", default="(512,384)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
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


def fmt_seconds(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(done, total, elapsed):
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else math.inf
    pct = 100.0 * done / total if total else 0.0
    return {
        "done": done,
        "total": total,
        "pct": round(pct, 4),
        "elapsed_sec": round(elapsed, 3),
        "eta_sec": None if math.isinf(eta) else round(eta, 3),
        "elapsed": fmt_seconds(elapsed),
        "eta": "unknown" if math.isinf(eta) else fmt_seconds(eta),
        "samples_per_sec": round(rate, 3),
    }


def write_zip(result_path):
    zip_path = result_path.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_path, arcname="result.txt")
    return zip_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.txt"
    progress_path = output_dir / "inference_progress.log"
    manifest_path = output_dir / "manifest.json"

    dataset_expr = (
        "PairUAV("
        f"json_root={args.json_root!r}, "
        f"image_root={args.image_root!r}, "
        f"split={args.split!r}, "
        f"resolution={args.resolution}, "
        f"seed={args.seed}, "
        "require_labels=False"
        ")"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model)
    model.to(device)
    model.eval()
    load_result = load_checkpoint(model, args.checkpoint, device)

    loader = get_data_loader(
        dataset_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )
    total_dataset = len(loader.dataset)
    total = total_dataset if args.max_samples <= 0 else min(total_dataset, args.max_samples)
    total_batches = math.ceil(total / args.batch_size)

    manifest = {
        "checkpoint": args.checkpoint,
        "json_root": args.json_root,
        "image_root": args.image_root,
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "device": str(device),
        "model": args.model,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "dataset_expr": dataset_expr,
        "total_dataset_samples": total_dataset,
        "target_samples": total,
        "target_batches": total_batches,
        "load_state_dict": str(load_result),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    done = 0
    started = time.time()
    with result_path.open("w", encoding="utf-8") as result_file, progress_path.open("w", encoding="utf-8") as progress_file:
        progress_file.write(json.dumps({"event": "start", **manifest}, ensure_ascii=False) + "\n")
        progress_file.flush()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                view1, view2 = batch
                for view in batch:
                    for name in "img camera_intrinsics camera_pose".split():
                        if name in view:
                            view[name] = view[name].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    _, pred2 = model(view1, view2)
                pred_heading = pred2["heading_vec"]
                pred_range = pred2["range_value"].view(-1)
                pred_deg = torch.rad2deg(torch.atan2(pred_heading[:, 1], pred_heading[:, 0]))
                pred_deg = pred_deg.detach().cpu().numpy()
                pred_range = pred_range.detach().cpu().numpy()
                remaining = total - done
                take = min(len(pred_deg), remaining)
                if take <= 0:
                    break
                for heading, range_value in zip(pred_deg[:take], pred_range[:take]):
                    result_file.write(f"{float(heading):.6f} {float(range_value):.6f}\n")
                done += take
                if batch_idx == 1 or batch_idx % args.log_every == 0 or done >= total:
                    row = {"event": "progress", "batch": batch_idx, "total_batches": total_batches}
                    row.update(progress_line(done, total, time.time() - started))
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    progress_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    result_file.flush()
                if done >= total:
                    break

        final_row = {"event": "done"}
        final_row.update(progress_line(done, total, time.time() - started))
        progress_file.write(json.dumps(final_row, ensure_ascii=False) + "\n")
        progress_file.flush()
        print(json.dumps(final_row, ensure_ascii=False), flush=True)

    zip_path = None
    if args.zip:
        zip_path = write_zip(result_path)
    manifest.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "written_rows": done,
            "progress_path": str(progress_path),
            "zip_path": str(zip_path) if zip_path else None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
