#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reloc3r.datasets import get_data_loader  # noqa: E402
from reloc3r.datasets.pairuav import PairUAV  # noqa: E402
from reloc3r.reloc3r_relpose import Reloc3rRelpose  # noqa: E402,F401


DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"
DEFAULT_METHOD = "m5_reverse_disagreement_gated_avg_p90"
DEFAULT_THRESHOLD = 0.11338043199998538
DEFAULT_CHECKPOINT_SHA256 = "45d7f1d403ff3e2c823667ddfcb900775bfdb4a73afc8ad7c1f7d482aef4ae54"
SAME_FORWARD_METHODS = {"m4_same_forward_average", "m5_reverse_disagreement_gated_avg_p90"}
TRUE_SWAP_METHODS = {"m6_true_swap_average", "m7_true_swap_disagreement_gated_avg_p90"}


def extract_int(value: Any) -> int | float:
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group())
    return float("inf")


def json_path_sort_key(path: Path) -> tuple[int | float, str, int | float, str]:
    path = Path(path)
    return (extract_int(path.parent.name), path.parent.name, extract_int(path.stem), path.stem)


def iter_json_paths_fast(root: Path) -> list[Path]:
    return sorted([p for p in Path(root).rglob("*.json") if p.is_file()], key=json_path_sort_key)


def select_first_json_paths_fast(root: Path, limit: int) -> list[Path]:
    selected: list[Path] = []
    root = Path(root)
    group_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: (extract_int(p.name), p.name))
    for group_dir in group_dirs:
        json_paths = sorted(group_dir.glob("*.json"), key=lambda p: (extract_int(p.stem), p.stem))
        for json_path in json_paths:
            selected.append(json_path)
            if len(selected) >= int(limit):
                return selected
    return selected


def materialize_max_samples_json_root(source_root: Path, output_root: Path, limit: int) -> dict[str, Any]:
    if int(limit) <= 0:
        raise ValueError("limit must be positive when materializing a smoke JSON root")
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.rglob("*.json")):
        raise FileExistsError(f"materialized JSON root already contains JSON files: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = select_first_json_paths_fast(source_root, int(limit))
    for src in selected:
        rel = src.relative_to(source_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return {
        "source_json_root": str(source_root),
        "materialized_json_root": str(output_root),
        "requested_limit": int(limit),
        "written": len(selected),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_resolution(value: str) -> tuple[int, int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise argparse.ArgumentTypeError(f"resolution must be a pair, got {value!r}")
    return int(parsed[0]), int(parsed[1])


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def wrap_angle_deg_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def heading_deg_from_vec(heading_vec: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.atan2(heading_vec[:, 1], heading_vec[:, 0]))


def circular_mean_deg_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar = torch.deg2rad(a)
    br = torch.deg2rad(b)
    return torch.rad2deg(torch.atan2(torch.sin(ar) + torch.sin(br), torch.cos(ar) + torch.cos(br)))


def apply_forward_heading_correction(
    rank1_heading: torch.Tensor,
    candidate_forward_heading: torch.Tensor,
    *,
    method: str,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    disagreement = torch.abs(wrap_angle_deg_tensor(rank1_heading - candidate_forward_heading))
    avg_heading = circular_mean_deg_tensor(rank1_heading, candidate_forward_heading)
    if method in {"m4_same_forward_average", "m6_true_swap_average"}:
        use_average = torch.ones_like(disagreement, dtype=torch.bool)
    elif method in {"m5_reverse_disagreement_gated_avg_p90", "m7_true_swap_disagreement_gated_avg_p90"}:
        use_average = disagreement <= float(threshold)
    else:
        raise ValueError(f"unsupported method: {method}")
    final_heading = torch.where(use_average, avg_heading, rank1_heading)
    final_heading = wrap_angle_deg_tensor(final_heading)
    return final_heading, disagreement, avg_heading, use_average


def apply_heading_correction(
    rank1_heading: torch.Tensor,
    reverse_heading: torch.Tensor,
    *,
    method: str,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reverse_forward_heading = wrap_angle_deg_tensor(-reverse_heading)
    final_heading, disagreement, avg_heading, use_average = apply_forward_heading_correction(
        rank1_heading,
        reverse_forward_heading,
        method=method,
        threshold=threshold,
    )
    return final_heading, reverse_forward_heading, disagreement, avg_heading, use_average


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Any:
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


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(done: int, total: int, elapsed: float) -> dict[str, Any]:
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


def write_zip(result_path: Path) -> Path:
    zip_path = result_path.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_path, arcname="result.txt")
    return zip_path


def build_loader(args: argparse.Namespace, json_root: Path) -> Any:
    dataset = PairUAV(
        json_root=json_root,
        image_root=args.image_root,
        split=args.split,
        resolution=args.resolution,
        seed=args.seed,
        require_labels=False,
    )
    return get_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )


def open_diagnostics(path_text: str) -> tuple[Any | None, csv.DictWriter | None]:
    if not path_text:
        return None, None
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sample_id",
            "rank1_heading",
            "reverse_heading",
            "reverse_forward_heading",
            "same_forward_heading_disagreement",
            "same_forward_avg_heading",
            "final_heading",
            "rank1_distance",
            "used_average",
            "method",
        ],
    )
    writer.writeheader()
    return handle, writer


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.txt"
    progress_path = output_dir / "inference_progress.log"
    manifest_path = output_dir / "manifest.json"

    checkpoint = Path(args.checkpoint)
    actual_sha = sha256_file(checkpoint)
    if args.expected_checkpoint_sha256 and actual_sha != args.expected_checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {actual_sha} != {args.expected_checkpoint_sha256}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model)
    model.to(device)
    model.eval()
    load_result = load_checkpoint(model, checkpoint, device)

    effective_json_root = Path(args.json_root)
    materialize_report = None
    if args.materialized_json_root and args.max_samples > 0:
        materialize_report = materialize_max_samples_json_root(
            Path(args.json_root),
            Path(args.materialized_json_root),
            args.max_samples,
        )
        effective_json_root = Path(args.materialized_json_root)

    loader = build_loader(args, effective_json_root)
    total_dataset = len(loader.dataset)
    total = total_dataset if args.max_samples <= 0 else min(total_dataset, args.max_samples)
    total_batches = math.ceil(total / args.batch_size)

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "json_root": str(args.json_root),
        "effective_json_root": str(effective_json_root),
        "materialize_report": materialize_report,
        "image_root": str(args.image_root),
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "device": str(device),
        "model": args.model,
        "method": args.method,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "split": args.split,
        "resolution": args.resolution,
        "seed": args.seed,
        "total_dataset_samples": total_dataset,
        "target_samples": total,
        "target_batches": total_batches,
        "load_state_dict": str(load_result),
        "diagnostics_output": args.diagnostics_output,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    diagnostics_handle, diagnostics_writer = open_diagnostics(args.diagnostics_output)
    done = 0
    used_average_count = 0
    min_disagreement = math.inf
    max_disagreement = 0.0
    sum_disagreement = 0.0

    with result_path.open("w", encoding="utf-8") as result_file, progress_path.open("w", encoding="utf-8") as progress_file:
        progress_file.write(json.dumps({"event": "start", **manifest}, ensure_ascii=False) + "\n")
        progress_file.flush()
        for batch_idx, batch in enumerate(loader, start=1):
            view1, view2 = batch
            for view in batch:
                for name in "img camera_intrinsics camera_pose".split():
                    if name in view:
                        view[name] = view[name].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                pred1, pred2 = model(view1, view2)

            rank1_heading = heading_deg_from_vec(pred2["heading_vec"])
            pred_range = pred2["range_value"].view(-1)
            secondary_raw_heading = heading_deg_from_vec(pred1["heading_vec"])
            secondary_forward_distance = -pred1["range_value"].view(-1)
            correction_source = "same_forward_pose1"
            if args.method in SAME_FORWARD_METHODS:
                final_heading, secondary_forward_heading, disagreement, avg_heading, use_average = apply_heading_correction(
                    rank1_heading,
                    secondary_raw_heading,
                    method=args.method,
                    threshold=args.threshold,
                )
            elif args.method in TRUE_SWAP_METHODS:
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    _, pred2_swapped = model(view2, view1)
                secondary_raw_heading = heading_deg_from_vec(pred2_swapped["heading_vec"])
                secondary_forward_heading = wrap_angle_deg_tensor(-secondary_raw_heading)
                secondary_forward_distance = -pred2_swapped["range_value"].view(-1)
                correction_source = "true_swap_pred2_inverse"
                final_heading, disagreement, avg_heading, use_average = apply_forward_heading_correction(
                    rank1_heading,
                    secondary_forward_heading,
                    method=args.method,
                    threshold=args.threshold,
                )
            else:
                raise ValueError(f"unsupported method: {args.method}")

            remaining = total - done
            take = min(int(final_heading.shape[0]), remaining)
            if take <= 0:
                break

            final_np = final_heading[:take].detach().cpu().numpy()
            range_np = pred_range[:take].detach().cpu().numpy()
            disagreement_np = disagreement[:take].detach().cpu().numpy()
            use_avg_np = use_average[:take].detach().cpu().numpy()

            for heading, range_value in zip(final_np, range_np):
                result_file.write(f"{float(heading):.6f} {float(range_value):.6f}\n")

            if diagnostics_writer is not None:
                sample_ids = view2.get("sample_id", [""] * take)
                sample_ids = as_string_list(sample_ids)
                rank1_np = rank1_heading[:take].detach().cpu().numpy()
                secondary_raw_np = secondary_raw_heading[:take].detach().cpu().numpy()
                secondary_forward_np = secondary_forward_heading[:take].detach().cpu().numpy()
                avg_np = avg_heading[:take].detach().cpu().numpy()
                for i in range(take):
                    diagnostics_writer.writerow(
                        {
                            "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                            "rank1_heading": f"{float(rank1_np[i]):.9f}",
                            "reverse_heading": f"{float(secondary_raw_np[i]):.9f}",
                            "reverse_forward_heading": f"{float(secondary_forward_np[i]):.9f}",
                            "same_forward_heading_disagreement": f"{float(disagreement_np[i]):.9f}",
                            "same_forward_avg_heading": f"{float(avg_np[i]):.9f}",
                            "final_heading": f"{float(final_np[i]):.9f}",
                            "rank1_distance": f"{float(range_np[i]):.9f}",
                            "used_average": int(bool(use_avg_np[i])),
                            "method": f"{args.method}:{correction_source}",
                        }
                    )
                diagnostics_handle.flush()

            used_average_count += int(use_avg_np.sum())
            min_disagreement = min(min_disagreement, float(disagreement_np.min()))
            max_disagreement = max(max_disagreement, float(disagreement_np.max()))
            sum_disagreement += float(disagreement_np.sum())
            done += take

            if batch_idx == 1 or batch_idx % args.log_every == 0 or done >= total:
                row = {
                    "event": "progress",
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "used_average": used_average_count,
                    "used_average_rate": round(used_average_count / done, 6) if done else 0.0,
                }
                row.update(progress_line(done, total, time.time() - started))
                print(json.dumps(row, ensure_ascii=False), flush=True)
                progress_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress_file.flush()
                result_file.flush()

            if done >= total:
                break

        final_row = {
            "event": "done",
            "used_average": used_average_count,
            "used_average_rate": round(used_average_count / done, 6) if done else 0.0,
            "min_disagreement": None if done == 0 else min_disagreement,
            "max_disagreement": None if done == 0 else max_disagreement,
            "mean_disagreement": None if done == 0 else sum_disagreement / done,
        }
        final_row.update(progress_line(done, total, time.time() - started))
        progress_file.write(json.dumps(final_row, ensure_ascii=False) + "\n")
        progress_file.flush()
        print(json.dumps(final_row, ensure_ascii=False), flush=True)

    if diagnostics_handle is not None:
        diagnostics_handle.close()

    zip_path = write_zip(result_path) if args.zip else None
    manifest.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "written_rows": done,
            "progress_path": str(progress_path),
            "zip_path": str(zip_path) if zip_path else None,
            "used_average": used_average_count,
            "used_average_rate": used_average_count / done if done else 0.0,
            "min_disagreement": None if done == 0 else min_disagreement,
            "max_disagreement": None if done == 0 else max_disagreement,
            "mean_disagreement": None if done == 0 else sum_disagreement / done,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase57 PairUAV official inference with same-forward/reverse p90-gated angle correction.")
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--method",
        choices=[
            "m4_same_forward_average",
            "m5_reverse_disagreement_gated_avg_p90",
            "m6_true_swap_average",
            "m7_true_swap_disagreement_gated_avg_p90",
        ],
        default=DEFAULT_METHOD,
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--split", default="test")
    parser.add_argument("--resolution", type=parse_resolution, default=(512, 384))
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--materialized-json-root", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--diagnostics-output", default="")
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_inference(args)
    print(json.dumps({"status": "pass", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
