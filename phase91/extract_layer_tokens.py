from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .common import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_VAL_JSON_ROOT,
    DEFAULT_WSTRIP_CHECKPOINT,
    ensure_run_root,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_MODEL_EXPR = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase91 G2 token extraction smoke.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WSTRIP_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> str:
    if not checkpoint_path or not checkpoint_path.exists():
        return f"checkpoint_missing_or_not_used: {checkpoint_path}"
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
        # Match the existing inference path: metadata such as target_group_index
        # stays on CPU unless a specific head moves it internally.
        for name in "img camera_intrinsics camera_pose".split():
            if name in view and hasattr(view[name], "to"):
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def tensor_stats(layer_id: int, side: str, tensor: torch.Tensor) -> dict:
    finite = torch.isfinite(tensor)
    finite_ratio = float(finite.float().mean().detach().cpu().item()) if tensor.numel() else 0.0
    data = tensor.detach().float()
    return {
        "side": side,
        "layer_id": layer_id,
        "shape": "x".join(str(x) for x in tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_ratio": finite_ratio,
        "mean": float(data.mean().cpu().item()) if tensor.numel() else math.nan,
        "std": float(data.std(unbiased=False).cpu().item()) if tensor.numel() else math.nan,
        "abs_max": float(data.abs().max().cpu().item()) if tensor.numel() else math.nan,
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    ensure_run_root(run_root)

    from reloc3r.datasets import get_data_loader
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    dataset_expr = (
        "PairUAV("
        f"json_root={str(args.json_root)!r}, "
        f"image_root={str(args.image_root)!r}, "
        "split='dev', resolution=(512,384), seed=777, require_labels=True)"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(args.model, {"Reloc3rRelpose": Reloc3rRelpose})
    model.to(device)
    model.eval()
    load_result = load_checkpoint(model, args.checkpoint, device)

    loader = get_data_loader(
        dataset_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )

    started = time.time()
    processed = 0
    layer_accumulator: dict[tuple[str, int], dict] = {}
    first_batch_shapes = []
    normal_path_ok = None
    normal_path_error = ""

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            view1, view2 = batch
            if normal_path_ok is None:
                try:
                    with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                        _ = model(view1, view2)
                    normal_path_ok = True
                except Exception as exc:
                    normal_path_ok = False
                    normal_path_error = repr(exc)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                (shape1, shape2), (feat1, feat2), (pos1, pos2) = model._encoder(view1, view2)
                dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
            dec1 = [tok.float() for tok in dec1]
            dec2 = [tok.float() for tok in dec2]
            current_batch = int(dec2[-1].shape[0])
            take = min(current_batch, args.max_pairs - processed)
            for side, decout in (("view1", dec1), ("view2", dec2)):
                for layer_id, tokens in enumerate(decout):
                    sliced = tokens[:take]
                    row = tensor_stats(layer_id, side, sliced)
                    key = (side, layer_id)
                    existing = layer_accumulator.setdefault(
                        key,
                        {
                            "side": side,
                            "layer_id": layer_id,
                            "shape": row["shape"],
                            "dtype": row["dtype"],
                            "batches": 0,
                            "sample_count": 0,
                            "finite_ratio_min": 1.0,
                            "abs_max_max": 0.0,
                        },
                    )
                    existing["batches"] += 1
                    existing["sample_count"] += take
                    existing["finite_ratio_min"] = min(existing["finite_ratio_min"], row["finite_ratio"])
                    existing["abs_max_max"] = max(existing["abs_max_max"], row["abs_max"])
                    if not first_batch_shapes:
                        pass
                    if processed == 0:
                        first_batch_shapes.append(row)
            processed += take
            if processed >= args.max_pairs:
                break

    summary_rows = [layer_accumulator[key] for key in sorted(layer_accumulator)]
    smoke = {
        "phase": "phase91_g2_token_extraction_smoke",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_root": str(run_root),
        "json_root": str(args.json_root),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_result": load_result,
        "model": args.model,
        "device": str(device),
        "max_pairs": args.max_pairs,
        "processed_pairs": processed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "elapsed_sec": round(time.time() - started, 3),
        "normal_forward_path_ok": normal_path_ok,
        "normal_forward_path_error": normal_path_error,
        "normal_path_modified": False,
        "layer_count_per_side": {
            side: len([row for row in summary_rows if row["side"] == side]) for side in ("view1", "view2")
        },
        "pass": processed > 0
        and bool(normal_path_ok)
        and all(float(row["finite_ratio_min"]) >= 1.0 for row in summary_rows),
    }

    write_json(run_root / "layer_probes" / "token_extraction_smoke.json", smoke)
    write_csv(run_root / "layer_probes" / "token_shape_inventory.csv", summary_rows)
    write_csv(run_root / "layer_probes" / "token_first_batch_shapes.csv", first_batch_shapes)
    md = [
        "# Phase91 G2 Token Extraction Smoke",
        "",
        f"- processed_pairs: {processed}",
        f"- device: `{device}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint_load_result: `{load_result}`",
        f"- normal_forward_path_ok: {normal_path_ok}",
        f"- normal_path_modified: False",
        f"- pass: {smoke['pass']}",
        f"- elapsed_sec: {smoke['elapsed_sec']}",
        "",
        "This smoke calls `_encoder` and `_decoder` directly and does not modify the official forward or inference path.",
    ]
    if normal_path_error:
        md.extend(["", "## Normal Forward Error", "", "```text", normal_path_error, "```"])
    write_text(run_root / "layer_probes" / "token_extraction_smoke.md", "\n".join(md))
    print(json.dumps({"pass": smoke["pass"], "processed_pairs": processed, "elapsed_sec": smoke["elapsed_sec"]}, ensure_ascii=False))
    return 0 if smoke["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
