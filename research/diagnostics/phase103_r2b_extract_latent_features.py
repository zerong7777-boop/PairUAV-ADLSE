#!/usr/bin/env python3
"""Extract Phase103-R2b pooled latent features.

This script runs on a Reloc3r environment with checkpoint files. It saves
compact pooled decoder-token features and manifests for CPU-side R2b audits.
It does not read hidden official-test labels or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_EXPR = "Reloc3rRelpose(img_size=512, output_mode='pairuav_range_h0_heading_mid_late_heading_range')"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--phase100-per-sample", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--prediction", action="append", default=[], help="Optional NAME=val_predict_output.txt")
    parser.add_argument("--sample-scope", default="val811")
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-pairs", type=int, default=811)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--numeric-layers", default="6,11,12")
    parser.add_argument("--semantic-layers", default="mid,late")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_specs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH spec: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty name in spec: {item}")
        if name in out:
            raise ValueError(f"Duplicate spec name: {name}")
        out[name] = Path(path)
    return out


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_predictions(path: Path, required_rows: int) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
            if len(rows) >= required_rows:
                break
    if len(rows) < required_rows:
        raise ValueError(f"{path} has {len(rows)} prediction rows; need {required_rows}")
    return rows


def load_checkpoint(model, checkpoint_path: Path, device):
    import gc
    import torch

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
    result = str(model.load_state_dict(new_state, strict=False))
    del payload, state_dict, new_state
    gc.collect()
    return result


def move_batch_to_device(batch, device):
    for view in batch:
        for name in "img camera_intrinsics camera_pose".split():
            if name in view and hasattr(view[name], "to"):
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def decoder_dim_indices(decout) -> list[int]:
    decoder_dim = decout[-1].shape[-1]
    return [idx for idx, tokens in enumerate(decout) if tokens.shape[-1] == decoder_dim]


def semantic_index_map(decout) -> dict[str, int]:
    indices = decoder_dim_indices(decout)
    if not indices:
        raise ValueError("decoder output has no decoder-dim token layers")
    return {
        "early": indices[0],
        "mid": indices[len(indices) // 2],
        "late": indices[-1],
    }


def extract_checkpoint_features(
    model,
    loader,
    device,
    numeric_layers: list[int],
    semantic_layers: list[str],
    max_pairs: int,
    amp: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    import torch

    features: dict[str, list[Any]] = {str(layer): [] for layer in numeric_layers}
    for layer in semantic_layers:
        features[layer] = []
    layer_meta: dict[str, Any] = {}
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
            sem = semantic_index_map(dec2)
            layer_requests: list[tuple[str, int, str]] = []
            for layer in numeric_layers:
                layer_requests.append((str(layer), layer, "numeric"))
            for layer in semantic_layers:
                layer_requests.append((layer, sem[layer], "semantic"))
            for layer_name, decoder_idx, kind in layer_requests:
                pooled = dec2[decoder_idx][:take].detach().float().mean(dim=1).cpu()
                features[layer_name].append(pooled)
                layer_meta[layer_name] = {
                    "layer_kind": kind,
                    "decoder_index": decoder_idx,
                    "token_count": int(dec2[decoder_idx].shape[1]),
                    "feature_dim": int(dec2[decoder_idx].shape[-1]),
                }
            processed += take
            if processed >= max_pairs:
                break
    merged = {name: torch.cat(parts, dim=0) for name, parts in features.items() if parts}
    return merged, layer_meta, processed


def build_sample_manifest(
    phase_rows: list[dict[str, str]],
    predictions: dict[str, list[tuple[float, float]]],
    sample_scope: str,
    processed_pairs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keep_columns = [
        "sample_index",
        "group_id",
        "json_id",
        "fold_id",
        "best_final_case",
        "best_heading_case",
        "best_range_case",
        "heading_range_best_case_mismatch",
        "baseline_final_error",
        "baseline_minus_best_final_error",
        "baseline_minus_axiswise_oracle",
        "pred_heading_deg",
        "pred_range",
        "true_heading_deg",
        "true_range",
        "pred_heading_bin_idx",
        "pred_range_abs_bucket",
        "pred_range_sign",
        "range_span",
    ]
    for idx, row in enumerate(phase_rows[:processed_pairs]):
        out: dict[str, Any] = {"sample_scope": sample_scope, "sample_index": row.get("sample_index", idx)}
        for column in keep_columns:
            if column in row:
                out[column] = row[column]
        if "fold_id" not in out:
            out["fold_id"] = int(idx % 5)
        for case, values in predictions.items():
            pred_h, pred_r = values[idx]
            out[f"pred_heading_{case}"] = pred_h
            out[f"pred_range_{case}"] = pred_r
        rows.append(out)
    return rows


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from reloc3r.datasets import get_data_loader
    from reloc3r.reloc3r_relpose import Reloc3rRelpose

    checkpoint_specs = parse_specs(args.checkpoint)
    prediction_specs = parse_specs(args.prediction)
    numeric_layers = parse_int_list(args.numeric_layers)
    semantic_layers = parse_str_list(args.semantic_layers)
    phase_rows = read_csv(Path(args.phase100_per_sample))
    requested_pairs = min(args.max_pairs, len(phase_rows))
    predictions = {
        name: read_predictions(path, requested_pairs)
        for name, path in prediction_specs.items()
    }

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
    loader = get_data_loader(
        dataset_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )

    feature_rows: list[dict[str, Any]] = []
    load_results: dict[str, str] = {}
    processed_by_case: dict[str, int] = {}
    for case_name, checkpoint_path in checkpoint_specs.items():
        load_results[case_name] = load_checkpoint(model, checkpoint_path, device)
        features, layer_meta, processed = extract_checkpoint_features(
            model,
            loader,
            device,
            numeric_layers,
            semantic_layers,
            requested_pairs,
            args.amp,
        )
        processed_by_case[case_name] = processed
        for layer_name, tensor in features.items():
            meta = layer_meta[layer_name]
            file_name = f"features__{args.sample_scope}__{case_name}__{layer_name}.npz"
            feature_path = output_dir / file_name
            np.savez_compressed(
                feature_path,
                features=tensor.numpy().astype("float32"),
                sample_index=np.arange(int(tensor.shape[0]), dtype="int64"),
            )
            feature_rows.append(
                {
                    "sample_scope": args.sample_scope,
                    "checkpoint_case": case_name,
                    "layer_name": layer_name,
                    "layer_kind": meta["layer_kind"],
                    "decoder_index": meta["decoder_index"],
                    "token_count": meta["token_count"],
                    "feature_dim": meta["feature_dim"],
                    "row_count": int(tensor.shape[0]),
                    "feature_path": str(feature_path),
                    "dtype": "float32",
                    "pooling": "decoder_token_mean",
                }
            )
        torch.cuda.empty_cache()

    processed_pairs = min(processed_by_case.values()) if processed_by_case else 0
    sample_manifest = build_sample_manifest(phase_rows, predictions, args.sample_scope, processed_pairs)
    write_csv(output_dir / "phase103_r2b_feature_manifest.csv", feature_rows)
    write_csv(output_dir / "phase103_r2b_sample_manifest.csv", sample_manifest)
    summary = {
        "phase": "phase103_r2b_extract_latent_features",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "sample_scope": args.sample_scope,
        "json_root": str(args.json_root),
        "image_root": str(args.image_root),
        "phase100_per_sample": str(args.phase100_per_sample),
        "checkpoint_specs": {name: str(path) for name, path in checkpoint_specs.items()},
        "prediction_specs": {name: str(path) for name, path in prediction_specs.items()},
        "load_results": load_results,
        "processed_pairs": processed_pairs,
        "feature_manifest_rows": len(feature_rows),
        "sample_manifest_rows": len(sample_manifest),
        "numeric_layers": numeric_layers,
        "semantic_layers": semantic_layers,
        "device": str(device),
        "amp": args.amp,
        "uses_hidden_test_labels": False,
        "uses_leaderboard_feedback": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "phase103_r2b_feature_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

