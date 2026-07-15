"""A-v3.2c route-v2 checkpoint/model contract and repeatability check.

This script is additive. It does not modify existing A-v3.2c runners.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OFFICIAL_ROOT = Path("/media/jgzn/SSD_lexar/RZ/UAVM/official/UAVM_2026")
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

from baseline.reloc3r_route_v2 import (  # noqa: E402
    Reloc3rRouteModelV2,
    denormalize_range_symmetric,
    load_rgb_tensor,
)
from transformers import AutoFeatureExtractor  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_image(image_root: Path, relative: str) -> Path:
    rel = Path(relative)
    candidates = [image_root / rel, image_root / rel.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for {relative}; tried {candidates}")


class FixedManifestRouteV2Dataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path, model_path: Path) -> None:
        self.rows = rows
        self.image_root = image_root
        self.processor = AutoFeatureExtractor.from_pretrained(str(model_path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        source = row.get("source_image_path") or row.get("source_image_key")
        target = row.get("target_image_path") or row.get("target_image_key")
        image_a = load_rgb_tensor(resolve_image(self.image_root, source), self.processor)
        image_b = load_rgb_tensor(resolve_image(self.image_root, target), self.processor)
        return image_a, image_b, row.get("canonical_pair_id", "")


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(isinstance(k, str) and k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_route_v2_model(checkpoint: Path, model_path: Path, device: torch.device, hidden_dim: int, dropout: float):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise TypeError("Expected checkpoint payload with model_state_dict")
    model = Reloc3rRouteModelV2(model_path, hidden_dim=hidden_dim, dropout=dropout)
    result = model.load_state_dict(normalize_state_dict(payload["model_state_dict"]), strict=True)
    model.to(device)
    model.eval()
    diagnostics = {
        "payload_keys": sorted(payload.keys()),
        "model_variant": payload.get("model_variant", ""),
        "epoch": payload.get("epoch", ""),
        "next_epoch": payload.get("next_epoch", ""),
        "missing_key_count": len(result.missing_keys),
        "unexpected_key_count": len(result.unexpected_keys),
        "missing_key_sample": list(result.missing_keys[:20]),
        "unexpected_key_sample": list(result.unexpected_keys[:20]),
    }
    return model, diagnostics


def run_forward(model, rows: list[dict[str, str]], image_root: Path, model_path: Path, device: torch.device, batch_size: int) -> list[dict[str, str]]:
    dataset = FixedManifestRouteV2Dataset(rows, image_root, model_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[dict[str, str]] = []
    with torch.inference_mode():
        for image_a, image_b, cids in loader:
            image_a = image_a.to(device, non_blocking=True)
            image_b = image_b.to(device, non_blocking=True)
            pred = model(image_a, image_b)
            pred_xy = F.normalize(pred[:, :2], dim=-1)
            pred_deg = torch.rad2deg(torch.atan2(pred_xy[:, 1], pred_xy[:, 0]))
            pred_range = denormalize_range_symmetric(pred[:, 2])
            for cid, deg, rng in zip(cids, pred_deg.detach().cpu().tolist(), pred_range.detach().cpu().tolist()):
                outputs.append({
                    "canonical_pair_id": str(cid),
                    "prediction_heading": f"{float(deg):.12g}",
                    "prediction_range": f"{float(rng):.12g}",
                    "row_status": "ok",
                })
    return outputs


def safe_float(value: str):
    try:
        if value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except ValueError:
        return None


def angle_delta(a: str, b: str):
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return None
    return abs((af - bf + 180.0) % 360.0 - 180.0)


def abs_delta(a: str, b: str):
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return None
    return abs(af - bf)


def summarize(values: list[float | None]) -> dict[str, float | None]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {"mean": None, "max": None}
    return {"mean": sum(clean) / len(clean), "max": clean[-1]}


def compare(reference: list[dict[str, str]], candidate: list[dict[str, str]], name: str) -> dict[str, Any]:
    ref_by_id = {r["canonical_pair_id"]: r for r in reference}
    heading = []
    rng = []
    same = 0
    for row in candidate:
        ref = ref_by_id[row["canonical_pair_id"]]
        hd = angle_delta(ref["prediction_heading"], row["prediction_heading"])
        rd = abs_delta(ref["prediction_range"], row["prediction_range"])
        heading.append(hd)
        rng.append(rd)
        if hd == 0 and rd == 0:
            same += 1
    hs = summarize(heading)
    rs = summarize(rng)
    return {
        "comparison": name,
        "row_count": len(candidate),
        "same_prediction_count": same,
        "same_prediction_fraction": same / len(candidate) if candidate else 0.0,
        "heading_delta_mean": hs["mean"],
        "heading_delta_max": hs["max"],
        "range_delta_mean": rs["mean"],
        "range_delta_max": rs["max"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.manifest)[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_outputs = []
    diagnostics = None
    for idx in range(args.repeats):
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        model, diag = load_route_v2_model(args.checkpoint, args.model_path, device, args.hidden_dim, args.dropout)
        diagnostics = diag
        outputs = run_forward(model, rows, args.image_root, args.model_path, device, args.batch_size)
        all_outputs.append(outputs)
        write_csv(args.output_dir / "tables" / f"route_v2_repeat_{idx}.csv", outputs, [
            "canonical_pair_id",
            "prediction_heading",
            "prediction_range",
            "row_status",
        ])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparisons = [compare(all_outputs[0], all_outputs[idx], f"repeat_0_vs_repeat_{idx}") for idx in range(1, len(all_outputs))]
    min_same = min((c["same_prediction_fraction"] for c in comparisons), default=1.0)
    max_heading = max((c["heading_delta_max"] or 0.0 for c in comparisons), default=0.0)
    max_range = max((c["range_delta_max"] or 0.0 for c in comparisons), default=0.0)
    if diagnostics and diagnostics["missing_key_count"] == 0 and diagnostics["unexpected_key_count"] == 0 and min_same == 1.0 and max_heading == 0.0 and max_range == 0.0:
        verdict = "route-v2-contract-repeatability-pass"
        reason = "strict_checkpoint_load_and_repeated_forward_identical"
    elif diagnostics and diagnostics["missing_key_count"] == 0 and diagnostics["unexpected_key_count"] == 0:
        verdict = "route-v2-contract-load-pass-repeatability-fail"
        reason = "strict_checkpoint_load_passed_but_repeated_forward_changed"
    else:
        verdict = "route-v2-contract-load-fail"
        reason = "strict_checkpoint_load_failed_or_reported_key_mismatch"
    metrics = {
        "verdict": verdict,
        "reason": reason,
        "device": str(device),
        "limit": args.limit,
        "repeats": args.repeats,
        "checkpoint": str(args.checkpoint),
        "model_path": str(args.model_path),
        "load_diagnostics": diagnostics,
        "comparisons": comparisons,
        "min_same_prediction_fraction": min_same,
        "max_heading_delta": max_heading,
        "max_range_delta": max_range,
    }
    write_json(args.output_dir / "metrics" / "route_v2_contract_repeatability_metrics.json", metrics)
    write_csv(args.output_dir / "tables" / "route_v2_repeatability_summary.csv", [
        {
            k: ("" if v is None else f"{v:.12g}" if isinstance(v, float) else v)
            for k, v in item.items()
        }
        for item in comparisons
    ], [
        "comparison",
        "row_count",
        "same_prediction_count",
        "same_prediction_fraction",
        "heading_delta_mean",
        "heading_delta_max",
        "range_delta_mean",
        "range_delta_max",
    ])
    lines = [
        "# A-v3.2c Route-v2 Checkpoint/Model Contract Check",
        "",
        f"verdict: `{verdict}`",
        f"reason: `{reason}`",
        "",
        f"- missing_key_count: {diagnostics.get('missing_key_count') if diagnostics else ''}",
        f"- unexpected_key_count: {diagnostics.get('unexpected_key_count') if diagnostics else ''}",
        f"- model_variant: `{diagnostics.get('model_variant') if diagnostics else ''}`",
        f"- min_same_prediction_fraction: {min_same:.6f}",
        f"- max_heading_delta: {max_heading:.12g}",
        f"- max_range_delta: {max_range:.12g}",
        "",
        "No existing runner/model code was modified.",
    ]
    (args.output_dir / "reports" / "route_v2_contract_repeatability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "reports" / "route_v2_contract_repeatability_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
