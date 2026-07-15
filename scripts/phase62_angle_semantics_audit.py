#!/usr/bin/env python3
"""Audit PairUAV angle-label semantics before Phase62 training.

The audit uses only train/validation labels. It checks simple invariants such
as self-pairs and reverse-pair consistency, then records what label transforms
are or are not justified by the observed data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SURFACE_ROOT = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase56_reloc3r_geometry_consistent_angle_training_v1/"
    "surfaces/phase54_8192_fixed_val811"
)
DEFAULT_TRAIN_JSON_ROOT = DEFAULT_SURFACE_ROOT / "train_json"
DEFAULT_VAL_JSON_ROOT = DEFAULT_SURFACE_ROOT / "val_json"
DEFAULT_OUTPUT_DIR = Path(
    "/media/jgzn/SSD_lexar/RZ/UAVM/runs/"
    "phase62_angle_semantics_partial_unfreeze_v1/semantics"
)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def wrapped_signed_angle_error_deg(pred: float, target: float) -> float:
    return ((pred - target + 180.0) % 360.0) - 180.0


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def iter_json_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def parse_pair_indices(path: Path) -> Tuple[Optional[str], Optional[str]]:
    stem = path.stem
    if "_" not in stem:
        return None, None
    left, right = stem.split("_", 1)
    return left, right


def normalize_sample(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    heading = safe_float(payload.get("heading_num", payload.get("heading_deg")))
    distance = safe_float(payload.get("range_num", payload.get("range_value")))
    if heading is None or distance is None:
        return None

    left, right = parse_pair_indices(path)
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = path
    group_id = path.parent.name
    return {
        "path": str(path),
        "rel_path": str(rel_path).replace("\\", "/"),
        "group_id": group_id,
        "pair_key": path.stem,
        "left": left,
        "right": right,
        "heading": heading,
        "distance": distance,
        "image_a": payload.get("image_a"),
        "image_b": payload.get("image_b"),
    }


def connected_components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> int:
    adjacency: Dict[str, set] = defaultdict(set)
    node_set = set(nodes)
    for a, b in edges:
        node_set.add(a)
        node_set.add(b)
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    count = 0
    for node in node_set:
        if node in seen:
            continue
        count += 1
        queue: deque[str] = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            for nxt in adjacency.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return count


def audit_split(root: Path) -> Dict[str, Any]:
    samples = [s for s in (normalize_sample(p, root) for p in iter_json_paths(root)) if s is not None]
    by_group_pair = {(s["group_id"], s["pair_key"]): s for s in samples}

    reverse_heading_errors: List[float] = []
    reverse_distance_errors: List[float] = []
    reverse_pairs = 0
    self_heading_errors: List[float] = []
    self_distance_errors: List[float] = []
    nodes = set()
    edges = []

    for sample in samples:
        group_id = sample["group_id"]
        left = sample["left"]
        right = sample["right"]
        if left is None or right is None:
            continue
        nodes.add(f"{group_id}/{left}")
        nodes.add(f"{group_id}/{right}")
        edges.append((f"{group_id}/{left}", f"{group_id}/{right}"))

        if left == right:
            self_heading_errors.append(abs(wrapped_signed_angle_error_deg(sample["heading"], 0.0)))
            self_distance_errors.append(abs(sample["distance"]))
            continue

        reverse_key = f"{right}_{left}"
        reverse = by_group_pair.get((group_id, reverse_key))
        if reverse is None:
            continue
        if sample["pair_key"] > reverse_key:
            continue
        reverse_pairs += 1
        expected_reverse_heading = -sample["heading"]
        expected_reverse_distance = -sample["distance"]
        reverse_heading_errors.append(
            abs(wrapped_signed_angle_error_deg(reverse["heading"], expected_reverse_heading))
        )
        reverse_distance_errors.append(abs(reverse["distance"] - expected_reverse_distance))

    headings = [float(s["heading"]) for s in samples]
    distances = [float(s["distance"]) for s in samples]
    components = connected_components(nodes, edges)
    undirected_edges = {tuple(sorted(edge)) for edge in edges}

    return {
        "root": str(root),
        "num_samples": len(samples),
        "num_groups": len({s["group_id"] for s in samples}),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_undirected_edges": len(undirected_edges),
        "num_components": components,
        "cycle_rank_upper_bound": max(0, len(undirected_edges) - len(nodes) + components),
        "heading_min": min(headings) if headings else None,
        "heading_max": max(headings) if headings else None,
        "heading_mean_abs": mean([abs(v) for v in headings]),
        "heading_positive": sum(1 for v in headings if v > 0),
        "heading_negative": sum(1 for v in headings if v < 0),
        "heading_zero": sum(1 for v in headings if v == 0),
        "distance_min": min(distances) if distances else None,
        "distance_max": max(distances) if distances else None,
        "distance_mean_abs": mean([abs(v) for v in distances]),
        "distance_positive": sum(1 for v in distances if v > 0),
        "distance_negative": sum(1 for v in distances if v < 0),
        "distance_zero": sum(1 for v in distances if v == 0),
        "reverse_pair_count": reverse_pairs,
        "reverse_heading_abs_error_mean": mean(reverse_heading_errors),
        "reverse_heading_abs_error_max": max(reverse_heading_errors) if reverse_heading_errors else None,
        "reverse_distance_abs_error_mean": mean(reverse_distance_errors),
        "reverse_distance_abs_error_max": max(reverse_distance_errors) if reverse_distance_errors else None,
        "self_pair_count": len(self_heading_errors),
        "self_heading_abs_error_mean": mean(self_heading_errors),
        "self_heading_abs_error_max": max(self_heading_errors) if self_heading_errors else None,
        "self_distance_abs_error_mean": mean(self_distance_errors),
        "self_distance_abs_error_max": max(self_distance_errors) if self_distance_errors else None,
    }


def build_transform_assessment(train: Dict[str, Any], val: Dict[str, Any]) -> Dict[str, Any]:
    reverse_ok = all(
        split.get("reverse_pair_count", 0) > 0
        and (split.get("reverse_heading_abs_error_max") or 0.0) <= 1e-6
        and (split.get("reverse_distance_abs_error_max") or 0.0) <= 1e-6
        for split in (train, val)
    )
    self_ok = all(
        (split.get("self_pair_count", 0) == 0)
        or (
            (split.get("self_heading_abs_error_max") or 0.0) <= 1e-6
            and (split.get("self_distance_abs_error_max") or 0.0) <= 1e-6
        )
        for split in (train, val)
    )
    return {
        "reverse_pair_negation_supported": reverse_ok,
        "self_pair_zero_supported": self_ok,
        "rotation_or_crop_label_transform_supported": "unknown",
        "flip_label_transform_supported": "unknown",
        "single_image_absolute_heading_supported": "unknown",
        "equivariant_training_transform_status": "not_promoted_without_dataset_coordinate_rule",
        "notes": [
            "Reverse-pair negation can be used for train/val audits if supported above.",
            "Image-space transforms are not treated as legal label transforms without an explicit coordinate convention.",
            "Graph statistics are recorded only for train/val diagnostics and must not be transferred to official test inference.",
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    train = payload["splits"]["train"]
    val = payload["splits"]["val"]
    assessment = payload["transform_assessment"]
    lines = [
        "# Phase62 G1 Angle Semantics Audit",
        "",
        "This audit uses only train/validation labels. It is not an official test-time graph method.",
        "",
        "## Train",
        "",
        f"- samples: {train['num_samples']}",
        f"- groups: {train['num_groups']}",
        f"- reverse_pairs: {train['reverse_pair_count']}",
        f"- reverse_heading_abs_error_max: {train['reverse_heading_abs_error_max']}",
        f"- reverse_distance_abs_error_max: {train['reverse_distance_abs_error_max']}",
        f"- self_pairs: {train['self_pair_count']}",
        f"- graph_nodes: {train['num_nodes']}",
        f"- graph_components: {train['num_components']}",
        f"- cycle_rank_upper_bound: {train['cycle_rank_upper_bound']}",
        "",
        "## Val",
        "",
        f"- samples: {val['num_samples']}",
        f"- groups: {val['num_groups']}",
        f"- reverse_pairs: {val['reverse_pair_count']}",
        f"- reverse_heading_abs_error_max: {val['reverse_heading_abs_error_max']}",
        f"- reverse_distance_abs_error_max: {val['reverse_distance_abs_error_max']}",
        f"- self_pairs: {val['self_pair_count']}",
        f"- graph_nodes: {val['num_nodes']}",
        f"- graph_components: {val['num_components']}",
        f"- cycle_rank_upper_bound: {val['cycle_rank_upper_bound']}",
        "",
        "## Transform Assessment",
        "",
        f"- reverse_pair_negation_supported: {assessment['reverse_pair_negation_supported']}",
        f"- self_pair_zero_supported: {assessment['self_pair_zero_supported']}",
        f"- rotation_or_crop_label_transform_supported: {assessment['rotation_or_crop_label_transform_supported']}",
        f"- flip_label_transform_supported: {assessment['flip_label_transform_supported']}",
        f"- single_image_absolute_heading_supported: {assessment['single_image_absolute_heading_supported']}",
        f"- equivariant_training_transform_status: {assessment['equivariant_training_transform_status']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(train_json_root: Path, val_json_root: Path, output_dir: Path) -> Dict[str, Any]:
    train = audit_split(train_json_root)
    val = audit_split(val_json_root)
    payload = {
        "train_json_root": str(train_json_root),
        "val_json_root": str(val_json_root),
        "splits": {
            "train": train,
            "val": val,
        },
        "transform_assessment": build_transform_assessment(train, val),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "angle_semantics_audit.json"
    report_path = output_dir / "ANGLE_SEMANTICS_AUDIT.md"
    write_json(json_path, payload)
    write_report(report_path, payload)
    return {
        "json_path": str(json_path),
        "report_path": str(report_path),
        "payload": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json-root", type=Path, default=DEFAULT_TRAIN_JSON_ROOT)
    parser.add_argument("--val-json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        train_json_root=args.train_json_root,
        val_json_root=args.val_json_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
