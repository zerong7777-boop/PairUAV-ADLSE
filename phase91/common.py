from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_UAVM_ROOT = Path(os.environ.get("UAVM_ROOT", "/media/jgzn/SSD_lexar/RZ/UAVM"))
DEFAULT_REPO_ROOT = DEFAULT_UAVM_ROOT / "external" / "reloc3r_pairuav"
DEFAULT_RUN_ROOT = DEFAULT_UAVM_ROOT / "runs" / "phase91_polarrel_problem_mechanism_validation_v1"
DEFAULT_TRAIN_JSON_ROOT = DEFAULT_UAVM_ROOT / "runs" / "devsplit_v1" / "train_json"
DEFAULT_VAL_JSON_ROOT = DEFAULT_UAVM_ROOT / "runs" / "devsplit_v1" / "val_json"
DEFAULT_IMAGE_ROOT = DEFAULT_UAVM_ROOT / "official" / "UAVM_2026" / "pairUAV" / "train_tour"
DEFAULT_WSTRIP_CHECKPOINT = (
    DEFAULT_UAVM_ROOT
    / "runs"
    / "explore_axisdecouple_reloc3r_head_v1"
    / "checkpoints"
    / "reloc3r512_backbone_only_no_pose_head.pth"
)

RUN_SUBDIRS = (
    "audits",
    "layer_probes",
    "router_smokes",
    "polar_loss_ablations",
    "diagnostics",
    "manifests",
    "reports",
)


def ensure_run_root(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for name in RUN_SUBDIRS:
        (run_root / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_command(args: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "cmd": args,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }


def iter_json_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []

    def sort_key(path: Path) -> tuple[Any, ...]:
        def first_int(text: str) -> int:
            match = re.search(r"\d+", text)
            return int(match.group(0)) if match else 10**12

        return (first_int(path.parent.name), path.parent.name, first_int(path.stem), path.stem)

    return sorted([p for p in root.rglob("*.json") if p.is_file()], key=sort_key)


def load_pair_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    group_id = str(payload.get("group_id", path.parent.name))
    json_id = str(payload.get("json_id", path.stem))
    return {
        "group_id": group_id,
        "json_id": json_id,
        "sample_id": f"{group_id}/{json_id}",
        "json_path": str(path),
        "image_a": str(payload.get("image_a", payload.get("image_a_path", ""))),
        "image_b": str(payload.get("image_b", payload.get("image_b_path", ""))),
        "heading_deg": float(payload.get("heading_deg", payload.get("heading_num", 0.0))),
        "range_value": float(payload.get("range_value", payload.get("range_num", 0.0))),
        "raw_keys": sorted(payload.keys()),
    }


def circular_diff_deg(pred: float, target: float) -> float:
    return ((float(pred) - float(target) + 180.0) % 360.0) - 180.0


def circular_abs_error_deg(pred: float, target: float) -> float:
    return abs(circular_diff_deg(pred, target))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": quantile(finite, 0.5),
        "p95": quantile(finite, 0.95),
        "max": max(finite),
    }


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)

