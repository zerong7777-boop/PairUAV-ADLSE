#!/usr/bin/env python3
"""Phase98-R3 checkpoint trajectory driver.

Runs the Phase95-R1/R2 representation audits over a checkpoint trajectory and
summarizes how pose-regime geometry changes with validation performance.

This script is diagnostic-only. It reads fixed local train/val labels and never
reads official hidden-test labels or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_EXPR = (
    "Reloc3rRelpose(img_size=512, "
    "output_mode='pairuav_range_h0_heading_mid_late_heading_range')"
)

DEFAULT_CASES = [
    ("step050000", 50000),
    ("step100000", 100000),
    ("step150000", 150000),
    ("step200000", 200000),
    ("step250000", 250000),
    ("step300000", 300000),
    ("step350000", 350000),
    ("step400000", 400000),
    ("step450000", 450000),
    ("final", 459999),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model", default=DEFAULT_MODEL_EXPR)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-pairs", type=int, default=811)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--layers", default="6,11,12")
    parser.add_argument("--shuffle-seeds", default="777,778,779")
    parser.add_argument("--split-seeds", default="777,778,779")
    parser.add_argument("--sample-sizes", default="256,811")
    parser.add_argument("--sample-seeds", default="777,778,779")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--case-list", default=",".join(case for case, _ in DEFAULT_CASES))
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--run-r1", type=int, choices=[0, 1], default=1)
    parser.add_argument("--run-r2", type=int, choices=[0, 1], default=1)
    parser.add_argument("--skip-existing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-checkpoints", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_cases(case_list: str) -> list[tuple[str, int]]:
    wanted = [item.strip() for item in case_list.split(",") if item.strip()]
    known = {case: step for case, step in DEFAULT_CASES}
    cases: list[tuple[str, int]] = []
    for item in wanted:
        if item in known:
            cases.append((item, known[item]))
            continue
        if item.isdigit():
            step = int(item)
            cases.append((f"step{step:06d}", step))
            continue
        if item.startswith("step") and item[4:].isdigit():
            cases.append((item, int(item[4:])))
            continue
        raise ValueError(f"Unknown case-list item: {item!r}")
    return cases


def checkpoint_path(checkpoint_dir: Path, case_id: str) -> Path:
    if case_id == "final":
        candidates = [
            checkpoint_dir / "checkpoint-final.pth",
            checkpoint_dir / "checkpoint-final.model-only.pth",
        ]
    else:
        candidates = [
            checkpoint_dir / f"checkpoint-{case_id}.model-only.pth",
            checkpoint_dir / f"checkpoint-{case_id}.pth",
        ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_metrics(metrics_csv: str) -> dict[str, dict[str, str]]:
    if not metrics_csv:
        return {}
    path = Path(metrics_csv)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def run_command(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.write(f"CWD: {cwd}\n")
        log.write(f"START: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"END: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log.write(f"ELAPSED_SEC: {time.time() - started:.3f}\n")
        log.write(f"RETURN_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with rc={proc.returncode}: {' '.join(cmd)}")


def maybe_run_r1(args: argparse.Namespace, repo_root: Path, case_dir: Path, case_name: str, ckpt: Path) -> None:
    out = case_dir / "r1_controls"
    summary = out / "phase95_r1_control_summary.json"
    if args.skip_existing and summary.exists():
        return
    cmd = [
        args.python_bin,
        "phase95_r1_representation_control_audit.py",
        "--output-dir",
        str(out),
        "--case-name",
        f"{case_name}_r1",
        "--json-root",
        args.json_root,
        "--image-root",
        args.image_root,
        "--checkpoint",
        str(ckpt),
        "--model",
        args.model,
        "--split",
        args.split,
        "--max-pairs",
        str(args.max_pairs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--layers",
        args.layers,
        "--shuffle-seeds",
        args.shuffle_seeds,
        "--sample-sizes",
        args.sample_sizes,
        "--sample-seeds",
        args.sample_seeds,
        "--amp",
        str(args.amp),
    ]
    run_command(cmd, case_dir / "r1_controls.log", repo_root)


def maybe_run_r2(args: argparse.Namespace, repo_root: Path, case_dir: Path, case_name: str, ckpt: Path) -> None:
    out = case_dir / "r2_subspace"
    summary = out / "phase95_r2_subspace_summary.json"
    if args.skip_existing and summary.exists():
        return
    cmd = [
        args.python_bin,
        "phase95_r2_subspace_overlap_audit.py",
        "--output-dir",
        str(out),
        "--case-name",
        f"{case_name}_r2",
        "--json-root",
        args.json_root,
        "--image-root",
        args.image_root,
        "--checkpoint",
        str(ckpt),
        "--model",
        args.model,
        "--split",
        args.split,
        "--max-pairs",
        str(args.max_pairs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--layers",
        args.layers,
        "--split-seeds",
        args.split_seeds,
        "--include-shuffle",
        "1",
        "--amp",
        str(args.amp),
    ]
    run_command(cmd, case_dir / "r2_subspace.log", repo_root)


def float_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def add_r1_fields(row: dict[str, Any], case_dir: Path) -> None:
    path = case_dir / "r1_controls" / "phase95_r1_control_summary.json"
    if not path.exists():
        return
    summary = read_json(path)
    controls = summary.get("shuffle_control_by_regime", {})
    for regime in ["heading_8bin", "range_abs_bucket", "range_signed_bucket"]:
        item = controls.get(regime, {})
        prefix = "r1_" + regime.replace("_bucket", "").replace("_8bin", "")
        row[f"{prefix}_best_layer"] = item.get("best_true_layer", "")
        row[f"{prefix}_same_minus_diff"] = float_or_blank(item.get("best_true_same_minus_diff"))
        row[f"{prefix}_true_minus_shuffle_mean"] = float_or_blank(item.get("true_minus_shuffle_mean"))
        row[f"{prefix}_between_total"] = float_or_blank(item.get("best_true_between_total_var_ratio"))
        row[f"{prefix}_nn_purity"] = float_or_blank(item.get("best_true_nn_same_regime_rate"))


def add_r2_fields(row: dict[str, Any], case_dir: Path) -> None:
    path = case_dir / "r2_subspace" / "phase95_r2_subspace_summary.json"
    if not path.exists():
        return
    summary = read_json(path)
    pairs = summary.get("best_true_by_pair", {})
    for pair, short in [
        ("heading_8bin__vs__range_abs_bucket", "abs"),
        ("heading_8bin__vs__range_signed_bucket", "signed"),
    ]:
        item = pairs.get(pair, {})
        row[f"r2_{short}_best_layer"] = item.get("layer_id", "")
        row[f"r2_{short}_heading_acc"] = float_or_blank(item.get("heading_test_acc_mean"))
        row[f"r2_{short}_range_acc"] = float_or_blank(item.get("range_test_acc_mean"))
        row[f"r2_{short}_cross_to_self"] = float_or_blank(item.get("cross_to_self_ratio"))
        row[f"r2_{short}_non_overlap"] = float_or_blank(item.get("non_overlap_score"))


def summarize(args: argparse.Namespace, cases: list[tuple[str, int]]) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    metrics = load_metrics(args.metrics_csv)
    rows: list[dict[str, Any]] = []
    for case_id, train_step in cases:
        case_dir = output_root / case_id
        row: dict[str, Any] = {"case_id": case_id, "train_step": train_step}
        row.update(metrics.get(case_id, {}))
        row["case_id"] = case_id
        row["train_step"] = train_step
        add_r1_fields(row, case_dir)
        add_r2_fields(row, case_dir)
        rows.append(row)
    write_csv(output_root / "phase98_r3_checkpoint_trajectory_summary.csv", rows)
    (output_root / "phase98_r3_checkpoint_trajectory_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def write_markdown(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase98-R3 Checkpoint Trajectory Summary",
        "",
        "Diagnostic-only local-val representation trajectory. No hidden-test labels are used.",
        "",
        "| case | step | final | angle | distance | heading gap | range abs gap | range signed gap | r2 abs cross/self | r2 signed cross/self |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{case}` | {step} | {final} | {angle} | {dist} | {hg} | {rag} | {rsg} | {r2a} | {r2s} |".format(
                case=row.get("case_id", ""),
                step=row.get("train_step", ""),
                final=format_cell(row.get("final_score_proxy", row.get("final", ""))),
                angle=format_cell(row.get("angle_rel_error", row.get("angle", ""))),
                dist=format_cell(row.get("distance_rel_error", row.get("distance", ""))),
                hg=format_cell(row.get("r1_heading_same_minus_diff", "")),
                rag=format_cell(row.get("r1_range_abs_same_minus_diff", "")),
                rsg=format_cell(row.get("r1_range_signed_same_minus_diff", "")),
                r2a=format_cell(row.get("r2_abs_cross_to_self", "")),
                r2s=format_cell(row.get("r2_signed_cross_to_self", "")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- Rising heading gap with falling val score supports late emergence of heading-regime readability.",
            "- Stable strong range-signed gap across early checkpoints supports range structure being present before final H8 convergence.",
            "- Cross/self ratios above 1 do not support a simple non-overlapping linear-subspace claim.",
        ]
    )
    (output_root / "phase98_r3_checkpoint_trajectory_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def format_cell(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    cases = parse_cases(args.case_list)

    manifest = {
        "phase": "phase98_r3_checkpoint_trajectory",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_root": str(output_root),
        "checkpoint_dir": str(checkpoint_dir),
        "json_root": args.json_root,
        "image_root": args.image_root,
        "model": args.model,
        "cases": [{"case_id": case, "train_step": step} for case, step in cases],
        "run_r1": bool(args.run_r1),
        "run_r2": bool(args.run_r2),
        "no_hidden_test_labels": True,
    }
    (output_root / "phase98_r3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    missing = []
    for case_id, _train_step in cases:
        ckpt = checkpoint_path(checkpoint_dir, case_id)
        if not ckpt.exists():
            missing.append(str(ckpt))
    if missing and args.require_checkpoints:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    for case_id, train_step in cases:
        ckpt = checkpoint_path(checkpoint_dir, case_id)
        if not ckpt.exists():
            print(f"[skip] missing checkpoint for {case_id}: {ckpt}", flush=True)
            continue
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_meta = {
            "case_id": case_id,
            "train_step": train_step,
            "checkpoint": str(ckpt),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (case_dir / "case_manifest.json").write_text(
            json.dumps(case_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[case] {case_id} step={train_step} checkpoint={ckpt}", flush=True)
        if args.run_r1:
            maybe_run_r1(args, repo_root, case_dir, case_id, ckpt)
        if args.run_r2:
            maybe_run_r2(args, repo_root, case_dir, case_id, ckpt)

    rows = summarize(args, cases)
    write_markdown(output_root, rows)
    print(json.dumps({"output_root": str(output_root), "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
