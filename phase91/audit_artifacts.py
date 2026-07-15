from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from .common import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_REPO_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAIN_JSON_ROOT,
    DEFAULT_VAL_JSON_ROOT,
    DEFAULT_WSTRIP_CHECKPOINT,
    ensure_run_root,
    iter_json_paths,
    load_pair_json,
    repo_relative,
    run_command,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase91 G0 artifact/code audit.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-json-root", type=Path, default=DEFAULT_TRAIN_JSON_ROOT)
    parser.add_argument("--val-json-root", type=Path, default=DEFAULT_VAL_JSON_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    return parser.parse_args()


def parse_output_modes(repo_root: Path) -> list[str]:
    source = repo_root / "reloc3r" / "reloc3r_relpose.py"
    text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
    modes = set(re.findall(r"output_mode\s*==\s*['\"]([^'\"]+)['\"]", text))
    defaults = re.findall(r"output_mode\s*=\s*['\"]([^'\"]+)['\"]", text)
    modes.update(defaults)
    return sorted(modes)


def parse_pose_heads(repo_root: Path) -> list[str]:
    source = repo_root / "reloc3r" / "pose_head.py"
    text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
    return sorted(set(re.findall(r"^class\s+([A-Za-z0-9_]*PairUAVHead)\b", text, flags=re.MULTILINE)))


def dataset_summary(root: Path) -> dict:
    paths = iter_json_paths(root)
    groups = sorted({p.parent.name for p in paths})
    samples = []
    for path in paths[:3]:
        try:
            samples.append(load_pair_json(path))
        except Exception as exc:
            samples.append({"json_path": str(path), "error": repr(exc)})
    return {
        "root": str(root),
        "exists": root.exists(),
        "json_count": len(paths),
        "group_count": len(groups),
        "first_groups": groups[:10],
        "sample_records": samples,
    }


def checkpoint_candidates(uavm_root: Path) -> list[dict]:
    patterns = [
        "synced_results/*/checkpoint-final.pth",
        "runs/phase88_axiswise_interaction_position_v1/train_runs/*/checkpoint-final.pth",
        "runs/explore_axisdecouple_reloc3r_head_v1/checkpoints/*.pth",
        "runs/explore_axisdecouple_reloc3r_head_v1/train_runs/*/checkpoint-final.pth",
        "runs/reloc3r_official_pairuav/*/checkpoint-final.pth",
    ]
    rows = []
    for pattern in patterns:
        for path in sorted(uavm_root.glob(pattern)):
            try:
                stat = path.stat()
            except OSError:
                continue
            rows.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size_mb": round(stat.st_size / (1024 * 1024), 3),
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                    "pattern": pattern,
                }
            )
    return rows


def phase88_metric_rows(uavm_root: Path) -> list[dict]:
    metric_root = uavm_root / "runs" / "phase88_axiswise_interaction_position_v1" / "metrics"
    rows = []
    for path in sorted(metric_root.glob("*/val_metrics_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"metric_path": str(path), "error": repr(exc)})
            continue
        row = {
            "run_name": path.parent.name,
            "metric_file": path.name,
            "metric_path": str(path),
        }
        for key in (
            "sample_count",
            "angle_mae_deg",
            "distance_mae",
            "distance_rel_error",
            "angle_rel_error",
            "final_score",
            "score",
        ):
            if key in payload:
                row[key] = payload[key]
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    uavm_root = repo_root.parents[1] if repo_root.name == "reloc3r_pairuav" else DEFAULT_REPO_ROOT.parents[1]
    ensure_run_root(run_root)

    git_status = run_command(["git", "status", "--short"], cwd=repo_root)
    git_head = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)

    expected_files = [
        "train.py",
        "eval_pairuav.py",
        "infer_pairuav_with_progress.py",
        "scripts/train_pairuav_full_devsplit.sh",
        "scripts/eval_pairuav_official_devsplit.sh",
        "reloc3r/reloc3r_relpose.py",
        "reloc3r/pose_head.py",
        "reloc3r/pairuav_metrics.py",
        "reloc3r/loss.py",
        "reloc3r/datasets/pairuav.py",
    ]
    file_rows = [
        {
            "path": item,
            "exists": (repo_root / item).exists(),
            "kind": "required_code" if item in expected_files[:6] else "support_code",
        }
        for item in expected_files
    ]

    train_summary = dataset_summary(args.train_json_root)
    val_summary = dataset_summary(args.val_json_root)
    checkpoints = checkpoint_candidates(uavm_root)
    baselines = phase88_metric_rows(uavm_root)
    output_modes = parse_output_modes(repo_root)
    pose_heads = parse_pose_heads(repo_root)

    selected_surface = {
        "train_json_root": str(args.train_json_root),
        "val_json_root": str(args.val_json_root),
        "image_root": str(args.image_root),
        "resolution": "(512,384)",
        "seed": 777,
        "smoke_max_pairs": 32,
        "phase91_default_init_policy": "Wstrip/Reloc3r-512 backbone-only when available; token extraction does not require pose-head weights.",
        "phase91_default_checkpoint": str(DEFAULT_WSTRIP_CHECKPOINT),
        "phase91_default_checkpoint_exists": DEFAULT_WSTRIP_CHECKPOINT.exists(),
    }

    audit = {
        "phase": "phase91_g0_artifact_code_audit",
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_status": git_status,
        "git_head": git_head,
        "required_files": file_rows,
        "output_modes": output_modes,
        "pairuav_head_classes": pose_heads,
        "train_dataset": train_summary,
        "val_dataset": val_summary,
        "image_root_exists": args.image_root.exists(),
        "selected_surface_contract": selected_surface,
        "pass": all(row["exists"] for row in file_rows[:6])
        and train_summary["json_count"] > 0
        and val_summary["json_count"] > 0,
    }

    write_json(run_root / "audits" / "artifact_code_audit.json", audit)
    write_json(run_root / "manifests" / "selected_surface_contract.json", selected_surface)
    write_csv(run_root / "manifests" / "available_checkpoints.csv", checkpoints)
    write_csv(run_root / "manifests" / "available_baselines.csv", baselines)
    write_csv(run_root / "audits" / "required_code_files.csv", file_rows)
    write_csv(
        run_root / "audits" / "output_modes.csv",
        [{"output_mode": mode} for mode in output_modes],
        fieldnames=["output_mode"],
    )

    md_lines = [
        "# Phase91 G0 Artifact And Code Audit",
        "",
        f"- repo_root: `{repo_root}`",
        f"- run_root: `{run_root}`",
        f"- git_head: `{git_head.get('stdout', '').strip()}`",
        f"- git_status_lines: {len([x for x in git_status.get('stdout', '').splitlines() if x.strip()])}",
        f"- required_code_pass: {audit['pass']}",
        f"- train_json_count: {train_summary['json_count']}",
        f"- val_json_count: {val_summary['json_count']}",
        f"- image_root_exists: {args.image_root.exists()}",
        f"- output_mode_count: {len(output_modes)}",
        f"- pairuav_head_class_count: {len(pose_heads)}",
        f"- checkpoint_candidates: {len(checkpoints)}",
        f"- imported_phase88_metric_rows: {len(baselines)}",
        "",
        "## Selected Surface Contract",
        "",
        "```json",
        json.dumps(selected_surface, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Dirty Worktree Note",
        "",
        "Existing dirty or untracked files are audit context only. Phase91 should add an isolated `phase91/` package and run-root artifacts without reverting prior changes.",
    ]
    write_text(run_root / "audits" / "artifact_code_audit.md", "\n".join(md_lines))
    print(json.dumps({"ok": audit["pass"], "run_root": str(run_root), "audit": str(run_root / "audits" / "artifact_code_audit.json")}, ensure_ascii=False))
    return 0 if audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

