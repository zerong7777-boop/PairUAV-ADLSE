from __future__ import annotations

import argparse
import csv
import json
import shlex
import time
from pathlib import Path
from typing import Any

from .common import DEFAULT_RUN_ROOT, DEFAULT_WSTRIP_CHECKPOINT, ensure_run_root, write_csv, write_json, write_text


BASELINES = [
    {
        "baseline_id": "B0",
        "baseline_name": "H0/shared readout",
        "output_mode": "pairuav_heading_range",
        "rationale": "Shared token-to-grid trunk with separate scalar heading/range output layers.",
    },
    {
        "baseline_id": "B1",
        "baseline_name": "ordinary two-head readout",
        "output_mode": "pairuav_mid_split_heading_range",
        "rationale": "Shared projection/resconv with separate heading/range MLP trunks.",
    },
    {
        "baseline_id": "B2",
        "baseline_name": "fixed H3 heading branch",
        "output_mode": "pairuav_range_h0_heading_h3_heading_range",
        "rationale": "Range stays on H0; heading gets an independent last-layer projection/resconv/MLP path.",
    },
    {
        "baseline_id": "B3",
        "baseline_name": "dual-path early split",
        "output_mode": "pairuav_early_split_heading_range",
        "rationale": "Heading and range use separate projection/resconv/MLP paths from final decoder tokens.",
    },
    {
        "baseline_id": "B4",
        "baseline_name": "H8 mid-late readout",
        "output_mode": "pairuav_range_h0_heading_mid_late_heading_range",
        "rationale": "Range stays on H0; heading reads mid and late decoder layers with feature fusion.",
    },
]


def _shell_env(values: dict[str, Any]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def build_replay_rows(
    run_root: Path,
    max_train_steps: int,
    eval_max_samples: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    checkpoint: str,
) -> list[dict[str, Any]]:
    train_root = run_root / "router_smokes" / "baseline_replay_train_runs"
    eval_root = run_root / "router_smokes" / "baseline_replay_eval"
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        run_name = (
            f"phase91_{baseline['baseline_id']}_matched_"
            f"steps{int(max_train_steps)}_val{int(eval_max_samples)}_lr{lr:g}"
        )
        train_dir = train_root / run_name
        eval_dir = eval_root / run_name
        model_expr = f"Reloc3rRelpose(img_size=512, output_mode='{baseline['output_mode']}')"
        train_env = {
            "UAVM_ROOT": "/media/jgzn/SSD_lexar/RZ/UAVM",
            "PYTHON_BIN": "/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python",
            "OUTPUT_ROOT": str(train_root),
            "RUN_NAME": run_name,
            "MODEL_EXPR": model_expr,
            "PRETRAINED": checkpoint,
            "MAX_TRAIN_STEPS": int(max_train_steps),
            "STEP_CHECKPOINT_FREQ": 0,
            "EVAL_FREQ": 0,
            "EPOCHS": 1,
            "BATCH_SIZE": int(batch_size),
            "NUM_WORKERS": 4,
            "LR": lr,
            "WARMUP_EPOCHS": 0,
            "AMP": 1,
        }
        eval_env = {
            "UAVM_ROOT": "/media/jgzn/SSD_lexar/RZ/UAVM",
            "PYTHON_BIN": "/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python",
            "MODEL_EXPR": model_expr,
            "BATCH_SIZE": int(eval_batch_size),
            "NUM_WORKERS": 4,
            "AMP": 1,
            "LOG_EVERY": 200,
        }
        rows.append(
            {
                "baseline_id": baseline["baseline_id"],
                "baseline_name": baseline["baseline_name"],
                "output_mode": baseline["output_mode"],
                "rationale": baseline["rationale"],
                "run_name": run_name,
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "checkpoint": checkpoint,
                "max_train_steps": int(max_train_steps),
                "eval_max_samples": int(eval_max_samples),
                "batch_size": int(batch_size),
                "eval_batch_size": int(eval_batch_size),
                "lr": lr,
                "train_env": _shell_env(train_env),
                "eval_env": _shell_env(eval_env),
            }
        )
    return rows


def _write_launcher(run_root: Path, rows: list[dict[str, Any]], skip_existing: bool) -> Path:
    script_path = run_root / "router_smokes" / "run_matched_baseline_replay.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "REPO_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav",
        f"RUN_ROOT={shlex.quote(str(run_root))}",
        "cd \"$REPO_ROOT\"",
        "mkdir -p \"$RUN_ROOT/router_smokes/logs\"",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"echo '=== {row['baseline_id']} {row['baseline_name']} train start' | tee -a \"$RUN_ROOT/router_smokes/logs/replay_driver.log\"",
                f"mkdir -p {shlex.quote(row['train_dir'])} {shlex.quote(row['eval_dir'])}",
            ]
        )
        if skip_existing:
            lines.extend(
                [
                    f"if [[ -f {shlex.quote(row['train_dir'] + '/checkpoint-final.pth')} ]]; then",
                    f"  echo 'skip existing train {row['run_name']}' | tee -a \"$RUN_ROOT/router_smokes/logs/replay_driver.log\"",
                    "else",
                    f"  env {row['train_env']} bash scripts/train_pairuav_full_devsplit.sh > {shlex.quote(row['train_dir'] + '/train.log')} 2>&1",
                    "fi",
                ]
            )
        else:
            lines.append(
                f"env {row['train_env']} bash scripts/train_pairuav_full_devsplit.sh > {shlex.quote(row['train_dir'] + '/train.log')} 2>&1"
            )
        lines.extend(
            [
                f"echo '=== {row['baseline_id']} eval start' | tee -a \"$RUN_ROOT/router_smokes/logs/replay_driver.log\"",
                f"env {row['eval_env']} bash scripts/eval_pairuav_official_devsplit.sh "
                f"{shlex.quote(row['train_dir'] + '/checkpoint-final.pth')} "
                f"{shlex.quote(row['eval_dir'])} "
                f"{int(row['eval_max_samples'])} > {shlex.quote(row['eval_dir'] + '/eval.log')} 2>&1",
                f"echo '=== {row['baseline_id']} done' | tee -a \"$RUN_ROOT/router_smokes/logs/replay_driver.log\"",
                "",
            ]
        )
    lines.extend(
        [
            "/media/jgzn/SSD_lexar/conda_envs/uavm/bin/python -m phase91.matched_baseline_replay --collect --run-root \"$RUN_ROOT\"",
            "echo 'all matched baseline replay jobs completed' | tee -a \"$RUN_ROOT/router_smokes/logs/replay_driver.log\"",
        ]
    )
    write_text(script_path, "\n".join(lines))
    return script_path


def write_replay_plan(
    run_root: Path,
    max_train_steps: int,
    eval_max_samples: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    checkpoint: str,
    skip_existing: bool,
) -> dict[str, Any]:
    ensure_run_root(run_root)
    rows = build_replay_rows(
        run_root=run_root,
        max_train_steps=max_train_steps,
        eval_max_samples=eval_max_samples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr,
        checkpoint=checkpoint,
    )
    fields = [
        "baseline_id",
        "baseline_name",
        "output_mode",
        "rationale",
        "run_name",
        "train_dir",
        "eval_dir",
        "checkpoint",
        "max_train_steps",
        "eval_max_samples",
        "batch_size",
        "eval_batch_size",
        "lr",
        "train_env",
        "eval_env",
    ]
    write_csv(run_root / "router_smokes" / "matched_baseline_replay_plan.csv", rows, fieldnames=fields)
    script_path = _write_launcher(run_root, rows, skip_existing=skip_existing)
    payload = {
        "phase": "phase91_matched_baseline_replay_plan",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_root": str(run_root),
        "launcher": str(script_path),
        "skip_existing": bool(skip_existing),
        "rows": rows,
    }
    write_json(run_root / "router_smokes" / "matched_baseline_replay_plan.json", payload)

    md = [
        "# Phase91 Matched B0-B4 Baseline Replay Plan",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- max_train_steps: {max_train_steps}",
        f"- eval_max_samples: {eval_max_samples}",
        f"- checkpoint: `{checkpoint}`",
        f"- launcher: `{script_path}`",
        "",
        "## Baselines",
        "",
    ]
    for row in rows:
        md.append(f"- `{row['baseline_id']}` `{row['output_mode']}`: {row['rationale']}")
    md.extend(
        [
            "",
            "## Rule",
            "",
            "These runs are matched Phase91 smoke baselines. They are valid for Router-L/Q comparison only under the same selected surface, init, step budget, lr, batch size, and evaluator.",
        ]
    )
    write_text(run_root / "router_smokes" / "matched_baseline_replay_plan.md", "\n".join(md))
    return payload


def collect_replay_results(run_root: Path) -> dict[str, Any]:
    plan_path = run_root / "router_smokes" / "matched_baseline_replay_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing replay plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    for row in plan["rows"]:
        eval_dir = Path(row["eval_dir"])
        metrics_path = eval_dir / "official_metrics.json"
        train_ckpt = Path(row["train_dir"]) / "checkpoint-final.pth"
        out = {
            "baseline_id": row["baseline_id"],
            "baseline_name": row["baseline_name"],
            "output_mode": row["output_mode"],
            "run_name": row["run_name"],
            "train_checkpoint_exists": train_ckpt.exists(),
            "metrics_exists": metrics_path.exists(),
            "train_dir": row["train_dir"],
            "eval_dir": row["eval_dir"],
        }
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "final_score",
                    "score",
                    "angle_mae_deg",
                    "distance_mae",
                    "angle_rel_error",
                    "distance_rel_error",
                    "final_score_proxy",
                    "samples",
                ):
                    if key in metrics:
                        out[key] = metrics[key]
            except Exception as exc:
                out["metrics_error"] = repr(exc)
        rows.append(out)
    write_csv(run_root / "router_smokes" / "matched_baseline_replay_status.csv", rows)
    summary = {
        "phase": "phase91_matched_baseline_replay_status",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete_count": sum(1 for row in rows if row["metrics_exists"]),
        "required_count": len(rows),
        "complete": all(row["metrics_exists"] for row in rows),
        "rows": rows,
    }
    write_json(run_root / "router_smokes" / "matched_baseline_replay_status.json", summary)
    if summary["complete"]:
        write_csv(run_root / "router_smokes" / "matched_baseline_metrics.csv", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or collect Phase91 matched baseline replay.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-train-steps", type=int, default=2500)
    parser.add_argument("--eval-max-samples", type=int, default=811)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--checkpoint", default=str(DEFAULT_WSTRIP_CHECKPOINT))
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--collect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if args.collect:
        payload = collect_replay_results(run_root)
        print(json.dumps({"complete": payload["complete"], "complete_count": payload["complete_count"]}, ensure_ascii=False))
    else:
        payload = write_replay_plan(
            run_root=run_root,
            max_train_steps=args.max_train_steps,
            eval_max_samples=args.eval_max_samples,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            checkpoint=args.checkpoint,
            skip_existing=not args.no_skip_existing,
        )
        print(json.dumps({"launcher": payload["launcher"], "rows": len(payload["rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
