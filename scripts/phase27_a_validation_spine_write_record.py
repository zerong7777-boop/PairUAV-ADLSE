#!/usr/bin/env python3
"""Write Phase27 A validation spine experiment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path, text):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--spec-path", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    combined = read_json(out_dir / "metrics" / "combined_validation_metrics.json")
    readiness = read_json(out_dir / "metrics" / "training_readiness_metrics.json")
    registry = read_json(out_dir / "registries" / "artifacts.json")
    verdict = readiness["verdict"]
    blocker = combined.get("blocker", "")
    row_counts = combined.get("row_counts", {})
    knowledge_review_required = combined.get("knowledge_review_required", True)

    artifact_lines = []
    for artifact_id, entry in registry.items():
        artifact_lines.append(f"- `{artifact_id}`: `{entry.get('path', '')}` rows={entry.get('row_count', '')}")

    metrics_paths = [
        "identity_metrics.json",
        "lineage_metrics.json",
        "leakage_metrics.json",
        "state_distribution_metrics.json",
        "state_error_association_metrics.json",
        "matcher_sufficiency_metrics.json",
        "control_stability_metrics.json",
        "training_readiness_metrics.json",
        "combined_validation_metrics.json",
    ]
    metric_lines = [f"- `{out_dir / 'metrics' / name}`" for name in metrics_paths]

    record = "\n".join([
        "# Exp 20260508 Phase27 A Validation Spine",
        "",
        f"Spec: `{args.spec_path}`",
        "",
        f"Plan: `{args.plan_path}`",
        "",
        f"Command: `{args.command}`",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"Blocker: `{blocker}`",
        "",
        "Row counts:",
        "",
        *[f"- `{k}`: {v}" for k, v in row_counts.items()],
        "",
        "Artifacts:",
        "",
        *artifact_lines,
        "",
        "Metrics:",
        "",
        *metric_lines,
        "",
        f"Knowledge-review required: `{knowledge_review_required}`",
        "",
        "Conclusion:",
        "",
        "The validation spine bounded eval executed on existing read-only artifacts. It did not train, finetune, modify checkpoints, change inference, package results, or submit to Codabench.",
        "",
    ])
    write_text(out_dir / "exp-20260508-phase27-a-validation-spine.md", record)
    write_text(project_root / "docs" / "ai" / "2026-05-08-phase27-a-validation-spine.md", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
