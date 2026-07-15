"""Write a Phase27 A taxonomy redesign-v2 experiment record.

The record is a stdlib-only markdown artifact intended for controller review.
It records commands, artifact paths, row counts, old/new taxonomy metrics, CI,
bias, leakage verdict, final route verdict, and knowledge-review requirement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


OLD_HARD = "hard_trainable"
OLD_CONTROL = "ordinary_control_anchor"
NEW_HARD = "evidence_sufficient_hard"
NEW_CONTROL = "stable_control_anchor"
NEW_DELTA_KEYS = (
    "new_evidence_sufficient_hard_minus_new_stable_control_anchor_delta",
    "new_evidence_sufficient_hard_minus_new_stable_control_anchor",
    "evidence_sufficient_hard_minus_stable_control_anchor_delta",
    "evidence_sufficient_hard_minus_stable_control_anchor",
)


def _read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _commands_from_args(commands: list[str], command_file: str | None) -> list[str]:
    result = list(commands)
    if command_file and Path(command_file).exists():
        result.extend(
            line.strip()
            for line in Path(command_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return result


def _top_bias(rows: list[dict[str, str]], limit: int = 5) -> list[str]:
    ranked = []
    for row in rows:
        try:
            fraction = float(row.get("fraction_within_target_scene", ""))
        except ValueError:
            continue
        ranked.append((fraction, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        "- `{target}` / `{scene}` -> `{state}`: {count} rows ({pct})".format(
            target=row.get("target_key") or row.get("target", "unknown"),
            scene=row.get("scene_key") or row.get("scene", "unknown"),
            state=row.get("derived_state", "unknown"),
            count=row.get("count", "0"),
            pct=_pct(fraction),
        )
        for fraction, row in ranked[:limit]
    ]


def _new_delta_value(ci: dict[str, Any]) -> Any:
    for key in NEW_DELTA_KEYS:
        if key in ci:
            return ci.get(key)
    return None


def build_record(
    commands: list[str],
    evidence_manifest: str,
    full_dev_surface: str,
    stress_surfaces: list[str],
    manifest: str,
    metrics_dir: str,
    source_registry: str | None,
    leakage_audit: str | None,
    report: str | None,
    summary: str | None,
) -> str:
    manifest_rows = _read_csv_rows(manifest)
    states = Counter(row.get("derived_state") or "missing" for row in manifest_rows)
    old = Counter(row.get("old_base_regime") or "missing" for row in manifest_rows)
    metrics_path = Path(metrics_dir)
    coverage = _read_json(metrics_path / "join_coverage.json", {})
    ci = _read_json(metrics_path / "bootstrap_ci.json", {})
    new_delta = _new_delta_value(ci)
    tail = _read_json(metrics_path / "tail_risk_metrics.json", {})
    verdict = _read_json(metrics_path / "final_verdict.json", {})
    audit = _read_json(leakage_audit, {}) if leakage_audit else {}
    bias_rows = _read_csv_rows(metrics_path / "target_scene_bias.csv")

    leakage_checks = audit.get("leakage_checks", {}) if isinstance(audit, dict) else {}
    leakage_ok = bool(leakage_checks.get("preserved_all_evidence_rows", False)) and not bool(
        verdict.get("forbidden_training_policy_spec_emitted", False)
    )
    leakage_verdict = "PASS" if leakage_ok else "REVIEW_REQUIRED"

    artifact_lines = [
        f"- Evidence manifest: `{evidence_manifest}` ({len(_read_csv_rows(evidence_manifest))} rows)",
        f"- Full-dev baseline surface: `{full_dev_surface}` ({len(_read_csv_rows(full_dev_surface))} rows)",
    ]
    for index, path in enumerate(stress_surfaces, start=1):
        artifact_lines.append(f"- Stress surface {index}: `{path}` ({len(_read_csv_rows(path))} rows)")
    artifact_lines.extend(
        [
            f"- Analysis manifest: `{manifest}` ({len(manifest_rows)} rows)",
            f"- Metrics directory: `{metrics_dir}`",
        ]
    )
    if source_registry:
        artifact_lines.append(f"- Source registry: `{source_registry}`")
    if leakage_audit:
        artifact_lines.append(f"- Leakage audit: `{leakage_audit}`")
    if report:
        artifact_lines.append(f"- Report: `{report}`")
    if summary:
        artifact_lines.append(f"- Docs/AI summary: `{summary}`")

    lines = [
        "# Phase27 A Taxonomy Redesign-v2 Experiment Record",
        "",
        "## Commands Run",
        "",
    ]
    lines.extend([f"- `{command}`" for command in commands] if commands else ["- No command log was provided."])
    lines.extend(
        [
            "",
            "## Artifacts and Row Counts",
            "",
            *artifact_lines,
            "",
            "## Old/New Metrics",
            "",
            f"- Old `hard_trainable` (`old_base_regime == hard_trainable`): {old.get(OLD_HARD, 0)} rows",
            f"- New `{NEW_HARD}`: {states.get(NEW_HARD, 0)} rows",
            f"- Old `ordinary_control_anchor` (`old_base_regime == ordinary_control_anchor`): {old.get(OLD_CONTROL, 0)} rows",
            f"- New `{NEW_CONTROL}`: {states.get(NEW_CONTROL, 0)} rows",
            f"- New `ambiguous_unreliable`: {states.get('ambiguous_unreliable', 0)} rows",
            "",
            "## Join Coverage",
            "",
            f"- Total rows: {_fmt(coverage.get('total_rows', len(manifest_rows)))}",
            f"- Full-dev joined: {_fmt(coverage.get('full_dev_joined_count'))} ({_pct(coverage.get('full_dev_joined_fraction'))})",
            f"- Stress joined: {_fmt(coverage.get('stress_joined_count'))} ({_pct(coverage.get('stress_joined_fraction'))})",
            "",
            "## Bootstrap CI",
            "",
            f"- Bootstrap iterations: {_fmt(ci.get('bootstrap_iters'))}",
            f"- Seed: {_fmt(ci.get('seed'))}",
            f"- New `{NEW_HARD}` minus new `{NEW_CONTROL}` delta: {_fmt(new_delta)}",
            f"- 95% CI: [{_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}]",
            "",
            "## Bias and Tail Risk",
            "",
            f"- Tail outlier rows: {_fmt(tail.get('tail_outlier_count'))} ({_pct(tail.get('tail_outlier_rate'))})",
            f"- Tail state counts: `{json.dumps(tail.get('tail_state_counts', {}), sort_keys=True)}`",
        ]
    )
    lines.extend(_top_bias(bias_rows) or ["- No target/scene bias table was available."])
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            f"- Leakage verdict: `{leakage_verdict}`",
            f"- Final route verdict: `{verdict.get('label', 'missing')}`",
            "- This is bounded analysis only and is not training evidence.",
            "- Knowledge-review requirement: controller/human knowledge review is required before any downstream route, training-policy change, packaging, or submission.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--command-file")
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--full-dev-surface", required=True)
    parser.add_argument("--stress-surface", action="append", default=[])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--source-registry")
    parser.add_argument("--leakage-audit")
    parser.add_argument("--report")
    parser.add_argument("--summary")
    parser.add_argument("--out-record", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = build_record(
        commands=_commands_from_args(args.command, args.command_file),
        evidence_manifest=args.evidence_manifest,
        full_dev_surface=args.full_dev_surface,
        stress_surfaces=args.stress_surface,
        manifest=args.manifest,
        metrics_dir=args.metrics_dir,
        source_registry=args.source_registry,
        leakage_audit=args.leakage_audit,
        report=args.report,
        summary=args.summary,
    )
    out_record = Path(args.out_record)
    out_record.parent.mkdir(parents=True, exist_ok=True)
    out_record.write_text(record + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
