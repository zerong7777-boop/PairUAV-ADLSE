"""Write Phase27 A taxonomy redesign-v2 markdown reports.

The report is deliberately stdlib-only. It summarizes analysis artifacts from
the manifest and metrics stages and states that the route is not training
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATE_HARD = "evidence_sufficient_hard"
STATE_CONTROL = "stable_control_anchor"
STATE_AMBIGUOUS = "ambiguous_unreliable"
OLD_HARD = "hard_trainable"
OLD_CONTROL = "ordinary_control_anchor"
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


def _count_csv_rows(path: str | Path) -> int:
    return len(_read_csv_rows(path))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _state_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter((row.get("derived_state") or "missing") for row in rows)


def _old_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter((row.get("old_base_regime") or "missing") for row in rows)


def _old_to_new_count(rows: list[dict[str, str]], old_label: str, new_label: str) -> int:
    return sum(
        1
        for row in rows
        if (row.get("old_base_regime") or "").lower() == old_label.lower()
        and row.get("derived_state") == new_label
    )


def _top_bias_rows(rows: list[dict[str, str]], limit: int = 5) -> list[str]:
    parsed = []
    for row in rows:
        try:
            fraction = float(row.get("fraction_within_target_scene", ""))
        except ValueError:
            continue
        parsed.append(
            (
                fraction,
                row.get("target_key") or row.get("target", "unknown"),
                row.get("scene_key") or row.get("scene", "unknown"),
                row.get("derived_state", "unknown"),
                row.get("count", "0"),
            )
        )
    parsed.sort(reverse=True)
    return [
        f"- `{target}` / `{scene}` -> `{state}`: {count} rows ({fraction * 100.0:.2f}%)"
        for fraction, target, scene, state, count in parsed[:limit]
    ]


def _new_delta_value(ci: dict[str, Any]) -> Any:
    for key in NEW_DELTA_KEYS:
        if key in ci:
            return ci.get(key)
    return None


def build_report(
    evidence_manifest: str | Path,
    full_dev_surface: str | Path,
    stress_surfaces: list[str],
    manifest: str | Path,
    metrics_dir: str | Path,
    source_registry: str | Path | None,
    leakage_audit: str | Path | None,
    title: str,
) -> str:
    metrics_dir = Path(metrics_dir)
    manifest_rows = _read_csv_rows(manifest)
    state_counts = _state_counts(manifest_rows)
    old_counts = _old_counts(manifest_rows)
    row_count = len(manifest_rows)

    registry = _read_json(source_registry, {}) if source_registry else {}
    audit = _read_json(leakage_audit, {}) if leakage_audit else {}
    coverage = _read_json(metrics_dir / "join_coverage.json", {})
    tail = _read_json(metrics_dir / "tail_risk_metrics.json", {})
    ci = _read_json(metrics_dir / "bootstrap_ci.json", {})
    verdict = _read_json(metrics_dir / "final_verdict.json", {})
    bias_rows = _read_csv_rows(metrics_dir / "target_scene_bias.csv")

    leakage_checks = audit.get("leakage_checks", {}) if isinstance(audit, dict) else {}
    leakage_ok = bool(leakage_checks.get("preserved_all_evidence_rows", False)) and not bool(
        verdict.get("forbidden_training_policy_spec_emitted", False)
    )
    leakage_verdict = "PASS: analysis-only route preserved evidence rows and emitted no training policy spec"
    if not leakage_ok:
        leakage_verdict = "REVIEW: leakage/audit checks are incomplete or failed"

    old_hard = old_counts.get(OLD_HARD, 0)
    new_hard = state_counts.get(STATE_HARD, 0)
    old_hard_to_new_hard = _old_to_new_count(manifest_rows, OLD_HARD, STATE_HARD)
    old_ordinary = old_counts.get(OLD_CONTROL, 0)
    new_control = state_counts.get(STATE_CONTROL, 0)
    new_delta = _new_delta_value(ci)

    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "This report summarizes Phase27 A taxonomy redesign-v2 analysis artifacts. It is not training evidence, not inference evidence, and not a training-policy specification.",
        "",
        "## Inputs",
        "",
        f"- Evidence manifest: `{evidence_manifest}` ({_count_csv_rows(evidence_manifest)} rows)",
        f"- Full-dev baseline surface: `{full_dev_surface}` ({_count_csv_rows(full_dev_surface)} rows)",
    ]
    for index, path in enumerate(stress_surfaces, start=1):
        lines.append(f"- Stress surface {index}: `{path}` ({_count_csv_rows(path)} rows)")
    if registry:
        lines.append(f"- Source registry: `{source_registry}`")
    lines.extend(
        [
            f"- Analysis manifest: `{manifest}` ({row_count} rows)",
            "",
            "## Join Coverage",
            "",
            f"- Total rows: {_num(coverage.get('total_rows', row_count))}",
            f"- Full-dev joined: {_num(coverage.get('full_dev_joined_count'))} ({_pct(coverage.get('full_dev_joined_fraction'))})",
            f"- Stress joined: {_num(coverage.get('stress_joined_count'))} ({_pct(coverage.get('stress_joined_fraction'))})",
            f"- Leakage verdict: {leakage_verdict}",
            "",
            "## Old/New Taxonomy Results",
            "",
            f"- Old `hard_trainable` (`old_base_regime == hard_trainable`): {old_hard} rows",
            f"- New `{STATE_HARD}`: {new_hard} rows",
            f"- Old `hard_trainable` rows retained as `{STATE_HARD}`: {old_hard_to_new_hard} rows",
            f"- Old `ordinary_control_anchor` (`old_base_regime == ordinary_control_anchor`): {old_ordinary} rows",
            f"- New `{STATE_CONTROL}`: {new_control} rows",
            "",
            "## Ambiguous/Tail Risk",
            "",
            f"- `{STATE_AMBIGUOUS}` rows: {state_counts.get(STATE_AMBIGUOUS, 0)}",
            f"- Tail outlier rows: {_num(tail.get('tail_outlier_count'))} ({_pct(tail.get('tail_outlier_rate'))})",
            f"- Tail state counts: `{json.dumps(tail.get('tail_state_counts', {}), sort_keys=True)}`",
            "",
            "## Target/Scene Bias",
            "",
        ]
    )
    top_bias = _top_bias_rows(bias_rows)
    lines.extend(top_bias if top_bias else ["- No target/scene bias table was available."])
    lines.extend(
        [
            "",
            "## Bootstrap CI",
            "",
            f"- Bootstrap iterations: {_num(ci.get('bootstrap_iters'))}",
            f"- Seed: {_num(ci.get('seed'))}",
            f"- New `{STATE_HARD}` minus new `{STATE_CONTROL}` delta: {_num(new_delta)}",
            f"- 95% CI: [{_num(ci.get('ci_low'))}, {_num(ci.get('ci_high'))}]",
            "",
            "## Final Metrics Verdict",
            "",
            f"- Verdict label: `{verdict.get('label', 'missing')}`",
            f"- Minimum join coverage: {_pct(verdict.get('min_join_coverage'))}",
            f"- Tail outlier rate: {_pct(verdict.get('tail_outlier_rate'))}",
            "- Route disposition: bounded analysis artifact for knowledge review only; it must not be used as training evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--full-dev-surface", required=True)
    parser.add_argument("--stress-surface", action="append", default=[])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--source-registry")
    parser.add_argument("--leakage-audit")
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-summary")
    parser.add_argument("--title", default="Phase27 A Taxonomy Redesign-v2 Report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(
        evidence_manifest=args.evidence_manifest,
        full_dev_surface=args.full_dev_surface,
        stress_surfaces=args.stress_surface,
        manifest=args.manifest,
        metrics_dir=args.metrics_dir,
        source_registry=args.source_registry,
        leakage_audit=args.leakage_audit,
        title=args.title,
    )
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report + "\n", encoding="utf-8")
    if args.out_summary:
        out_summary = Path(args.out_summary)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
