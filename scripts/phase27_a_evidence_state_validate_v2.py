#!/usr/bin/env python3
"""Validate Phase27 A evidence-state manifest v2 without training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_schema_v2 import (
    BASE_FLAG_COLUMNS,
    BASE_REGIMES,
    RISK_TAGS,
    audit_construction_columns_v2,
)


VALIDATION_SOURCES = {
    "baseline_matched": "experiments/phase26_b1_geometry_local_alignment/eval/baseline_matched/official_per_sample.csv",
    "b_scr": "experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_heading_selective/official_per_sample.csv",
    "b_scr_gate_off": "experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_gate_off/official_per_sample.csv",
    "b1_frozen_matcher": "experiments/phase26_b1_frozen_external_matcher/eval/frozen_matcher_fusion/official_per_sample.csv",
}

FEATURE_PATHS = {
    "train": "experiments/phase26_b_selective_correspondence_reasoning/features/train_bscr_features.jsonl",
    "eval": "experiments/phase26_b_selective_correspondence_reasoning/features/eval_bscr_features.jsonl",
}

IDENTITY_MANIFESTS = {
    "train": "experiments/paper_pillars/15_train_test_distribution_gap/manifests/train_labeled_manifest.csv",
    "dev": "experiments/paper_pillars/15_train_test_distribution_gap/manifests/dev_labeled_manifest.csv",
}

V1_METRICS = "experiments/phase27_a_evidence_state_manifest/metrics/a_evidence_state_metrics.json"

CONSTRUCTION_EXEMPT = {
    "pair_id",
    "query_name",
    "reference_name",
    "construction_source",
    "schema_version",
    "leakage_audit_passed",
    "base_regime",
    "ordinary_with_risk_tag",
    "semantic_proxy",
    "geometry_proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        count = sum(1 for _ in handle)
    return count - 1 if path.suffix.lower() == ".csv" and count > 0 else count


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _active_base(row: dict[str, str]) -> list[str]:
    return [regime for regime in BASE_REGIMES if str(row.get(f"base_{regime}", "0")) in {"1", "1.0", "true", "True"}]


def _base_and_tag_counts(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int], int]:
    base_counts = {regime: 0 for regime in BASE_REGIMES}
    tag_counts = {tag: 0 for tag in RISK_TAGS}
    ordinary_with_risk = 0
    for row in rows:
        active = _active_base(row)
        if len(active) == 1:
            base_counts[active[0]] += 1
        for tag in RISK_TAGS:
            tag_counts[tag] += int(str(row.get(tag, "0")) in {"1", "1.0", "true", "True"})
        ordinary_with_risk += int(str(row.get("ordinary_with_risk_tag", "0")) in {"1", "1.0", "true", "True"})
    return base_counts, tag_counts, ordinary_with_risk


def _invariant_violations(rows: list[dict[str, str]]) -> list[str]:
    violations: list[str] = []
    for idx, row in enumerate(rows):
        active = _active_base(row)
        if len(active) != 1:
            violations.append(f"row {idx} has {len(active)} active base regimes")
        elif row.get("base_regime") != active[0]:
            violations.append(f"row {idx} base_regime mismatch: {row.get('base_regime')} != {active[0]}")
        for tag in RISK_TAGS:
            if str(row.get(tag, "")) not in {"0", "1"}:
                violations.append(f"row {idx} risk tag {tag} is not binary: {row.get(tag)}")
    return violations[:20]


def _load_validation(project_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, rel_path in VALIDATION_SOURCES.items():
        rows = _read_csv(project_root / rel_path)
        out[name] = {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}
    return out


def _joined_validation(rows: list[dict[str, str]], validation: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    manifest = {row["pair_id"]: row for row in rows if row.get("pair_id")}
    source_join_counts = {
        name: sum(1 for pair_id in manifest if pair_id in source)
        for name, source in validation.items()
    }
    baseline = validation.get("baseline_matched", {})
    bscr = validation.get("b_scr", {})
    gate_off = validation.get("b_scr_gate_off", {})
    joined_ids = sorted(set(manifest) & set(baseline) & set(bscr))
    baseline_scores = [
        _to_float(baseline[pair_id].get("final_score"))
        for pair_id in joined_ids
        if _to_float(baseline[pair_id].get("final_score")) is not None
    ]
    sorted_scores = sorted(score for score in baseline_scores if score is not None)
    if sorted_scores:
        hard_threshold = sorted_scores[int(0.75 * (len(sorted_scores) - 1))]
        control_threshold = sorted_scores[int(0.25 * (len(sorted_scores) - 1))]
    else:
        hard_threshold = None
        control_threshold = None

    joined_rows: list[dict[str, Any]] = []
    for pair_id in joined_ids:
        base_score = _to_float(baseline[pair_id].get("final_score"))
        bscr_score = _to_float(bscr[pair_id].get("final_score"))
        gate_score = _to_float(gate_off.get(pair_id, {}).get("final_score")) if pair_id in gate_off else None
        if base_score is None or bscr_score is None:
            continue
        row = manifest[pair_id]
        joined_rows.append(
            {
                "pair_id": pair_id,
                "base_regime": row.get("base_regime"),
                "baseline_final": base_score,
                "bscr_delta": bscr_score - base_score,
                "gate_off_delta": gate_score - base_score if gate_score is not None else None,
                "hard_label": hard_threshold is not None and base_score >= hard_threshold,
                "control_label": control_threshold is not None and base_score <= control_threshold,
            }
        )

    per_base: dict[str, dict[str, Any]] = {}
    for regime in BASE_REGIMES:
        selected = [row for row in joined_rows if row["base_regime"] == regime]
        per_base[regime] = {
            "count": len(selected),
            "baseline_final_mean": _mean([row["baseline_final"] for row in selected]),
            "bscr_delta_mean": _mean([row["bscr_delta"] for row in selected]),
            "gate_off_delta_mean": _mean([row["gate_off_delta"] for row in selected if row["gate_off_delta"] is not None]),
            "hard_label_count": sum(1 for row in selected if row["hard_label"]),
            "control_label_count": sum(1 for row in selected if row["control_label"]),
        }
    hard_rows = [row for row in joined_rows if row["hard_label"]]
    control_rows = [row for row in joined_rows if row["control_label"]]
    return {
        "source_join_counts": source_join_counts,
        "joined_pair_count": len(joined_rows),
        "hard_count": len(hard_rows),
        "control_count": len(control_rows),
        "hard_threshold_from_baseline_final": hard_threshold,
        "control_threshold_from_baseline_final": control_threshold,
        "hard_bscr_delta_mean": _mean([row["bscr_delta"] for row in hard_rows]),
        "control_bscr_delta_mean": _mean([row["bscr_delta"] for row in control_rows]),
        "per_base_validation": per_base,
    }


def _load_v1(project_root: Path) -> dict[str, Any]:
    path = project_root / V1_METRICS
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    row_count = data.get("row_count") or 0
    ordinary_count = data.get("state_counts", {}).get("ordinary_control", 0)
    return {
        "available": True,
        "path": str(path),
        "verdict": data.get("verdict"),
        "row_count": row_count,
        "ordinary_control_count": ordinary_count,
        "ordinary_control_fraction": ordinary_count / row_count if row_count else 0.0,
    }


def _coverage(project_root: Path) -> dict[str, Any]:
    feature_rows = {
        name: _count_lines(project_root / rel_path)
        for name, rel_path in FEATURE_PATHS.items()
    }
    identity_rows = {
        name: _count_lines(project_root / rel_path)
        for name, rel_path in IDENTITY_MANIFESTS.items()
    }
    train_fraction = None
    dev_fraction = None
    if identity_rows.get("train"):
        train_fraction = (feature_rows.get("train") or 0) / identity_rows["train"]
    if identity_rows.get("dev"):
        dev_fraction = (feature_rows.get("eval") or 0) / identity_rows["dev"]
    feasible_beyond_eval_packet = (feature_rows.get("train") or 0) > (feature_rows.get("eval") or 0)
    return {
        "feature_rows": feature_rows,
        "identity_rows": identity_rows,
        "train_feature_to_identity_fraction": train_fraction,
        "dev_feature_to_identity_fraction": dev_fraction,
        "feasible_beyond_eval_packet": feasible_beyond_eval_packet,
        "missing_observable_feature_reason": (
            "frozen B-SCR feature packets cover only a small subset of train/dev identity manifests"
            if train_fraction is not None and train_fraction < 0.05
            else ""
        ),
    }


def _verdict(
    rows: list[dict[str, str]],
    audit: dict[str, Any],
    invariant_violations: list[str],
    base_counts: dict[str, int],
    joined: dict[str, Any],
    v1: dict[str, Any],
    coverage: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    row_count = len(rows)
    ordinary_count = base_counts.get("ordinary_control_anchor", 0)
    ordinary_fraction = ordinary_count / row_count if row_count else 0.0
    ordinary_joined = joined.get("per_base_validation", {}).get("ordinary_control_anchor", {}).get("count", 0)
    max_fraction = max((count / row_count for count in base_counts.values()), default=0.0) if row_count else 0.0

    if not audit["passed"]:
        reasons.append("leakage audit failed")
    if invariant_violations:
        reasons.append("base-regime invariant failed")
    if ordinary_fraction < 0.15:
        reasons.append(f"ordinary_control_anchor coverage below 15%: {ordinary_fraction:.4f}")
    if ordinary_joined < 3:
        reasons.append(f"ordinary_control_anchor joined validation count below 3: {ordinary_joined}")
    if max_fraction > 0.90:
        reasons.append(f"base regime collapse: max fraction {max_fraction:.4f}")
    if joined.get("joined_pair_count", 0) == 0:
        reasons.append("no reliable joined validation records")
    if v1.get("available") and ordinary_fraction <= v1.get("ordinary_control_fraction", 0.0):
        reasons.append("v2 does not improve ordinary/control anchor coverage over v1")
    if not coverage.get("feasible_beyond_eval_packet"):
        reasons.append("coverage audit found no non-leaky path beyond eval packet")

    if reasons:
        return "manifest-rejected", reasons
    if coverage.get("train_feature_to_identity_fraction") is not None and coverage["train_feature_to_identity_fraction"] < 0.05:
        return "manifest-weak-inconclusive", ["observable feature coverage is sparse; additional non-leaky feature generation required"]
    return "manifest-ready-for-knowledge-review", ["all v2 bounded manifest gates passed"]


def _write_csv(path: Path, per_base: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_regime",
        "count",
        "baseline_final_mean",
        "bscr_delta_mean",
        "gate_off_delta_mean",
        "hard_label_count",
        "control_label_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime, row in per_base.items():
            out = {"base_regime": regime}
            out.update(row)
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    rows = _read_csv(args.manifest)
    construction_columns = [
        column
        for column in (rows[0].keys() if rows else [])
        if column not in CONSTRUCTION_EXEMPT
        and column not in BASE_FLAG_COLUMNS
        and column not in RISK_TAGS
    ]
    audit = audit_construction_columns_v2(construction_columns)
    invariant_violations = _invariant_violations(rows)
    base_counts, tag_counts, ordinary_with_risk = _base_and_tag_counts(rows)
    base_fractions = {key: (value / len(rows) if rows else 0.0) for key, value in base_counts.items()}
    tag_fractions = {key: (value / len(rows) if rows else 0.0) for key, value in tag_counts.items()}
    joined = _joined_validation(rows, _load_validation(args.project_root))
    v1 = _load_v1(args.project_root)
    coverage = _coverage(args.project_root)
    verdict, reasons = _verdict(rows, audit, invariant_violations, base_counts, joined, v1, coverage)
    ordinary_fraction = base_fractions.get("ordinary_control_anchor", 0.0)
    payload = {
        "schema_version": "phase27_a_manifest_v2_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "row_count": len(rows),
        "leakage_audit": audit,
        "invariant_violations": invariant_violations,
        "base_regime_counts": base_counts,
        "base_regime_fractions": base_fractions,
        "risk_tag_counts": tag_counts,
        "risk_tag_fractions": tag_fractions,
        "ordinary_with_risk_tag_count": ordinary_with_risk,
        "ordinary_control_anchor_fraction": ordinary_fraction,
        "joined_validation": joined,
        "v1_comparison": {
            **v1,
            "v2_ordinary_control_anchor_count": base_counts.get("ordinary_control_anchor", 0),
            "v2_ordinary_control_anchor_fraction": ordinary_fraction,
            "improved_anchor_coverage": ordinary_fraction > v1.get("ordinary_control_fraction", -1.0)
            if v1.get("available")
            else None,
        },
        "coverage_audit": coverage,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "claim_level": "bounded_manifest",
        "training_started": False,
        "submission_created": False,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.metrics_csv, joined["per_base_validation"])
    print(json.dumps({"verdict": verdict, "reasons": reasons, "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
