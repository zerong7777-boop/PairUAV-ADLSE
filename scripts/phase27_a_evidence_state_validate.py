#!/usr/bin/env python3
"""Validate Phase27 A evidence-state manifest without training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.phase27_a_evidence_state_schema import STATE_COLUMNS, audit_construction_columns


VALIDATION_SOURCES = {
    "baseline_matched": "experiments/phase26_b1_geometry_local_alignment/eval/baseline_matched/official_per_sample.csv",
    "b_scr": "experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_heading_selective/official_per_sample.csv",
    "b_scr_gate_off": "experiments/phase26_b_selective_correspondence_reasoning/eval/b_scr_gate_off/official_per_sample.csv",
    "b1_frozen_matcher": "experiments/phase26_b1_frozen_external_matcher/eval/frozen_matcher_fusion/official_per_sample.csv",
}

SLICE_SOURCES = {
    "b_scr_slice": "experiments/phase26_b_selective_correspondence_reasoning/slice_metrics/slice_metrics.csv",
    "b_scr_gate_off_slice": "experiments/phase26_b_selective_correspondence_reasoning/slice_metrics_gate_off/slice_metrics.csv",
    "b1_frozen_matcher_slice": "experiments/phase26_b1_frozen_external_matcher/slice_metrics/slice_metrics.csv",
}

CONSTRUCTION_EXEMPT_COLUMNS = {
    "pair_id",
    "query_name",
    "reference_name",
    "construction_source",
    "schema_version",
    "leakage_audit_passed",
    "state_count",
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


def _state_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        state: sum(1 for row in rows if str(row.get(state, "0")).strip() in {"1", "1.0", "true", "True"})
        for state in STATE_COLUMNS
    }


def _overlap_matrix(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {state: {other: 0 for other in STATE_COLUMNS} for state in STATE_COLUMNS}
    for row in rows:
        active = [state for state in STATE_COLUMNS if str(row.get(state, "0")).strip() in {"1", "1.0", "true", "True"}]
        for left in active:
            for right in active:
                matrix[left][right] += 1
    return matrix


def _load_validation(project_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    loaded: dict[str, dict[str, dict[str, str]]] = {}
    for name, rel_path in VALIDATION_SOURCES.items():
        rows = _read_csv(project_root / rel_path)
        loaded[name] = {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}
    return loaded


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _joined_metrics(
    manifest_rows: list[dict[str, str]], validation: dict[str, dict[str, dict[str, str]]]
) -> dict[str, Any]:
    manifest_by_id = {row["pair_id"]: row for row in manifest_rows if row.get("pair_id")}
    source_join_counts = {
        name: sum(1 for pair_id in manifest_by_id if pair_id in rows)
        for name, rows in validation.items()
    }
    baseline = validation.get("baseline_matched", {})
    bscr = validation.get("b_scr", {})
    gate_off = validation.get("b_scr_gate_off", {})
    joined_ids = sorted(set(manifest_by_id) & set(baseline) & set(bscr))
    hard_control_rows: list[dict[str, Any]] = []
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
    for pair_id in joined_ids:
        base_score = _to_float(baseline[pair_id].get("final_score"))
        bscr_score = _to_float(bscr[pair_id].get("final_score"))
        gate_score = _to_float(gate_off.get(pair_id, {}).get("final_score")) if pair_id in gate_off else None
        if base_score is None or bscr_score is None:
            continue
        manifest_row = manifest_by_id[pair_id]
        active_states = [
            state
            for state in STATE_COLUMNS
            if str(manifest_row.get(state, "0")).strip() in {"1", "1.0", "true", "True"}
        ]
        hard_label = hard_threshold is not None and base_score >= hard_threshold
        control_label = control_threshold is not None and base_score <= control_threshold
        hard_control_rows.append(
            {
                "pair_id": pair_id,
                "baseline_final": base_score,
                "bscr_final": bscr_score,
                "gate_off_final": gate_score,
                "delta_bscr_minus_baseline": bscr_score - base_score,
                "delta_gate_off_minus_baseline": gate_score - base_score if gate_score is not None else None,
                "validation_hard_label": hard_label,
                "validation_control_label": control_label,
                "active_states": active_states,
            }
        )
    per_state: dict[str, dict[str, Any]] = {}
    for state in STATE_COLUMNS:
        selected = [row for row in hard_control_rows if state in row["active_states"]]
        per_state[state] = {
            "count": len(selected),
            "baseline_final_mean": _mean([row["baseline_final"] for row in selected]),
            "bscr_delta_mean": _mean([row["delta_bscr_minus_baseline"] for row in selected]),
            "gate_off_delta_mean": _mean(
                [row["delta_gate_off_minus_baseline"] for row in selected if row["delta_gate_off_minus_baseline"] is not None]
            ),
            "hard_label_count": sum(1 for row in selected if row["validation_hard_label"]),
            "control_label_count": sum(1 for row in selected if row["validation_control_label"]),
        }
    hard_rows = [row for row in hard_control_rows if row["validation_hard_label"]]
    control_rows = [row for row in hard_control_rows if row["validation_control_label"]]
    return {
        "source_join_counts": source_join_counts,
        "joined_pair_count": len(joined_ids),
        "hard_threshold_from_baseline_final": hard_threshold,
        "control_threshold_from_baseline_final": control_threshold,
        "hard_count": len(hard_rows),
        "control_count": len(control_rows),
        "hard_bscr_delta_mean": _mean([row["delta_bscr_minus_baseline"] for row in hard_rows]),
        "control_bscr_delta_mean": _mean([row["delta_bscr_minus_baseline"] for row in control_rows]),
        "per_state_validation": per_state,
        "naive_semantic_proxy_available": False,
        "naive_semantic_proxy_comparison": "not_available_in_current_non_leaky_feature packet",
    }


def _slice_summary(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rel_path in SLICE_SOURCES.items():
        rows = _read_csv(project_root / rel_path)
        out[name] = {
            "path": rel_path,
            "row_count": len(rows),
            "slices": [row.get("slice") for row in rows],
            "rows": rows,
        }
    return out


def _verdict(row_count: int, audit: dict[str, Any], counts: dict[str, int], joined: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not audit["passed"]:
        reasons.append("forbidden construction columns present")
    ordinary_fraction = counts.get("ordinary_control", 0) / row_count if row_count else 0.0
    if ordinary_fraction < 0.10:
        reasons.append(f"ordinary/control coverage below 10%: {ordinary_fraction:.4f}")
    max_fraction = max((count / row_count for count in counts.values()), default=0.0) if row_count else 0.0
    non_optional_nonzero = sum(
        1
        for state in [
            "ordinary_control",
            "high_evidence_anchor",
            "low_observable",
            "heading_risk",
            "range_risk",
            "ambiguous_scale",
        ]
        if counts.get(state, 0) > 0
    )
    if max_fraction > 0.90 and non_optional_nonzero <= 2:
        reasons.append("state coverage collapsed")
    if joined.get("joined_pair_count", 0) == 0:
        reasons.append("no reliable joined hard/control validation records")
    if row_count == 0:
        reasons.append("manifest has zero rows")
    if reasons:
        return "manifest-rejected", reasons
    if joined.get("joined_pair_count", 0) < row_count:
        reasons.append("partial validation join coverage")
        return "manifest-weak-inconclusive", reasons
    return "manifest-ready-for-knowledge-review", ["all bounded manifest gates passed"]


def _write_metrics_csv(path: Path, per_state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state",
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
        for state, metrics in per_state.items():
            row = {"state": state}
            row.update(metrics)
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    manifest_rows = _read_csv(args.manifest)
    construction_columns = [
        column
        for column in (manifest_rows[0].keys() if manifest_rows else [])
        if column not in STATE_COLUMNS and column not in CONSTRUCTION_EXEMPT_COLUMNS
    ]
    audit = audit_construction_columns(construction_columns)
    counts = _state_counts(manifest_rows)
    fractions = {state: (count / len(manifest_rows) if manifest_rows else 0.0) for state, count in counts.items()}
    joined = _joined_metrics(manifest_rows, _load_validation(args.project_root))
    verdict, reasons = _verdict(len(manifest_rows), audit, counts, joined)
    payload = {
        "schema_version": "phase27_a_manifest_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "row_count": len(manifest_rows),
        "leakage_audit": audit,
        "state_counts": counts,
        "state_fractions": fractions,
        "state_overlap_matrix": _overlap_matrix(manifest_rows),
        "joined_validation": joined,
        "slice_validation_context": _slice_summary(args.project_root),
        "bscr_failure_pattern_explanation": (
            "Validation uses frozen post-schema states joined to B-SCR/gate-off per-sample records. "
            "Slice-level Phase26 evidence remains validation-only and is not used for state construction."
        ),
        "verdict": verdict,
        "verdict_reasons": reasons,
        "claim_level": "bounded_manifest",
        "training_started": False,
        "submission_created": False,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics_csv(args.metrics_csv, joined["per_state_validation"])
    print(json.dumps({"verdict": verdict, "reasons": reasons, "rows": len(manifest_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
