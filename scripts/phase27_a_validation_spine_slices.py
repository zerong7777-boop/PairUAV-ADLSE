"""Slice registry and metrics for Phase27 A validation spine."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SLICE_REGISTRY = {
    "ordinary_control_anchor": {"field": "evidence_state", "op": "eq", "value": "ordinary_control_anchor", "role": "control"},
    "hard_trainable": {"field": "evidence_state", "op": "eq", "value": "hard_trainable", "role": "hard"},
    "semantic_geometry_conflict": {"field": "semantic_geometry_conflict_score", "op": "ge", "value": 0.7, "role": "mechanism_probe"},
    "target_heterogeneous": {"field": "evidence_state", "op": "eq", "value": "target_heterogeneous", "role": "mechanism_probe"},
    "low_observable": {"field": "evidence_state", "op": "eq", "value": "low_observable", "role": "quarantine_candidate"},
    "ambiguous_unreliable": {"field": "evidence_state", "op": "eq", "value": "ambiguous_unreliable", "role": "separate_policy_candidate"},
    "unknown": {"field": "evidence_state", "op": "eq", "value": "unknown", "role": "quarantine"},
}


def _float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def row_in_slice(row, slice_name_or_def):
    slice_def = DEFAULT_SLICE_REGISTRY[slice_name_or_def] if isinstance(slice_name_or_def, str) else slice_name_or_def
    actual = row.get(slice_def["field"], "")
    op = slice_def["op"]
    expected = slice_def["value"]
    if op == "eq":
        return str(actual) == str(expected)
    if op == "ge":
        parsed = _float(actual)
        return parsed is not None and parsed >= float(expected)
    raise ValueError(f"Unsupported slice op: {op}")


def _mean(values):
    return sum(values) / len(values) if values else None


def _variance(values):
    if not values:
        return None
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def compute_slice_metrics(rows, registry=DEFAULT_SLICE_REGISTRY):
    total = len(rows)
    metrics = {}
    for name, slice_def in registry.items():
        selected = [row for row in rows if row_in_slice(row, slice_def)]
        scores = [value for value in (_float(row.get("baseline_final_score")) for row in selected) if value is not None]
        matcher_scores = [value for value in (_float(row.get("matcher_sufficiency_score")) for row in selected) if value is not None]
        metrics[name] = {
            "row_count": len(selected),
            "fraction": len(selected) / total if total else 0.0,
            "baseline_final_score_mean": _mean(scores),
            "baseline_final_score_variance": _variance(scores),
            "matcher_sufficiency_mean": _mean(matcher_scores),
            "reference_overlap_count": sum(1 for row in selected if _truthy(row.get("reference_overlap"))),
            "baseline_overlap_count": sum(1 for row in selected if _truthy(row.get("baseline_overlap"))),
            "role": slice_def.get("role", ""),
        }
    return metrics


def write_slice_registry(path, registry=DEFAULT_SLICE_REGISTRY):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False, sort_keys=True)
