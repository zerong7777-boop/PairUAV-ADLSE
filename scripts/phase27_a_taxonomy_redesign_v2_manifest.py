"""Build Phase27 A taxonomy redesign-v2 analysis manifest.

Stdlib-only implementation for remote Python environments without pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path


JOIN_KEY = "canonical_pair_id"
ANALYSIS_COLUMNS = [
    "ambiguity_tail_risk_score",
    "low_observable_flag",
    "control_stability_score",
    "validation_status",
    "evidence_sufficiency_score",
    "heading_observability_score",
    "range_observability_score",
    "semantic_geometric_conflict_score",
    "layout_scale_risk_score",
    "match_sufficiency_score",
    "match_sufficiency_source",
    "augmentation_consistency_score",
    "augmentation_consistency_note",
    "baseline_error_score",
    "heading_error_score",
    "range_error_score",
    "stress_sensitivity_score",
    "checkpoint_disagreement_score",
    "tail_outlier_flag",
]


def _read_csv(path: str | Path) -> tuple[list[dict], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_entry(path: str | Path, rows: list[dict], columns: list[str]) -> dict:
    return {"path": str(Path(path)), "rows": len(rows), "columns": columns, "sha256": _sha256(path)}


def _float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _fmt(value: float | None) -> str:
    if value is None:
        value = 0.0
    return f"{max(0.0, min(1.0, float(value))):.6f}"


def _truthy_false(value) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "n", "none"}


def _rank_normalize(values: list[float | None], joined: list[bool] | None = None) -> list[float]:
    joined = joined if joined is not None else [value is not None for value in values]
    valid = sorted({value for value, keep in zip(values, joined) if keep and value is not None})
    if not valid:
        return [0.0 for _ in values]
    if len(valid) == 1:
        return [1.0 if keep and value is not None else 0.0 for value, keep in zip(values, joined)]
    ranks = {value: idx / (len(valid) - 1) for idx, value in enumerate(valid)}
    return [ranks.get(value, 0.0) if keep and value is not None else 0.0 for value, keep in zip(values, joined)]


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        key = row.get(JOIN_KEY, "")
        if key and key not in indexed:
            indexed[key] = row
    return indexed


def _prefixed(row: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in row.items() if key != JOIN_KEY}


def _assign_rows(rows: list[dict]) -> list[dict]:
    rules = importlib.import_module("phase27_a_taxonomy_redesign_v2_rules")
    assigned = rules.assign_rows([dict(row) for row in rows])
    if not isinstance(assigned, list):
        raise TypeError("rules.assign_rows must return a list of row dictionaries")
    for row in assigned:
        missing = {"derived_state", "training_readiness_verdict"} - set(row)
        if missing:
            raise ValueError("rules.assign_rows did not create required columns: " + ", ".join(sorted(missing)))
    return assigned


def build_manifest(
    evidence_manifest: str | Path,
    full_dev_surface: str | Path,
    stress_surfaces: list[str | Path],
) -> tuple[list[dict], dict, dict]:
    evidence, evidence_columns = _read_csv(evidence_manifest)
    full_rows, full_columns = _read_csv(full_dev_surface)
    if any(JOIN_KEY not in row for row in evidence):
        raise ValueError(f"evidence manifest must contain {JOIN_KEY}")

    full_by_id = _index_by_id(full_rows)
    stress_sets = []
    registry = {
        "evidence_manifest": _source_entry(evidence_manifest, evidence, evidence_columns),
        "full_dev_surface": _source_entry(full_dev_surface, full_rows, full_columns),
        "stress_surfaces": [],
    }
    for idx, path in enumerate(stress_surfaces, start=1):
        rows, columns = _read_csv(path)
        stress_sets.append((idx, _index_by_id(rows)))
        registry["stress_surfaces"].append(_source_entry(path, rows, columns))

    joined_rows = []
    full_scores: list[float | None] = []
    heading_errors: list[float | None] = []
    range_errors: list[float | None] = []
    stress_means: list[float | None] = []
    stress_spreads: list[float | None] = []
    full_joined_flags: list[bool] = []
    stress_joined_flags: list[bool] = []

    for row in evidence:
        out = dict(row)
        key = row.get(JOIN_KEY, "")
        source_split = row.get("source_split", "")
        group_id = row.get("group_id", "")
        out["source_image_key"] = row.get("image_a", "")
        out["target_image_key"] = row.get("image_b", "")
        out["target_key"] = group_id
        out["scene_key"] = row.get("scene", "") or group_id or (f"{source_split}/{group_id}" if source_split or group_id else "")
        out["split_key"] = source_split
        out["key_schema_version"] = "phase27_a_taxonomy_redesign_v2"
        out["old_base_regime"] = row.get("base_regime", "")

        full = full_by_id.get(key)
        full_joined = full is not None
        out["full_dev_joined"] = str(full_joined)
        if full:
            out.update(_prefixed(full, "full_dev"))

        stress_values = []
        for idx, stress_by_id in stress_sets:
            stress = stress_by_id.get(key)
            if stress:
                out.update(_prefixed(stress, f"stress{idx}"))
                value = _float(stress.get("baseline_final_score"))
                if value is not None:
                    stress_values.append(value)
        stress_joined = bool(stress_values)
        out["stress_joined"] = str(stress_joined)

        obs = _float(row.get("observability_axis"))
        similarity = _float(row.get("pair_similarity_axis"))
        scale_risk = _float(row.get("scale_risk_axis"))
        layout_risk = _float(row.get("layout_risk_axis"))
        conflict = _float(row.get("conflict_risk_axis"))
        centrality = _float(row.get("control_centrality_score"))
        layout_scale_risk = max([value for value in [scale_risk, layout_risk] if value is not None], default=0.0)
        evidence_score = _mean([obs, similarity])
        if evidence_score is None:
            evidence_score = 0.0
        range_obs = obs if obs is not None else 0.0
        range_obs *= 1.0 - max(0.0, min(1.0, layout_scale_risk))

        out["evidence_sufficiency_score"] = _fmt(evidence_score)
        out["heading_observability_score"] = _fmt(obs)
        out["range_observability_score"] = _fmt(range_obs)
        out["semantic_geometric_conflict_score"] = _fmt(conflict)
        out["layout_scale_risk_score"] = _fmt(layout_scale_risk)
        out["match_sufficiency_score"] = _fmt(similarity)
        out["match_sufficiency_source"] = "MATCHER_SIDE_NON_LEAKING"
        out["augmentation_consistency_score"] = _fmt(centrality)
        out["augmentation_consistency_note"] = "proxy_from_control_centrality_score"
        low_reason = bool(str(row.get("low_observable_reason", "")).strip())
        observable_bad = _truthy_false(row.get("observable_adequate"))
        adequacy_bad = _truthy_false(row.get("adequacy_passed"))
        out["low_observable_flag"] = str(bool(low_reason or observable_bad or adequacy_bad))

        full_scores.append(_float(full.get("baseline_final_score")) if full else None)
        heading_errors.append(_float(full.get("baseline_angle_rel_error")) if full else None)
        range_errors.append(_float(full.get("baseline_distance_rel_error")) if full else None)
        stress_means.append(sum(stress_values) / len(stress_values) if stress_values else None)
        stress_spreads.append(max(stress_values) - min(stress_values) if len(stress_values) >= 2 else (0.0 if stress_values else None))
        full_joined_flags.append(full_joined)
        stress_joined_flags.append(stress_joined)
        joined_rows.append(out)

    baseline_scores = _rank_normalize(full_scores, full_joined_flags)
    heading_scores = _rank_normalize(heading_errors, full_joined_flags)
    range_scores = _rank_normalize(range_errors, full_joined_flags)
    stress_scores = _rank_normalize(stress_means, stress_joined_flags)
    disagreement_scores = _rank_normalize(stress_spreads, stress_joined_flags)

    risk_values = []
    for idx, out in enumerate(joined_rows):
        out["baseline_error_score"] = _fmt(baseline_scores[idx])
        out["heading_error_score"] = _fmt(heading_scores[idx])
        out["range_error_score"] = _fmt(range_scores[idx])
        out["stress_sensitivity_score"] = _fmt(stress_scores[idx])
        out["checkpoint_disagreement_score"] = _fmt(disagreement_scores[idx])
        risk_values.append(max(baseline_scores[idx], heading_scores[idx], range_scores[idx], stress_scores[idx]))

    threshold = sorted(risk_values)[max(0, int(math.ceil(len(risk_values) * 0.95)) - 1)] if risk_values else 1.0
    for idx, out in enumerate(joined_rows):
        tail_outlier = bool(risk_values[idx] > 0.0 and risk_values[idx] >= threshold)
        conflict = _float(out.get("semantic_geometric_conflict_score")) or 0.0
        tail_floor = 1.0 if tail_outlier else 0.0
        out["tail_outlier_flag"] = str(tail_outlier)
        out["ambiguity_tail_risk_score"] = _fmt(max(conflict, risk_values[idx], tail_floor))
        centrality = _float(out.get("augmentation_consistency_score")) or 0.0
        inverse_risk = 1.0 - max(risk_values[idx], disagreement_scores[idx])
        out["control_stability_score"] = _fmt(_mean([centrality, inverse_risk]))
        full_joined = full_joined_flags[idx]
        stress_joined = stress_joined_flags[idx]
        if full_joined and stress_joined:
            status = "joined_full_and_stress"
        elif full_joined:
            status = "joined_full_only"
        elif stress_joined:
            status = "joined_stress_only"
        else:
            status = "unvalidated"
        out["validation_status"] = status

    assigned = _assign_rows(joined_rows)
    audit = {
        "join_keys": [JOIN_KEY],
        "evidence_rows": len(evidence),
        "manifest_rows": len(assigned),
        "full_dev_rows": len(full_rows),
        "stress_surface_count": len(stress_surfaces),
        "unmatched_evidence_rows": sum(1 for flag in full_joined_flags if not flag),
        "full_dev_joined_fraction": sum(1 for flag in full_joined_flags if flag) / len(full_joined_flags) if full_joined_flags else 0.0,
        "stress_joined_fraction": sum(1 for flag in stress_joined_flags if flag) / len(stress_joined_flags) if stress_joined_flags else 0.0,
        "duplicate_evidence_ids": len(evidence) - len({row.get(JOIN_KEY, "") for row in evidence}),
        "analysis_columns": ANALYSIS_COLUMNS,
        "leakage_checks": {
            "preserved_all_evidence_rows": len(evidence) == len(assigned),
            "match_sufficiency_source": "MATCHER_SIDE_NON_LEAKING",
            "augmentation_consistency_note": "proxy_from_control_centrality_score",
            "training_policy_spec_emitted": False,
        },
    }
    return assigned, registry, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--full-dev-surface", required=True)
    parser.add_argument("--stress-surface", action="append", default=[])
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-source-registry", required=True)
    parser.add_argument("--out-leakage-audit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest, registry, audit = build_manifest(args.evidence_manifest, args.full_dev_surface, args.stress_surface)
    _write_csv(args.out_manifest, manifest)
    _write_json(args.out_source_registry, registry)
    _write_json(args.out_leakage_audit, audit)


if __name__ == "__main__":
    main()
