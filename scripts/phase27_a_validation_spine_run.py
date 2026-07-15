#!/usr/bin/env python3
"""CLI runner for Phase27 A validation spine."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.phase27_a_validation_spine_keys import attach_canonical_keys, audit_overlap
from scripts.phase27_a_validation_spine_manifest import make_shadow_rows, read_csv_rows, write_csv_rows
from scripts.phase27_a_validation_spine_registry import artifact_entry, validate_registry, write_registry
from scripts.phase27_a_validation_spine_slices import DEFAULT_SLICE_REGISTRY, compute_slice_metrics, write_slice_registry
from scripts.phase27_a_validation_spine_suites import (
    control_stability_suite,
    identity_suite,
    leakage_suite,
    lineage_suite,
    state_distribution_suite,
    state_error_association_suite,
    training_readiness_suite,
)


def _write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _read_csv_optional(path):
    if not path:
        return []
    candidate = Path(path)
    if not candidate.exists():
        return []
    return read_csv_rows(candidate)


def _normalize_evidence_rows(rows):
    normalized = []
    for row in rows:
        out = dict(row)
        if not out.get("evidence_state"):
            out["evidence_state"] = out.get("base_regime", "unknown")
        if not out.get("semantic_geometry_conflict_score"):
            out["semantic_geometry_conflict_score"] = out.get("conflict_risk_axis", "")
        if not out.get("matcher_sufficiency_score"):
            out["matcher_sufficiency_score"] = out.get("observability_axis", "")
        if not out.get("observability_score"):
            out["observability_score"] = out.get("observability_axis", "")
        if not out.get("canonical_pair_id"):
            out = attach_canonical_keys(out)
        normalized.append(out)
    return normalized


def _normalize_reference_rows(rows):
    normalized = []
    for row in rows:
        out = dict(row)
        if not out.get("canonical_pair_id") and out.get("sample_id"):
            out["canonical_pair_id"] = out["sample_id"]
        normalized.append(out)
    return normalized


def _normalize_baseline_rows(rows):
    normalized = []
    for row in rows:
        out = dict(row)
        if not out.get("canonical_pair_id") and out.get("sample_id") and "/" in str(out.get("sample_id")):
            out["canonical_pair_id"] = out["sample_id"]
        if not out.get("baseline_final_score") and out.get("final_score"):
            out["baseline_final_score"] = out["final_score"]
        normalized.append(out)
    return normalized


def _synthetic_rows():
    evidence = [
        attach_canonical_keys({
            "namespace": "smoke",
            "group_id": "g",
            "image_a_name": "image-01",
            "image_b_name": "image-02",
            "evidence_state": "ordinary_control_anchor",
            "observability_score": "0.9",
            "semantic_geometry_conflict_score": "0.1",
            "matcher_sufficiency_score": "0.8",
        }),
        attach_canonical_keys({
            "namespace": "smoke",
            "group_id": "g",
            "image_a_name": "image-03",
            "image_b_name": "image-04",
            "evidence_state": "hard_trainable",
            "observability_score": "0.7",
            "semantic_geometry_conflict_score": "0.8",
            "matcher_sufficiency_score": "0.6",
        }),
    ]
    baseline = [
        {"canonical_pair_id": evidence[0]["canonical_pair_id"], "baseline_final_score": "0.2", "baseline_overlap": "1"},
        {"canonical_pair_id": evidence[1]["canonical_pair_id"], "baseline_final_score": "0.7", "baseline_overlap": "1"},
    ]
    matcher = [
        {"canonical_pair_id": evidence[0]["canonical_pair_id"], "matcher_sufficiency_score": "0.8"},
        {"canonical_pair_id": evidence[1]["canonical_pair_id"], "matcher_sufficiency_score": "0.6"},
    ]
    reference = [
        {"canonical_pair_id": evidence[0]["canonical_pair_id"], "reference_overlap": "1"},
    ]
    return evidence, baseline, matcher, reference


def _state_fractions(rows):
    total = len(rows)
    counts = {}
    for row in rows:
        state = row.get("evidence_state", "unknown")
        counts[state] = counts.get(state, 0) + 1
    return {
        "ordinary": counts.get("ordinary_control_anchor", 0) / total if total else 0.0,
        "hard": counts.get("hard_trainable", 0) / total if total else 0.0,
        "unknown": counts.get("unknown", 0) / total if total else 0.0,
    }


def _score_map(rows):
    by_state = {}
    for row in rows:
        state = row.get("evidence_state", "unknown")
        score = row.get("baseline_final_score")
        if score in ("", None):
            continue
        try:
            by_state.setdefault(state, []).append(float(score))
        except ValueError:
            continue
    def mean(values):
        return sum(values) / len(values) if values else None
    return {
        "ordinary": mean(by_state.get("ordinary_control_anchor", [])),
        "hard": mean(by_state.get("hard_trainable", [])),
        "ambiguous": mean(by_state.get("ambiguous_unreliable", [])),
    }


def run_smoke(args):
    out_dir = Path(args.out_dir)
    evidence, baseline, matcher, reference = _synthetic_rows()
    shadow = make_shadow_rows(evidence, baseline, matcher_rows=matcher, reference_rows=reference)
    rows = shadow["rows"]
    for row in rows:
        row["baseline_overlap"] = "1"
        row["reference_overlap"] = "1" if row["canonical_pair_id"] == reference[0]["canonical_pair_id"] else "0"

    registry_path = out_dir / "registries" / "artifacts.json"
    slices_path = out_dir / "registries" / "slices.json"
    manifest_path = out_dir / "manifests" / "shadow_validation_manifest.csv"
    metrics_path = out_dir / "metrics" / "combined_validation_metrics.json"
    report_path = out_dir / "reports" / "smoke_report.md"

    registry = {
        "synthetic_evidence": artifact_entry("synthetic_evidence", "evidence_manifest", "synthetic", "evidence_v1", "pair_key_v1", len(evidence), evidence[0].keys(), [], "phase27_a_validation_spine_run.py --mode smoke", False),
        "synthetic_baseline": artifact_entry("synthetic_baseline", "baseline_prediction", "synthetic", "baseline_v1", "pair_key_v1", len(baseline), baseline[0].keys(), [], "phase27_a_validation_spine_run.py --mode smoke", False),
        "shadow_validation_manifest": artifact_entry("shadow_validation_manifest", "shadow_validation_surface", str(manifest_path), "shadow_v1", "pair_key_v1", len(rows), rows[0].keys(), ["synthetic_evidence", "synthetic_baseline"], "phase27_a_validation_spine_run.py --mode smoke", False),
    }
    lineage = lineage_suite(validate_registry(registry))
    identity_metrics = {
        "evidence_to_baseline": audit_overlap("evidence", evidence, "baseline", baseline),
        "evidence_to_reference": {"canonical_overlap": 0, "failure_classification": "reference_too_small"},
    }
    identity = identity_suite(identity_metrics["evidence_to_baseline"], identity_metrics["evidence_to_reference"])
    leakage = leakage_suite({"observability_score": "0.9", "evidence_state": "ordinary_control_anchor"})
    distribution = state_distribution_suite(_state_fractions(rows))
    state_error = state_error_association_suite(_score_map(rows))
    control = control_stability_suite(_score_map(rows))
    readiness = training_readiness_suite(
        identity=identity,
        lineage=lineage,
        leakage=leakage,
        distribution=distribution,
        state_error=state_error,
        control=control,
    )
    slice_metrics = compute_slice_metrics(rows)
    combined = {
        "mode": "smoke",
        "smoke_verdict": "mechanics-only-pass",
        "identity": identity,
        "lineage": lineage,
        "leakage": leakage,
        "state_distribution": distribution,
        "state_error_association": state_error,
        "control_stability": control,
        "training_readiness": readiness,
        "slice_metrics": slice_metrics,
        "shadow_unmatched": {k: v for k, v in shadow.items() if k != "rows"},
    }

    write_registry(registry_path, registry)
    write_slice_registry(slices_path, DEFAULT_SLICE_REGISTRY)
    write_csv_rows(manifest_path, rows)
    _write_json(metrics_path, combined)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Phase27 A Validation Spine Smoke Report",
            "",
            "Smoke verdict: mechanics-only-pass",
            "This does not validate A training readiness.",
            "",
            f"Training readiness diagnostic verdict: `{readiness['verdict']}`",
            f"Shadow rows: {len(rows)}",
            f"Output metrics: `{metrics_path}`",
            "",
        ]),
        encoding="utf-8",
    )
    return 0


def _write_bounded_report(path, combined, out_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = combined["training_readiness"]["verdict"]
    blocker = combined.get("blocker", "")
    path.write_text(
        "\n".join([
            "# Phase27 A Validation Spine Bounded Full Eval Report",
            "",
            f"- Verdict: `{readiness}`",
            f"- Blocker: `{blocker}`",
            f"- Evidence rows: {combined['row_counts']['evidence_rows']}",
            f"- Baseline rows: {combined['row_counts']['baseline_rows']}",
            f"- Shadow rows: {combined['row_counts']['shadow_rows']}",
            f"- Reference rows: {combined['row_counts']['reference_rows']}",
            f"- Knowledge-review required: {combined['knowledge_review_required']}",
            "",
            "## Interpretation",
            "",
            "This bounded eval uses existing read-only artifacts. It does not train, finetune, change inference, or submit results.",
            "If baseline rows are missing or unjoinable, state-error association and control stability remain unresolved.",
            "",
            f"Metrics directory: `{out_dir / 'metrics'}`",
            "",
        ]),
        encoding="utf-8",
    )


def run_bounded(args):
    out_dir = Path(args.out_dir)
    evidence = _normalize_evidence_rows(_read_csv_optional(args.evidence_manifest))
    baseline = _normalize_baseline_rows(_read_csv_optional(args.baseline_predictions))
    matcher = _read_csv_optional(args.matcher_cache_index)
    reference = _normalize_reference_rows(_read_csv_optional(args.reference_surface))

    if not evidence:
        raise SystemExit("bounded mode requires a readable evidence manifest")

    if baseline:
        shadow = make_shadow_rows(evidence, baseline, matcher_rows=matcher, reference_rows=reference)
        shadow_rows = shadow["rows"]
    else:
        shadow = {
            "rows": [],
            "unmatched_evidence_count": len(evidence),
            "unmatched_baseline_count": 0,
        }
        shadow_rows = []

    registry_path = out_dir / "registries" / "artifacts.json"
    slices_path = out_dir / "registries" / "slices.json"
    manifest_path = out_dir / "manifests" / "shadow_validation_manifest.csv"
    report_path = out_dir / "reports" / "bounded_full_eval_report.md"

    registry = {
        "evidence_manifest": artifact_entry("evidence_manifest", "evidence_manifest", args.evidence_manifest, "evidence_manifest_v1", "pair_key_v1", len(evidence), evidence[0].keys(), [], "phase27_a_validation_spine_run.py --mode bounded", True),
        "shadow_validation_manifest": artifact_entry("shadow_validation_manifest", "shadow_validation_surface", str(manifest_path), "shadow_v1", "pair_key_v1", len(shadow_rows), shadow_rows[0].keys() if shadow_rows else ["canonical_pair_id"], ["evidence_manifest"], "phase27_a_validation_spine_run.py --mode bounded", False),
    }
    if args.baseline_predictions:
        registry["baseline_predictions"] = artifact_entry("baseline_predictions", "baseline_prediction", args.baseline_predictions, "baseline_v1", "pair_key_v1", len(baseline), baseline[0].keys() if baseline else [], [], "read-only", True, notes="missing_or_unjoinable" if not baseline else "")
        registry["shadow_validation_manifest"]["source_artifacts"].append("baseline_predictions")
    if args.reference_surface:
        registry["reference_surface"] = artifact_entry("reference_surface", "reference_surface", args.reference_surface, "reference_v1", "pair_key_v1", len(reference), reference[0].keys() if reference else [], [], "read-only", True, notes="optional")

    lineage = lineage_suite(validate_registry(registry))
    if baseline:
        evidence_to_baseline = audit_overlap("evidence", evidence, "baseline", baseline)
    else:
        evidence_to_baseline = {"canonical_overlap": 0, "duplicates": 0, "collisions": 0, "failure_classification": "baseline_prediction_missing_or_unjoinable"}
    evidence_to_reference = audit_overlap("evidence", evidence, "reference", reference) if reference else {"canonical_overlap": 0, "failure_classification": "reference_too_small"}

    identity = identity_suite(evidence_to_baseline, evidence_to_reference)
    leakage = leakage_suite({"observability_score": "deployable", "evidence_state": "deployable"})
    distribution = state_distribution_suite(_state_fractions(evidence))
    state_error = state_error_association_suite(_score_map(shadow_rows))
    control = control_stability_suite(_score_map(shadow_rows))
    readiness = training_readiness_suite(
        identity=identity,
        lineage=lineage,
        leakage=leakage,
        distribution=distribution,
        state_error=state_error,
        control=control,
    )
    blocker = "" if baseline else "baseline_prediction_missing_or_unjoinable"
    if blocker:
        readiness["passed"] = False
        readiness["verdict"] = "A-validation-spine-needs-redesign"
        readiness["failed_suites"] = sorted(set(readiness.get("failed_suites", []) + ["baseline_prediction"]))

    slice_metrics = compute_slice_metrics(shadow_rows if shadow_rows else evidence)
    combined = {
        "mode": "bounded",
        "blocker": blocker,
        "row_counts": {
            "evidence_rows": len(evidence),
            "baseline_rows": len(baseline),
            "matcher_rows": len(matcher),
            "reference_rows": len(reference),
            "shadow_rows": len(shadow_rows),
        },
        "identity": identity,
        "identity_overlap_metrics": {
            "evidence_to_baseline": evidence_to_baseline,
            "evidence_to_reference": evidence_to_reference,
        },
        "lineage": lineage,
        "leakage": leakage,
        "state_distribution": distribution,
        "state_error_association": state_error,
        "control_stability": control,
        "training_readiness": readiness,
        "slice_metrics": slice_metrics,
        "shadow_unmatched": {k: v for k, v in shadow.items() if k != "rows"},
        "knowledge_review_required": True,
    }

    write_registry(registry_path, registry)
    write_slice_registry(slices_path, DEFAULT_SLICE_REGISTRY)
    write_csv_rows(manifest_path, shadow_rows)
    _write_json(out_dir / "metrics" / "identity_metrics.json", combined["identity_overlap_metrics"])
    _write_json(out_dir / "metrics" / "lineage_metrics.json", lineage)
    _write_json(out_dir / "metrics" / "leakage_metrics.json", leakage)
    _write_json(out_dir / "metrics" / "state_distribution_metrics.json", distribution)
    _write_json(out_dir / "metrics" / "state_error_association_metrics.json", state_error)
    _write_json(out_dir / "metrics" / "matcher_sufficiency_metrics.json", {"passed": bool(matcher), "matcher_rows": len(matcher)})
    _write_json(out_dir / "metrics" / "control_stability_metrics.json", control)
    _write_json(out_dir / "metrics" / "training_readiness_metrics.json", readiness)
    _write_json(out_dir / "metrics" / "combined_validation_metrics.json", combined)
    _write_bounded_report(report_path, combined, out_dir)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "bounded"), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--evidence-manifest")
    parser.add_argument("--baseline-predictions")
    parser.add_argument("--matcher-cache-index")
    parser.add_argument("--reference-surface")
    parser.add_argument("--calibration-metrics")
    args = parser.parse_args()
    if args.mode == "smoke":
        return run_smoke(args)
    return run_bounded(args)


if __name__ == "__main__":
    raise SystemExit(main())
