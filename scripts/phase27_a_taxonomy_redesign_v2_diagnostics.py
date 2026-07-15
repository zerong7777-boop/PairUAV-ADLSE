"""Failure attribution diagnostics for Phase27 A taxonomy redesign-v2.

This script is read-only with respect to model/training artifacts. It consumes
the existing A-v2 manifest and source-category rules, then writes diagnostics
that explain why `evidence_sufficient_hard` is empty.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


STATE_ORDER = [
    "stable_control_anchor",
    "ambiguous_unreliable",
    "unknown_unvalidated",
    "low_observable",
    "stress_sensitive_control",
    "conflict_candidate",
    "evidence_sufficient_hard",
]

SCORE_FIELDS = [
    "evidence_sufficiency_score",
    "heading_observability_score",
    "range_observability_score",
    "match_sufficiency_score",
    "semantic_geometric_conflict_score",
    "baseline_error_score",
    "heading_error_score",
    "range_error_score",
    "stress_sensitivity_score",
    "checkpoint_disagreement_score",
    "ambiguity_tail_risk_score",
    "control_stability_score",
]


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: str | Path, rows: list[dict]) -> None:
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


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(str(row.get(key, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def has_value(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip() != ""


def b(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes", "y"}


def raw_float(row: dict, key: str) -> float | None:
    text = str(row.get(key, "")).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summary(values: list[float]) -> dict:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "count": len(finite),
        "missing_count": len(values) - len(finite),
        "min": min(finite) if finite else None,
        "p5": quantile(finite, 0.05),
        "p25": quantile(finite, 0.25),
        "p50": quantile(finite, 0.50),
        "p75": quantile(finite, 0.75),
        "p90": quantile(finite, 0.90),
        "p95": quantile(finite, 0.95),
        "p99": quantile(finite, 0.99),
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def flags(row: dict, t: dict) -> dict:
    evidence = f(row, "evidence_sufficiency_score")
    match = f(row, "match_sufficiency_score")
    heading = f(row, "heading_observability_score")
    range_obs = f(row, "range_observability_score")
    observability = min(heading, range_obs)
    conflict = f(row, "semantic_geometric_conflict_score")
    baseline = f(row, "baseline_error_score")
    heading_err = f(row, "heading_error_score")
    range_err = f(row, "range_error_score")
    stress = f(row, "stress_sensitivity_score")
    tail = f(row, "ambiguity_tail_risk_score")
    stability = f(row, "control_stability_score")
    required_present = all(has_value(row, key) for key in getattr(flags, "required_axis_fields"))
    identity_present = has_value(row, "canonical_pair_id")
    low_observable = observability <= t["observability_low"]
    evidence_sufficient = evidence >= t["evidence_sufficiency_high"] and match >= t["evidence_sufficiency_high"]
    evidence_sufficient_loose = evidence >= t["evidence_sufficiency_high"]
    baseline_difficult = baseline >= t["baseline_difficulty_high"]
    heading_difficult = heading_err >= t["baseline_difficulty_high"]
    range_difficult = range_err >= t["baseline_difficulty_high"]
    stress_sensitive = stress >= t["stress_sensitivity_high"]
    ambiguous_primary = evidence <= t["evidence_sufficiency_low"] or tail >= t["tail_risk_high"]
    conflict_high = conflict >= t["conflict_high"]
    control_low = stability <= t["control_stability_low"]
    stable_control = (
        evidence >= t["evidence_sufficiency_high"]
        and match >= t["evidence_sufficiency_high"]
        and stability >= t["control_stability_high"]
    )
    actual_hard_pre_priority = baseline_difficult and evidence_sufficient
    spec_hard_pre_priority = evidence_sufficient and (baseline_difficult or stress_sensitive)
    return {
        "identity_present": identity_present,
        "required_present": required_present,
        "evidence_sufficient": evidence_sufficient,
        "evidence_sufficient_loose": evidence_sufficient_loose,
        "baseline_difficult": baseline_difficult,
        "heading_difficult": heading_difficult,
        "range_difficult": range_difficult,
        "stress_sensitive": stress_sensitive,
        "candidate_low_observable": low_observable,
        "candidate_ambiguous_primary": ambiguous_primary,
        "candidate_conflict": conflict_high,
        "candidate_control_low": control_low,
        "candidate_stable_control": stable_control,
        "actual_rule_hard_pre_priority": actual_hard_pre_priority,
        "spec_hard_pre_priority": spec_hard_pre_priority,
        "full_dev_joined": b(row, "full_dev_joined"),
        "stress_joined": b(row, "stress_joined"),
        "final_state": row.get("derived_state", ""),
    }


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def waterfall(rows: list[dict], flag_rows: list[dict]) -> list[dict]:
    stages = []

    def add(stage: str, condition: str, selected: list[int], notes: str = ""):
        prev = int(stages[-1]["remaining_count"]) if stages else len(rows)
        remaining = len(selected)
        stages.append(
            {
                "stage": stage,
                "condition": condition,
                "remaining_count": remaining,
                "dropped_count": prev - remaining,
                "notes": notes,
            }
        )

    all_idx = list(range(len(rows)))
    add("0", "total pairs", all_idx, "all rows in A-v2 manifest")
    idx = [i for i in all_idx if flag_rows[i]["evidence_sufficient"]]
    add("1", "evidence_sufficient: evidence>=0.67 and match>=0.67", idx)
    idx = [i for i in idx if flag_rows[i]["baseline_difficult"]]
    add("2", "+ baseline_difficult: baseline_error_score>=0.67", idx)
    idx_stress = [i for i in idx if flag_rows[i]["stress_sensitive"]]
    add("3", "+ stress_sensitive: stress_sensitivity_score>=0.67", idx_stress)
    idx_after_ambig = [i for i in idx if not flag_rows[i]["candidate_ambiguous_primary"]]
    add("4_actual", "actual-rule path after removing primary ambiguous/tail", idx_after_ambig)
    idx_after_low = [i for i in idx_after_ambig if not flag_rows[i]["candidate_low_observable"]]
    add("5_actual", "after removing low_observable", idx_after_low)
    idx_after_stable = [i for i in idx_after_low if not flag_rows[i]["candidate_stable_control"]]
    add("6_actual", "after removing stable_control candidates", idx_after_stable)
    idx_join = [i for i in idx_after_stable if flag_rows[i]["full_dev_joined"] or flag_rows[i]["stress_joined"]]
    add("7_actual", "after join constraints: full_dev or stress joined", idx_join)
    final_idx = [i for i in all_idx if rows[i].get("derived_state") == "evidence_sufficient_hard"]
    add("8", "final evidence_sufficient_hard", final_idx, "actual final assignment")
    return stages


def heading_range_split(rows: list[dict], flag_rows: list[dict]) -> list[dict]:
    specs = [
        ("heading_difficult", lambda fl: fl["heading_difficult"]),
        ("range_difficult", lambda fl: fl["range_difficult"]),
        ("both_heading_and_range_difficult", lambda fl: fl["heading_difficult"] and fl["range_difficult"]),
        ("heading_stress_sensitive", lambda fl: fl["stress_sensitive"] and fl["heading_difficult"]),
        ("range_stress_sensitive", lambda fl: fl["stress_sensitive"] and fl["range_difficult"]),
        ("evidence_sufficient + heading_difficult", lambda fl: fl["evidence_sufficient"] and fl["heading_difficult"]),
        ("evidence_sufficient + range_difficult", lambda fl: fl["evidence_sufficient"] and fl["range_difficult"]),
        ("evidence_sufficient + stress_sensitive", lambda fl: fl["evidence_sufficient"] and fl["stress_sensitive"]),
        (
            "evidence_sufficient + baseline_difficult",
            lambda fl: fl["evidence_sufficient"] and fl["baseline_difficult"],
        ),
    ]
    out = []
    for name, pred in specs:
        idx = [i for i, fl in enumerate(flag_rows) if pred(fl)]
        out.append(
            {
                "candidate_type": name,
                "count": len(idx),
                "full_dev_joinable_count": sum(1 for i in idx if flag_rows[i]["full_dev_joined"]),
                "stress_joinable_count": sum(1 for i in idx if flag_rows[i]["stress_joined"]),
                "notes": "",
            }
        )
    return out


def preassignment(rows: list[dict], flag_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    names = [
        "evidence_sufficient",
        "baseline_difficult",
        "stress_sensitive",
        "candidate_low_observable",
        "candidate_ambiguous_primary",
        "candidate_conflict",
        "candidate_stable_control",
        "actual_rule_hard_pre_priority",
        "spec_hard_pre_priority",
    ]
    pair_rows = []
    for row, fl in zip(rows, flag_rows):
        pair_rows.append(
            {
                "canonical_pair_id": row.get("canonical_pair_id", ""),
                "old_base_regime": row.get("old_base_regime", ""),
                "derived_state": row.get("derived_state", ""),
                **{name: int(bool(fl[name])) for name in names},
            }
        )
    summary_rows = []
    for name in names:
        final_counts = Counter(row["derived_state"] for row, fl in zip(rows, flag_rows) if fl[name])
        summary_rows.append(
            {
                "candidate_condition": name,
                "count_before_priority": sum(1 for fl in flag_rows if fl[name]),
                "assigned_final_state_dominant": final_counts.most_common(1)[0][0] if final_counts else "",
                "final_state_counts": json.dumps(dict(final_counts), sort_keys=True),
            }
        )
    overlap_rows = []
    for i, a in enumerate(names):
        for bb in names[i + 1 :]:
            overlap_rows.append(
                {
                    "condition_a": a,
                    "condition_b": bb,
                    "overlap_count": sum(1 for fl in flag_rows if fl[a] and fl[bb]),
                }
            )
    return pair_rows, summary_rows, overlap_rows


def value_summary_for_rows(rows: list[dict], value_getter) -> dict:
    vals = [value_getter(row) for row in rows]
    return summary([v for v in vals if v is not None])


def outcome_reports(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    state_rows = []
    stress_rows = []
    for state in STATE_ORDER:
        group = [row for row in rows if row.get("derived_state") == state]
        baseline_scores = [raw_float(row, "full_dev_baseline_final_score") for row in group]
        heading = [raw_float(row, "full_dev_baseline_angle_rel_error") for row in group]
        dist = [raw_float(row, "full_dev_baseline_distance_rel_error") for row in group]
        stress_values = []
        stress_deltas = []
        for row in group:
            base = raw_float(row, "full_dev_baseline_final_score")
            per_row_stress = [raw_float(row, f"stress{i}_baseline_final_score") for i in [1, 2, 3]]
            per_row_stress = [v for v in per_row_stress if v is not None]
            if per_row_stress:
                stress_values.append(statistics.fmean(per_row_stress))
            if base is not None and per_row_stress:
                stress_deltas.append(statistics.fmean(per_row_stress) - base)
        bs = summary([v for v in baseline_scores if v is not None])
        hs = summary([v for v in heading if v is not None])
        ds = summary([v for v in dist if v is not None])
        sd = summary(stress_deltas)
        state_rows.append(
            {
                "state": state,
                "count": len(group),
                "baseline_join_count": sum(1 for row in group if b(row, "full_dev_joined")),
                "stress_join_count": sum(1 for row in group if b(row, "stress_joined")),
                "mean_error": fmt(bs["mean"]),
                "median_error": fmt(bs["p50"]),
                "p90_error": fmt(bs["p90"]),
                "p95_error": fmt(bs["p95"]),
                "mean_angle_error": fmt(hs["mean"]),
                "p90_angle_error": fmt(hs["p90"]),
                "mean_distance_error": fmt(ds["mean"]),
                "p90_distance_error": fmt(ds["p90"]),
                "mean_stress_delta": fmt(sd["mean"]),
                "p90_stress_delta": fmt(sd["p90"]),
            }
        )
        for variant in [1, 2, 3]:
            vals = [raw_float(row, f"stress{variant}_baseline_final_score") for row in group]
            deltas = []
            for row in group:
                base = raw_float(row, "full_dev_baseline_final_score")
                stress = raw_float(row, f"stress{variant}_baseline_final_score")
                if base is not None and stress is not None:
                    deltas.append(stress - base)
            vs = summary([v for v in vals if v is not None])
            d = summary(deltas)
            stress_rows.append(
                {
                    "state": state,
                    "stress_variant": f"stress{variant}",
                    "join_count": vs["count"],
                    "mean_error": fmt(vs["mean"]),
                    "median_error": fmt(vs["p50"]),
                    "p90_error": fmt(vs["p90"]),
                    "mean_delta_vs_baseline": fmt(d["mean"]),
                    "median_delta": fmt(d["p50"]),
                    "p90_delta": fmt(d["p90"]),
                }
            )
    return state_rows, stress_rows


def group_distribution(rows: list[dict], group_predicates: dict[str, callable]) -> list[dict]:
    out = []
    states = ["stable_control_anchor", "ambiguous_unreliable", "unknown_unvalidated", "low_observable", "evidence_sufficient_hard"]
    for group, pred in group_predicates.items():
        subset = [row for row in rows if pred(row)]
        counts = Counter(row.get("derived_state", "") for row in subset)
        out.append({"group": group, "total_count": len(subset), **{state: counts.get(state, 0) for state in states}})
    return out


def join_score_distribution(rows: list[dict], flag: str) -> list[dict]:
    out = []
    groups = {
        f"{flag}_joined": lambda row: b(row, flag),
        f"{flag}_unjoined": lambda row: not b(row, flag),
    }
    for group, pred in groups.items():
        subset = [row for row in rows if pred(row)]
        for score in [
            "evidence_sufficiency_score",
            "ambiguity_tail_risk_score",
            "semantic_geometric_conflict_score",
            "control_stability_score",
        ]:
            s = summary([f(row, score) for row in subset])
            out.append({"group": group, "score": score, "p50": fmt(s["p50"]), "p90": fmt(s["p90"]), "count": s["count"]})
    return out


def join_target_distribution(rows: list[dict]) -> list[dict]:
    totals = Counter(row.get("target_key", "unknown") for row in rows)
    full = Counter(row.get("target_key", "unknown") for row in rows if b(row, "full_dev_joined"))
    stress = Counter(row.get("target_key", "unknown") for row in rows if b(row, "stress_joined"))
    out = []
    for target, total in totals.most_common():
        joined = full.get(target, 0) + stress.get(target, 0)
        out.append(
            {
                "target_key": target,
                "total_count": total,
                "full_dev_joined_count": full.get(target, 0),
                "stress_joined_count": stress.get(target, 0),
                "any_joined_count": joined,
                "any_join_rate": joined / total if total else 0.0,
            }
        )
    return out


def unknown_breakdown(rows: list[dict]) -> list[dict]:
    unknowns = [row for row in rows if row.get("derived_state") == "unknown_unvalidated"]
    reasons = Counter()
    for row in unknowns:
        if not b(row, "full_dev_joined"):
            reasons["missing_baseline_join"] += 1
        if not b(row, "stress_joined"):
            reasons["missing_stress_join"] += 1
        if any(not has_value(row, key) for key in flags.required_axis_fields):
            reasons["missing_features"] += 1
        if not reasons:
            reasons["no_rule_matched"] += 1
        if (
            f(row, "evidence_sufficiency_score") >= 0.67
            and f(row, "match_sufficiency_score") >= 0.67
            and f(row, "control_stability_score") < 0.67
            and f(row, "baseline_error_score") < 0.67
            and f(row, "stress_sensitivity_score") < 0.67
        ):
            reasons["middle_band_no_rule_matched"] += 1
    return [
        {"unknown_reason": key, "count": value, "ratio": value / len(unknowns) if unknowns else 0.0}
        for key, value in reasons.most_common()
    ]


def bucket_breakdown(rows: list[dict], flag_rows: list[dict]) -> list[dict]:
    out = []
    for bucket in ["ambiguous_unreliable", "stable_control_anchor", "unknown_unvalidated"]:
        subset = [(row, fl) for row, fl in zip(rows, flag_rows) if row.get("derived_state") == bucket]
        total = len(subset)
        specs = [
            ("evidence_sufficient", lambda fl: fl["evidence_sufficient"]),
            ("baseline_joinable", lambda fl: fl["full_dev_joined"]),
            ("stress_joinable", lambda fl: fl["stress_joined"]),
            ("baseline_difficult", lambda fl: fl["baseline_difficult"]),
            ("stress_sensitive", lambda fl: fl["stress_sensitive"]),
            ("heading_difficult", lambda fl: fl["heading_difficult"]),
            ("range_difficult", lambda fl: fl["range_difficult"]),
            ("evidence_sufficient + baseline_difficult", lambda fl: fl["evidence_sufficient"] and fl["baseline_difficult"]),
            ("evidence_sufficient + stress_sensitive", lambda fl: fl["evidence_sufficient"] and fl["stress_sensitive"]),
            (
                "evidence_sufficient + baseline_difficult + stress_sensitive",
                lambda fl: fl["evidence_sufficient"] and fl["baseline_difficult"] and fl["stress_sensitive"],
            ),
        ]
        for reason, pred in specs:
            count = sum(1 for _, fl in subset if pred(fl))
            out.append({"bucket": bucket, "condition": reason, "count": count, "ratio": count / total if total else 0.0})
    return out


def transition_matrix(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    old_states = sorted({row.get("old_base_regime", "") for row in rows})
    new_states = ["stable_control_anchor", "ambiguous_unreliable", "unknown_unvalidated", "low_observable", "evidence_sufficient_hard", "stress_sensitive_control", "conflict_candidate"]
    out = []
    for old in old_states:
        subset = [row for row in rows if row.get("old_base_regime", "") == old]
        counts = Counter(row.get("derived_state", "") for row in subset)
        out.append({"old_state": old, **{state: counts.get(state, 0) for state in new_states}, "total": len(subset)})
    old_hard = [row for row in rows if row.get("old_base_regime") == "hard_trainable"]
    breakdown = []
    for state in new_states:
        subset = [row for row in old_hard if row.get("derived_state") == state]
        bs = summary([raw_float(row, "full_dev_baseline_final_score") for row in subset if raw_float(row, "full_dev_baseline_final_score") is not None])
        stress_delta = []
        for row in subset:
            base = raw_float(row, "full_dev_baseline_final_score")
            stresses = [raw_float(row, f"stress{i}_baseline_final_score") for i in [1, 2, 3]]
            stresses = [v for v in stresses if v is not None]
            if base is not None and stresses:
                stress_delta.append(statistics.fmean(stresses) - base)
        sd = summary(stress_delta)
        breakdown.append(
            {
                "old_hard_trainable_to_new_state": state,
                "count": len(subset),
                "baseline_error_p50": fmt(bs["p50"]),
                "baseline_error_p90": fmt(bs["p90"]),
                "stress_delta_p90": fmt(sd["p90"]),
            }
        )
    return out, breakdown


def near_miss(rows: list[dict], flag_rows: list[dict], t: dict) -> list[dict]:
    specs = [
        ("evidence_sufficiency_score", t["evidence_sufficiency_high"]),
        ("match_sufficiency_score", t["evidence_sufficiency_high"]),
        ("baseline_error_score", t["baseline_difficulty_high"]),
        ("stress_sensitivity_score", t["stress_sensitivity_high"]),
        ("ambiguity_tail_risk_score", t["tail_risk_high"]),
        ("control_stability_score", t["control_stability_high"]),
    ]
    rows_out = []
    for field, threshold in specs:
        for width in [0.02, 0.05, 0.10]:
            subset = [row for row in rows if abs(f(row, field) - threshold) <= width]
            counts = Counter(row.get("derived_state", "") for row in subset)
            rows_out.append(
                {
                    "field": field,
                    "threshold": threshold,
                    "window": width,
                    "near_miss_count": len(subset),
                    "final_state_counts": json.dumps(dict(counts), sort_keys=True),
                }
            )
    return rows_out


def score_quantiles(rows: list[dict]) -> list[dict]:
    threshold_map = {
        "evidence_sufficiency_score": 0.67,
        "heading_observability_score": 0.33,
        "range_observability_score": 0.33,
        "match_sufficiency_score": 0.67,
        "baseline_error_score": 0.67,
        "stress_sensitivity_score": 0.67,
        "ambiguity_tail_risk_score": 0.67,
        "semantic_geometric_conflict_score": 0.67,
        "control_stability_score": 0.67,
    }
    out = []
    for field in SCORE_FIELDS:
        s = summary([raw_float(row, field) for row in rows])
        out.append({"score": field, **{k: fmt(v) for k, v in s.items()}, "threshold_used": threshold_map.get(field, "")})
    return out


def leakage_audit(schema) -> tuple[list[dict], dict]:
    rows = []
    for field, category in schema.FIELD_SOURCE_CATEGORY.items():
        uses_baseline = "baseline" in field or "stress" in field or "error" in field or "tail" in field
        uses_gt = False
        uses_residual = "error" in field
        non_leaking = category in {"INPUT_SIDE_NON_LEAKING", "MATCHER_SIDE_NON_LEAKING"}
        rows.append(
            {
                "field": field,
                "source_category": category,
                "uses_gt": str(uses_gt).lower(),
                "uses_baseline_prediction_or_error": str(uses_baseline).lower(),
                "uses_prediction_residual": str(uses_residual).lower(),
                "non_leaking_at_inference": str(non_leaking).lower(),
                "used_for_state_assignment": str(field in getattr(flags, "required_axis_fields") or field in {"canonical_pair_id"}).lower(),
                "used_only_for_validation": str(category in {"BASELINE_OUTCOME_VALIDATION_ONLY", "ANALYSIS_ONLY"}).lower(),
            }
        )
    return rows, {"field_count": len(rows), "non_leaking_count": sum(1 for row in rows if row["non_leaking_at_inference"] == "true")}


def rulebook(schema, rules, out_dir: Path) -> None:
    t = rules.default_thresholds()
    states = [
        {
            "state": "unknown_unvalidated",
            "priority_order": 1,
            "conditions": ["missing canonical_pair_id OR any required axis missing"],
            "thresholds": [],
            "required_join": "false",
            "leakage_status": "mixed; missingness check only",
        },
        {
            "state": "low_observable",
            "priority_order": 2,
            "conditions": ["min(heading_observability_score, range_observability_score) <= observability_low"],
            "thresholds": [f"observability_low={t['observability_low']}"],
            "required_join": "false",
            "leakage_status": "deployable_non_leaking",
        },
        {
            "state": "ambiguous_unreliable",
            "priority_order": 3,
            "conditions": ["evidence_sufficiency_score <= evidence_sufficiency_low OR ambiguity_tail_risk_score >= tail_risk_high"],
            "thresholds": [f"evidence_sufficiency_low={t['evidence_sufficiency_low']}", f"tail_risk_high={t['tail_risk_high']}"],
            "required_join": "false for evidence low; validation-only for tail risk",
            "leakage_status": "mixed; ambiguity_tail_risk_score is validation-only",
        },
        {
            "state": "stress_sensitive_control",
            "priority_order": 4,
            "conditions": ["stress_sensitivity_score >= stress_sensitivity_high"],
            "thresholds": [f"stress_sensitivity_high={t['stress_sensitivity_high']}"],
            "required_join": "stress outcome required to be meaningful",
            "leakage_status": "validation_only",
        },
        {
            "state": "conflict_candidate",
            "priority_order": 5,
            "conditions": ["semantic_geometric_conflict_score >= conflict_high"],
            "thresholds": [f"conflict_high={t['conflict_high']}"],
            "required_join": "false",
            "leakage_status": "deployable_non_leaking",
        },
        {
            "state": "evidence_sufficient_hard",
            "priority_order": 6,
            "conditions": [
                "baseline_error_score >= baseline_difficulty_high",
                "evidence_sufficiency_score >= evidence_sufficiency_high",
                "match_sufficiency_score >= evidence_sufficiency_high",
            ],
            "thresholds": [f"baseline_difficulty_high={t['baseline_difficulty_high']}", f"evidence_sufficiency_high={t['evidence_sufficiency_high']}"],
            "required_join": "baseline outcome required by current implementation",
            "leakage_status": "validation_only; not deployable",
        },
        {
            "state": "ambiguous_unreliable",
            "priority_order": 7,
            "conditions": ["baseline_error_score >= baseline_difficulty_high OR control_stability_score <= control_stability_low"],
            "thresholds": [f"baseline_difficulty_high={t['baseline_difficulty_high']}", f"control_stability_low={t['control_stability_low']}"],
            "required_join": "baseline outcome for baseline branch",
            "leakage_status": "mixed/validation_only",
        },
        {
            "state": "stable_control_anchor",
            "priority_order": 8,
            "conditions": [
                "evidence_sufficiency_score >= evidence_sufficiency_high",
                "match_sufficiency_score >= evidence_sufficiency_high",
                "control_stability_score >= control_stability_high",
            ],
            "thresholds": [f"evidence_sufficiency_high={t['evidence_sufficiency_high']}", f"control_stability_high={t['control_stability_high']}"],
            "required_join": "false in implementation",
            "leakage_status": "mixed; control_stability_score is derived and not proven deployable",
        },
    ]
    write_csv(out_dir / "a_v2_state_assignment_rulebook.csv", states)
    write_json(
        out_dir / "a_v2_state_assignment_rulebook.json",
        {
            "thresholds": t,
            "required_axis_fields": list(rules.REQUIRED_AXIS_FIELDS),
            "field_source_category": schema.FIELD_SOURCE_CATEGORY,
            "state_rules": states,
        },
    )
    lines = ["# A-v2 State Assignment Rulebook", "", "## Thresholds", ""]
    for k, v in t.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(["", "## Priority Rules", ""])
    for state in states:
        lines.append(f"### {state['priority_order']}. `{state['state']}`")
        lines.append("")
        lines.append(f"- Conditions: {'; '.join(state['conditions'])}")
        lines.append(f"- Thresholds: {', '.join(state['thresholds']) if state['thresholds'] else 'none'}")
        lines.append(f"- Required join: {state['required_join']}")
        lines.append(f"- Leakage status: {state['leakage_status']}")
        lines.append("")
    write_text(out_dir / "a_v2_state_assignment_rulebook.md", "\n".join(lines))


def failure_summary(rows, flag_rows, out_dir: Path) -> None:
    total = len(rows)
    hard_pre_actual = sum(1 for fl in flag_rows if fl["actual_rule_hard_pre_priority"])
    spec_hard_pre = sum(1 for fl in flag_rows if fl["spec_hard_pre_priority"])
    final_hard = sum(1 for row in rows if row.get("derived_state") == "evidence_sufficient_hard")
    evidence_sufficient = sum(1 for fl in flag_rows if fl["evidence_sufficient"])
    baseline_difficult = sum(1 for fl in flag_rows if fl["baseline_difficult"])
    stress_sensitive = sum(1 for fl in flag_rows if fl["stress_sensitive"])
    ambiguous = sum(1 for fl in flag_rows if fl["candidate_ambiguous_primary"])
    stable = sum(1 for fl in flag_rows if fl["candidate_stable_control"])
    reasons = [
        {
            "rank": 1,
            "cause": "all actual hard pre-candidates are swallowed by higher-priority ambiguous/tail assignment",
            "evidence": f"actual_rule_hard_pre_priority={hard_pre_actual}; final evidence_sufficient_hard={final_hard}",
        },
        {
            "rank": 2,
            "cause": "baseline/stress difficulty signal is sparse or incompatible with evidence-sufficient gate",
            "evidence": f"evidence_sufficient={evidence_sufficient}; baseline_difficult={baseline_difficult}; stress_sensitive={stress_sensitive}; spec_hard_pre_priority={spec_hard_pre}",
        },
        {
            "rank": 3,
            "cause": "ambiguous/tail branch captures risk rows before hard promotion",
            "evidence": f"candidate_ambiguous_primary={ambiguous}; ambiguous final={sum(1 for row in rows if row.get('derived_state') == 'ambiguous_unreliable')}",
        },
        {
            "rank": 4,
            "cause": "stable-control branch is broad",
            "evidence": f"candidate_stable_control={stable}; final stable_control_anchor={sum(1 for row in rows if row.get('derived_state') == 'stable_control_anchor')}",
        },
    ]
    write_json(
        out_dir / "a_v2_failure_attribution_summary.json",
        {
            "total_rows": total,
            "final_evidence_sufficient_hard": final_hard,
            "main_causes_ranked": reasons,
            "go_no_go_verdict": "NO-GO for training policy; REDESIGN REQUIRED before any A training use",
        },
    )
    lines = [
        "# A-v2 Failure Attribution Summary",
        "",
        "## Verdict",
        "",
        "`NO-GO`: A-v2 must not enter training policy. Redesign-v3 or route hold is required.",
        "",
        "## Main Causes Ranked",
        "",
    ]
    for r in reasons:
        lines.append(f"{r['rank']}. **{r['cause']}**: {r['evidence']}")
    lines.extend(
        [
            "",
            "## Confirmed",
            "",
            f"- Final `evidence_sufficient_hard`: `{final_hard}`.",
            "- Old `hard_trainable` must remain deprecated.",
            "- Current A-v2 states are not valid training-policy inputs.",
            "",
            "## Still Unknown",
            "",
            "- Whether a less conservative non-leaking hard candidate can be recovered without validation-only outcome fields.",
            "- Whether heading-hard and range-hard should be separated instead of using one combined hard state.",
            "- Whether joinable subset bias hides useful hard rows outside the current validation surface.",
            "",
            "## Recommendation",
            "",
            "Write a redesign-v3 spec only if A remains a priority. The next spec should separate candidate discovery from outcome validation and split heading/range hard states.",
        ]
    )
    write_text(out_dir / "a_v2_failure_attribution_summary.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.manifest)

    rules = importlib.import_module("phase27_a_taxonomy_redesign_v2_rules")
    schema = importlib.import_module("phase27_a_taxonomy_redesign_v2_schema")
    flags.required_axis_fields = tuple(rules.REQUIRED_AXIS_FIELDS)
    t = rules.default_thresholds()
    flag_rows = [flags(row, t) for row in rows]

    rulebook(schema, rules, out_dir)
    write_csv(out_dir / "hard_candidate_waterfall.csv", waterfall(rows, flag_rows))
    write_csv(out_dir / "heading_range_candidate_split.csv", heading_range_split(rows, flag_rows))
    pair_flags, pre_summary, pairwise = preassignment(rows, flag_rows)
    write_csv(out_dir / "preassignment_multilabel_flags.csv", pair_flags)
    write_csv(out_dir / "preassignment_multilabel_summary.csv", pre_summary)
    write_csv(out_dir / "preassignment_pairwise_overlap.csv", pairwise)
    statewise, stress_variant = outcome_reports(rows)
    write_csv(out_dir / "statewise_baseline_stress_outcome.csv", statewise)
    write_csv(out_dir / "statewise_stress_variant_outcome.csv", stress_variant)
    write_csv(
        out_dir / "joined_vs_unjoined_state_distribution.csv",
        group_distribution(
            rows,
            {
                "full_dev_joined": lambda row: b(row, "full_dev_joined"),
                "full_dev_unjoined": lambda row: not b(row, "full_dev_joined"),
                "stress_joined": lambda row: b(row, "stress_joined"),
                "stress_unjoined": lambda row: not b(row, "stress_joined"),
            },
        ),
    )
    write_csv(out_dir / "joined_vs_unjoined_score_distribution.csv", join_score_distribution(rows, "full_dev_joined") + join_score_distribution(rows, "stress_joined"))
    write_csv(out_dir / "joined_vs_unjoined_target_distribution.csv", join_target_distribution(rows))
    write_csv(out_dir / "unknown_unvalidated_breakdown.csv", unknown_breakdown(rows))
    write_csv(out_dir / "bucket_condition_breakdown.csv", bucket_breakdown(rows, flag_rows))
    trans, old_hard = transition_matrix(rows)
    write_csv(out_dir / "old_to_new_transition_matrix.csv", trans)
    write_csv(out_dir / "old_hard_trainable_breakdown.csv", old_hard)
    write_csv(out_dir / "threshold_near_miss_analysis.csv", near_miss(rows, flag_rows, t))
    write_csv(out_dir / "score_quantile_summary.csv", score_quantiles(rows))
    deploy_rows, deploy_summary = leakage_audit(schema)
    write_csv(out_dir / "leakage_deployability_audit.csv", deploy_rows)
    write_json(out_dir / "leakage_deployability_audit.json", deploy_summary)
    failure_summary(rows, flag_rows, out_dir)
    write_json(
        out_dir / "diagnostics_index.json",
        {
            "manifest": args.manifest,
            "out_dir": str(out_dir),
            "artifact_count": len(list(out_dir.iterdir())),
            "verdict": "failure-attribution-complete-no-go-training-policy",
        },
    )
    print("phase27_a_taxonomy_redesign_v2_failure_diagnostics COMPLETE")
    print(out_dir)


if __name__ == "__main__":
    main()
