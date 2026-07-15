"""Diagnose and reacquire A-state diversity for A-v3.2c fixed manifests."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_FIXED = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_v3_2b_fixed_manifest_shared_outcome_surface_reacquisition/manifests/fixed_shared_pair_manifest_bounded.csv"
DEFAULT_BOUNDED = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_v3_2c_bounded_outcome_consistency/manifests/fixed_manifest_tiny.csv"
DEFAULT_EVIDENCE = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_calibrated_v2.csv"
DEFAULT_TRAINING = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_taxonomy_redesign_v3/manifests/training_readiness_verdict_manifest.csv"
DEFAULT_OUT = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_v3_2c_state_diversity_reacquisition_diagnostic"


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def index_by_id(rows):
    return {row.get("canonical_pair_id", ""): row for row in rows if row.get("canonical_pair_id", "")}


def parse_flags(value):
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {"FLAG_PARSE_ERROR": "True"}
    return {k: str(v) for k, v in raw.items()}


def is_true(flags, key):
    return str(flags.get(key, "")).lower() == "true"


def flag_state(flags):
    if is_true(flags, "READY_CONTROL_PRESERVATION"):
        return "flag_ready_control_preservation"
    if is_true(flags, "READY_HEADING_HARD_TRAINING") and is_true(flags, "READY_RANGE_HARD_TRAINING"):
        return "flag_ready_joint_hard_training"
    if is_true(flags, "READY_HEADING_HARD_TRAINING"):
        return "flag_ready_heading_hard_training"
    if is_true(flags, "READY_RANGE_HARD_TRAINING"):
        return "flag_ready_range_hard_training"
    if is_true(flags, "READY_CORRESPONDENCE_DIAGNOSTIC") or is_true(flags, "semantic_geometric_conflict_candidate") or is_true(flags, "local_alignment_needed_candidate"):
        return "flag_correspondence_diagnostic"
    if is_true(flags, "QUARANTINE_LOW_OBSERVABLE") or is_true(flags, "low_observable_candidate"):
        return "flag_low_observable"
    if is_true(flags, "NOT_READY"):
        return "flag_not_ready"
    if is_true(flags, "ambiguity_candidate"):
        return "flag_ambiguity_candidate"
    if is_true(flags, "control_candidate"):
        return "flag_control_candidate"
    if is_true(flags, "evidence_sufficient_candidate"):
        return "flag_evidence_sufficient_candidate"
    return "flag_unknown"


def reacquire_state(row, evidence_row, training_row):
    if evidence_row and evidence_row.get("base_regime"):
        return "evidence_base_regime:" + evidence_row["base_regime"]
    if training_row:
        if training_row.get("READY_CONTROL_PRESERVATION") == "True":
            return "training_ready_control_preservation"
        if training_row.get("READY_HEADING_HARD_TRAINING") == "True" and training_row.get("READY_RANGE_HARD_TRAINING") == "True":
            return "training_ready_joint_hard"
        if training_row.get("READY_HEADING_HARD_TRAINING") == "True":
            return "training_ready_heading_hard"
        if training_row.get("READY_RANGE_HARD_TRAINING") == "True":
            return "training_ready_range_hard"
    return flag_state(parse_flags(row.get("candidate_flags_json", "")))


def summarize_source(name, rows, evidence_by_id, training_by_id):
    candidate_state = Counter((r.get("candidate_state") or "<empty>") for r in rows)
    flag_states = Counter(flag_state(parse_flags(r.get("candidate_flags_json", ""))) for r in rows)
    evidence_join = []
    training_join = []
    reacquired = []
    for row in rows:
        cid = row.get("canonical_pair_id", "")
        evidence = evidence_by_id.get(cid)
        training = training_by_id.get(cid)
        if evidence:
            evidence_join.append(evidence)
        if training:
            training_join.append(training)
        reacquired.append(reacquire_state(row, evidence, training))
    return {
        "name": name,
        "row_count": len(rows),
        "unique_canonical_pair_id": len({r.get("canonical_pair_id", "") for r in rows if r.get("canonical_pair_id", "")}),
        "candidate_state_distribution": dict(candidate_state.most_common()),
        "flag_state_distribution": dict(flag_states.most_common()),
        "evidence_join_count": len(evidence_join),
        "evidence_base_regime_distribution": dict(Counter(r.get("base_regime", "<empty>") for r in evidence_join).most_common()),
        "training_join_count": len(training_join),
        "training_ready_distribution": dict(Counter(reacquire_state({}, None, r) for r in training_join).most_common()),
        "reacquired_state_distribution": dict(Counter(reacquired).most_common()),
    }


def build_reacquired_manifest(rows, evidence_by_id, training_by_id):
    out = []
    for row in rows:
        cid = row.get("canonical_pair_id", "")
        evidence = evidence_by_id.get(cid)
        training = training_by_id.get(cid)
        flags = parse_flags(row.get("candidate_flags_json", ""))
        new = dict(row)
        new["reacquired_state"] = reacquire_state(row, evidence, training)
        new["flag_state"] = flag_state(flags)
        new["evidence_base_regime"] = evidence.get("base_regime", "") if evidence else ""
        new["evidence_joined"] = "1" if evidence else "0"
        new["training_joined"] = "1" if training else "0"
        new["ready_control_preservation"] = flags.get("READY_CONTROL_PRESERVATION", "")
        new["ready_heading_hard_training"] = flags.get("READY_HEADING_HARD_TRAINING", "")
        new["ready_range_hard_training"] = flags.get("READY_RANGE_HARD_TRAINING", "")
        new["not_ready"] = flags.get("NOT_READY", "")
        new["analysis_only"] = flags.get("ANALYSIS_ONLY", "")
        out.append(new)
    return out


def write_report(out_dir, metrics):
    lines = [
        "# A-v3.2c State-Diversity Reacquisition Diagnostic",
        "",
        f"verdict: `{metrics['verdict']}`",
        f"reason: `{metrics['reason']}`",
        "",
    ]
    for key in ("full_fixed", "current_bounded"):
        item = metrics[key]
        lines.extend([
            f"## {item['name']}",
            "",
            f"- row_count: {item['row_count']}",
            f"- unique_canonical_pair_id: {item['unique_canonical_pair_id']}",
            f"- candidate_state_distribution: `{json.dumps(item['candidate_state_distribution'], sort_keys=True)}`",
            f"- flag_state_distribution: `{json.dumps(item['flag_state_distribution'], sort_keys=True)}`",
            f"- evidence_join_count: {item['evidence_join_count']}",
            f"- evidence_base_regime_distribution: `{json.dumps(item['evidence_base_regime_distribution'], sort_keys=True)}`",
            f"- training_join_count: {item['training_join_count']}",
            f"- reacquired_state_distribution: `{json.dumps(item['reacquired_state_distribution'], sort_keys=True)}`",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- The fixed manifest's `candidate_state` is collapsed to `candidate_only_unvalidated`.",
        "- The same rows retain diversity in `candidate_flags_json`.",
        "- The evidence-state calibrated manifest can recover deployable-like `base_regime` labels for joined canonical ids.",
        "- This diagnostic does not train, finetune, tune thresholds, run full eval, build B/C gates, or submit results.",
    ])
    (Path(out_dir) / "reports" / "state_diversity_reacquisition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", default=DEFAULT_FIXED)
    parser.add_argument("--bounded-manifest", default=DEFAULT_BOUNDED)
    parser.add_argument("--evidence-manifest", default=DEFAULT_EVIDENCE)
    parser.add_argument("--training-readiness-manifest", default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    fixed = read_csv_rows(args.fixed_manifest)
    bounded = read_csv_rows(args.bounded_manifest)
    evidence = read_csv_rows(args.evidence_manifest)
    training = read_csv_rows(args.training_readiness_manifest)
    evidence_by_id = index_by_id(evidence)
    training_by_id = index_by_id(training)

    full_summary = summarize_source("full_fixed_manifest", fixed, evidence_by_id, training_by_id)
    bounded_summary = summarize_source("current_bounded_manifest", bounded, evidence_by_id, training_by_id)
    distinct_reacquired = len(bounded_summary["reacquired_state_distribution"])
    if distinct_reacquired >= 2 and bounded_summary["evidence_join_count"] == bounded_summary["row_count"]:
        verdict = "state-diversity-reacquisition-ready"
        reason = "bounded_manifest_can_recover_multiple_evidence_states_by_exact_canonical_join"
    elif distinct_reacquired >= 2:
        verdict = "state-diversity-reacquisition-partial"
        reason = "bounded_manifest_has_multiple_reacquired_states_but_join_is_incomplete"
    else:
        verdict = "state-diversity-reacquisition-blocked"
        reason = "bounded_manifest_still_has_one_reacquired_state"

    metrics = {
        "verdict": verdict,
        "reason": reason,
        "fixed_manifest": args.fixed_manifest,
        "bounded_manifest": args.bounded_manifest,
        "evidence_manifest": args.evidence_manifest,
        "training_readiness_manifest": args.training_readiness_manifest,
        "full_fixed": full_summary,
        "current_bounded": bounded_summary,
    }
    write_json(out / "metrics" / "state_diversity_reacquisition_metrics.json", metrics)
    write_report(out, metrics)

    reacquired_bounded = build_reacquired_manifest(bounded, evidence_by_id, training_by_id)
    base_columns = list(bounded[0].keys()) if bounded else []
    extra_columns = [
        "reacquired_state",
        "flag_state",
        "evidence_base_regime",
        "evidence_joined",
        "training_joined",
        "ready_control_preservation",
        "ready_heading_hard_training",
        "ready_range_hard_training",
        "not_ready",
        "analysis_only",
    ]
    write_csv(out / "tables" / "bounded_manifest_with_reacquired_states.csv", reacquired_bounded, base_columns + extra_columns)
    print((out / "reports" / "state_diversity_reacquisition_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
