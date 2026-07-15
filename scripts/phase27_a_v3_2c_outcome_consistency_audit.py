"""A-v3.2c bounded fixed-manifest outcome-consistency audit."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


STATE_COLUMNS = [
    "reacquired_state",
    "evidence_state",
    "a_state",
    "taxonomy_state",
    "final_state",
    "candidate_state",
    "state",
    "base_regime",
]

AUTHORIZED_VERDICTS = {
    "a-v3-2c-bounded-validation-pass",
    "a-v3-2c-bounded-validation-weak-inconclusive",
    "a-v3-2c-bounded-validation-fail",
    "a-v3-2c-bounded-validation-blocked-runner",
    "a-v3-2c-bounded-validation-blocked-coverage",
    "a-v3-2c-bounded-validation-blocked-identity",
}


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path, text):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def fmt(value):
    if value is None:
        return ""
    return f"{value:.12g}"


def choose_state(row):
    for column in STATE_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return "unknown_unlabeled"


def index_unique(rows):
    counts = Counter(row.get("canonical_pair_id", "") for row in rows)
    duplicates = {key for key, count in counts.items() if key and count > 1}
    missing = counts.get("", 0)
    return {row.get("canonical_pair_id", ""): row for row in rows if row.get("canonical_pair_id", "") and row.get("canonical_pair_id", "") not in duplicates}, duplicates, missing


def row_prediction_ok(row):
    return row.get("row_status") == "ok" and bool(row.get("prediction_heading"))


def build_shared_surface(manifest_rows, baseline_rows, stress_by_variant):
    manifest_by_id, duplicate_manifest, missing_manifest_ids = index_unique(manifest_rows)
    baseline_by_id, duplicate_baseline, missing_baseline_ids = index_unique(baseline_rows)
    stress_index = {}
    duplicate_stress = {}
    missing_stress_ids = {}
    for variant, rows in stress_by_variant.items():
        by_id, dupes, missing = index_unique(rows)
        stress_index[variant] = by_id
        duplicate_stress[variant] = dupes
        missing_stress_ids[variant] = missing

    shared_rows = []
    missing_baseline = 0
    missing_stress_rows = 0
    shared_outcome_count = 0
    stress_variants = list(stress_by_variant)
    blocked_ids = set(duplicate_manifest) | set(duplicate_baseline)
    for dupes in duplicate_stress.values():
        blocked_ids.update(dupes)

    for manifest_row in manifest_rows:
        cid = manifest_row.get("canonical_pair_id", "")
        baseline = baseline_by_id.get(cid)
        if not baseline:
            missing_baseline += 1
        out = {
            "canonical_pair_id": cid,
            "target_key": manifest_row.get("target_key") or manifest_row.get("group_id") or "",
            "state": choose_state(manifest_row),
            "baseline_status": baseline.get("row_status", "missing") if baseline else "missing",
            "baseline_joint_error": baseline.get("joint_error", "") if baseline else "",
            "baseline_heading_error": baseline.get("heading_abs_error", "") if baseline else "",
            "baseline_range_error": baseline.get("range_abs_error", "") if baseline else "",
        }
        baseline_error = safe_float(out["baseline_joint_error"])
        shared = bool(cid) and cid not in blocked_ids and row_prediction_ok(baseline or {})
        for variant in stress_variants:
            row = stress_index[variant].get(cid)
            if not row:
                missing_stress_rows += 1
            status = row.get("row_status", "missing") if row else "missing"
            joint = row.get("joint_error", "") if row else ""
            heading = row.get("heading_abs_error", "") if row else ""
            range_error = row.get("range_abs_error", "") if row else ""
            stress_error = safe_float(joint)
            delta = None if baseline_error is None or stress_error is None else stress_error - baseline_error
            out[f"{variant}_status"] = status
            out[f"{variant}_joint_error"] = joint
            out[f"{variant}_heading_error"] = heading
            out[f"{variant}_range_error"] = range_error
            out[f"{variant}_delta"] = fmt(delta)
            shared = shared and row_prediction_ok(row or {})
        out["shared_outcome"] = "1" if shared else "0"
        if shared:
            shared_outcome_count += 1
        shared_rows.append(out)

    issues = {
        "manifest_rows": len(manifest_rows),
        "missing_manifest_ids": missing_manifest_ids,
        "duplicate_manifest_ids": len(duplicate_manifest),
        "duplicate_baseline_ids": len(duplicate_baseline),
        "duplicate_stress_ids": sum(len(v) for v in duplicate_stress.values()),
        "missing_baseline_ids": missing_baseline_ids,
        "missing_stress_ids": sum(missing_stress_ids.values()),
        "missing_baseline": missing_baseline,
        "missing_stress_rows": missing_stress_rows,
        "shared_outcome_count": shared_outcome_count,
    }
    return shared_rows, issues


def quantile(values, q):
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def summarize_numeric(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    return {
        "mean": sum(clean) / len(clean),
        "median": quantile(clean, 0.5),
        "p90": quantile(clean, 0.9),
        "p95": quantile(clean, 0.95),
    }


def compute_variant_summary(rows):
    status_counts = Counter(row.get("row_status", "") for row in rows)
    prediction_success = sum(1 for row in rows if row_prediction_ok(row))
    return {
        "variant_id": rows[0].get("variant_id", "") if rows else "",
        "row_count": len(rows),
        "prediction_success_count": prediction_success,
        "prediction_success_fraction": prediction_success / len(rows) if rows else 0.0,
        "status_counts": dict(status_counts),
    }


def compute_state_report(shared_rows, stress_variants):
    global_errors = [safe_float(row.get("baseline_joint_error")) for row in shared_rows if row.get("shared_outcome") == "1"]
    global_median = quantile(global_errors, 0.5)
    by_state = defaultdict(list)
    for row in shared_rows:
        by_state[row.get("state", "unknown_unlabeled")].append(row)

    report = []
    for state, rows in sorted(by_state.items()):
        shared = [row for row in rows if row.get("shared_outcome") == "1"]
        base_errors = [safe_float(row.get("baseline_joint_error")) for row in shared]
        heading_errors = [safe_float(row.get("baseline_heading_error")) for row in shared]
        range_errors = [safe_float(row.get("baseline_range_error")) for row in shared]
        deltas = []
        same_sign_count = 0
        for row in shared:
            row_deltas = [safe_float(row.get(f"{variant}_delta")) for variant in stress_variants]
            row_deltas = [value for value in row_deltas if value is not None]
            deltas.extend(row_deltas)
            if row_deltas and (all(v >= 0 for v in row_deltas) or all(v <= 0 for v in row_deltas)):
                same_sign_count += 1
        base = summarize_numeric(base_errors)
        stress = summarize_numeric(deltas)
        heading = summarize_numeric(heading_errors)
        range_summary = summarize_numeric(range_errors)
        consistency = same_sign_count / len(shared) if shared else None
        mean_error = base["mean"]
        stress_delta_mean = stress["mean"]
        if not shared:
            note = "no_shared_rows"
        elif len(shared) >= 5 and mean_error is not None and stress_delta_mean is not None and mean_error > 0 and stress_delta_mean > 0:
            note = "high_error_sensitive"
        elif (
            len(shared) >= 5
            and mean_error is not None
            and global_median is not None
            and stress_delta_mean is not None
            and mean_error <= global_median
            and abs(stress_delta_mean) <= max(1.0, abs(global_median) * 0.1)
        ):
            note = "stable_control_like"
        else:
            note = "weak_or_mixed"
        report.append(
            {
                "state": state,
                "count": str(len(rows)),
                "prediction_success_count": str(len(shared)),
                "shared_outcome_count": str(len(shared)),
                "mean_error": fmt(base["mean"]),
                "median_error": fmt(base["median"]),
                "p90_error": fmt(base["p90"]),
                "p95_error": fmt(base["p95"]),
                "heading_error": fmt(heading["mean"]),
                "range_error": fmt(range_summary["mean"]),
                "stress_delta_mean": fmt(stress["mean"]),
                "stress_delta_median": fmt(stress["median"]),
                "consistency_score": fmt(consistency),
                "verdict_note": note,
            }
        )
    return report


def compute_target_bias_report(shared_rows):
    by_target = defaultdict(list)
    for row in shared_rows:
        by_target[row.get("target_key", "")].append(row)
    report = []
    for target, rows in sorted(by_target.items()):
        shared = [row for row in rows if row.get("shared_outcome") == "1"]
        report.append(
            {
                "target_key": target,
                "count": str(len(rows)),
                "shared_outcome_count": str(len(shared)),
                "shared_fraction": fmt(len(shared) / len(rows) if rows else 0.0),
                "state_distribution": json.dumps(dict(Counter(row.get("state", "") for row in rows)), sort_keys=True),
            }
        )
    return report


def decide_verdict(metrics, state_report, min_shared_fraction, min_prediction_success_fraction):
    if (
        metrics["duplicate_manifest_ids"]
        or metrics["duplicate_baseline_ids"]
        or metrics["duplicate_stress_ids"]
        or metrics["missing_manifest_ids"]
    ):
        return "a-v3-2c-bounded-validation-blocked-identity", "duplicate_or_missing_identity"
    if metrics["min_prediction_success_fraction"] < min_prediction_success_fraction:
        return "a-v3-2c-bounded-validation-blocked-runner", "prediction_success_below_required_fraction"
    if metrics["shared_fraction"] < min_shared_fraction:
        return "a-v3-2c-bounded-validation-blocked-coverage", "shared_fraction_below_required_fraction"
    if all(row["state"] == "unknown_unlabeled" for row in state_report):
        return "a-v3-2c-bounded-validation-fail", "all_rows_unknown_unlabeled"
    notes = Counter(row["verdict_note"] for row in state_report)
    has_sensitive = notes.get("high_error_sensitive", 0) > 0
    has_stable = notes.get("stable_control_like", 0) > 0
    if has_sensitive and has_stable:
        return "a-v3-2c-bounded-validation-pass", "sensitive_and_stable_states_present"
    if has_sensitive or has_stable:
        return "a-v3-2c-bounded-validation-weak-inconclusive", "partial_state_separation"
    return "a-v3-2c-bounded-validation-fail", "no_interpretable_state_separation"


def write_reports(out, metrics, state_report, variant_summaries, verdict, reason):
    lines = [
        "# A-v3.2c Bounded Outcome-Consistency Report",
        "",
        f"- manifest_row_count: {metrics['manifest_row_count']}",
        f"- variant_count: {metrics['variant_count']}",
        f"- shared_outcome_count: {metrics['shared_outcome_count']}",
        f"- shared_fraction: {metrics['shared_fraction']:.6f}",
        f"- min_prediction_success_fraction: {metrics['min_prediction_success_fraction']:.6f}",
        f"- verdict: `{verdict}`",
        f"- reason: `{reason}`",
        "",
        "## Variant Summary",
    ]
    for summary in variant_summaries:
        lines.append(
            f"- {summary['variant_id']}: rows={summary['row_count']}, "
            f"prediction_success={summary['prediction_success_count']}, "
            f"fraction={summary['prediction_success_fraction']:.6f}, "
            f"status={json.dumps(summary['status_counts'], sort_keys=True)}"
        )
    lines.extend(["", "## State Notes"])
    for row in state_report:
        lines.append(
            f"- {row['state']}: count={row['count']}, shared={row['shared_outcome_count']}, "
            f"mean_error={row['mean_error']}, stress_delta_mean={row['stress_delta_mean']}, "
            f"note={row['verdict_note']}"
        )
    lines.extend([
        "",
        "No training, finetuning, threshold tuning, B/C gate, fuzzy join, silent deduplication, full eval, submission packaging, or leaderboard probing was run.",
    ])
    write_text(out / "reports" / "outcome_consistency_report.md", "\n".join(lines) + "\n")
    write_text(
        out / "reports" / "bounded_go_no_go_verdict.md",
        "# A-v3.2c Bounded Outcome-Consistency Go/No-Go Verdict\n\n"
        f"verdict: `{verdict}`\n"
        f"reason: `{reason}`\n",
    )


def parse_stress(items):
    parsed = {}
    for item in items:
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stress", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-shared-fraction", type=float, default=0.90)
    parser.add_argument("--min-prediction-success-fraction", type=float, default=0.95)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    manifest_rows = read_csv_rows(args.manifest)
    baseline_rows = read_csv_rows(args.baseline)
    stress_paths = parse_stress(args.stress)
    stress_by_variant = {name: read_csv_rows(path) for name, path in stress_paths.items()}
    shared_rows, issues = build_shared_surface(manifest_rows, baseline_rows, stress_by_variant)

    variant_summaries = [compute_variant_summary(baseline_rows)]
    variant_summaries[0]["variant_id"] = "baseline"
    for variant, rows in stress_by_variant.items():
        summary = compute_variant_summary(rows)
        summary["variant_id"] = variant
        variant_summaries.append(summary)

    stress_variants = list(stress_by_variant)
    state_report = compute_state_report(shared_rows, stress_variants)
    target_report = compute_target_bias_report(shared_rows)
    min_pred_fraction = min((s["prediction_success_fraction"] for s in variant_summaries), default=0.0)
    manifest_count = len(manifest_rows)
    shared_count = issues["shared_outcome_count"]
    metrics = {
        **issues,
        "manifest_row_count": manifest_count,
        "variant_count": 1 + len(stress_by_variant),
        "shared_fraction": shared_count / manifest_count if manifest_count else 0.0,
        "min_prediction_success_fraction": min_pred_fraction,
        "state_count": len(state_report),
        "state_distribution": dict(Counter(row.get("state", "") for row in shared_rows)),
        "min_shared_fraction": args.min_shared_fraction,
        "min_prediction_success_fraction_required": args.min_prediction_success_fraction,
    }
    verdict, reason = decide_verdict(metrics, state_report, args.min_shared_fraction, args.min_prediction_success_fraction)
    metrics["verdict"] = verdict
    metrics["reason"] = reason

    shared_fields = [
        "canonical_pair_id",
        "target_key",
        "state",
        "baseline_status",
        "baseline_joint_error",
        "baseline_heading_error",
        "baseline_range_error",
    ]
    for variant in stress_variants:
        shared_fields.extend([
            f"{variant}_status",
            f"{variant}_joint_error",
            f"{variant}_heading_error",
            f"{variant}_range_error",
            f"{variant}_delta",
        ])
    shared_fields.append("shared_outcome")
    write_csv_rows(out / "tables" / "shared_outcome_surface.csv", shared_rows, shared_fields)
    write_csv_rows(out / "tables" / "state_wise_outcome_report.csv", state_report, [
        "state",
        "count",
        "prediction_success_count",
        "shared_outcome_count",
        "mean_error",
        "median_error",
        "p90_error",
        "p95_error",
        "heading_error",
        "range_error",
        "stress_delta_mean",
        "stress_delta_median",
        "consistency_score",
        "verdict_note",
    ])
    write_csv_rows(out / "tables" / "variant_summary.csv", [
        {
            **summary,
            "status_counts": json.dumps(summary["status_counts"], sort_keys=True),
            "prediction_success_fraction": fmt(summary["prediction_success_fraction"]),
        }
        for summary in variant_summaries
    ], ["variant_id", "row_count", "prediction_success_count", "prediction_success_fraction", "status_counts"])
    write_csv_rows(out / "tables" / "target_coverage_bias.csv", target_report, [
        "target_key",
        "count",
        "shared_outcome_count",
        "shared_fraction",
        "state_distribution",
    ])
    write_json(out / "metrics" / "outcome_consistency_metrics.json", metrics)
    write_reports(out, metrics, state_report, variant_summaries, verdict, reason)
    if verdict not in AUTHORIZED_VERDICTS:
        raise SystemExit(f"unauthorized verdict: {verdict}")


if __name__ == "__main__":
    main()
