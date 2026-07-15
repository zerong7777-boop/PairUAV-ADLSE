"""Audit deterministic repeatability for A-v3.2c fixed-manifest forward outputs."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def angle_delta(a, b):
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return None
    return abs((af - bf + 180.0) % 360.0 - 180.0)


def abs_delta(a, b):
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return None
    return abs(af - bf)


def summarize(values):
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {"mean": None, "median": None, "p95": None, "max": None}
    def q(frac):
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * frac
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return clean[int(pos)]
        return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)
    return {"mean": sum(clean) / len(clean), "median": q(0.5), "p95": q(0.95), "max": clean[-1]}


def fmt(value):
    return "" if value is None else f"{value:.12g}"


def compare_runs(reference_rows, candidate_rows, candidate_name):
    ref_by_id = {r["canonical_pair_id"]: r for r in reference_rows}
    heading = []
    rng = []
    joint = []
    same_prediction = 0
    same_identity = 0
    status_pairs = Counter()
    pair_rows = []
    for row in candidate_rows:
        cid = row.get("canonical_pair_id", "")
        ref = ref_by_id.get(cid, {})
        hd = angle_delta(row.get("prediction_heading"), ref.get("prediction_heading"))
        rd = abs_delta(row.get("prediction_range"), ref.get("prediction_range"))
        jd = abs_delta(row.get("joint_error"), ref.get("joint_error"))
        heading.append(hd)
        rng.append(rd)
        joint.append(jd)
        if all(row.get(k, "") == ref.get(k, "") for k in ("source_image_key", "target_image_key", "source_image_path", "target_image_path", "checkpoint_path")):
            same_identity += 1
        if hd == 0 and rd == 0:
            same_prediction += 1
        status_pairs[(ref.get("row_status", ""), row.get("row_status", ""))] += 1
        pair_rows.append({
            "comparison": candidate_name,
            "canonical_pair_id": cid,
            "heading_pred_abs_delta": fmt(hd),
            "range_pred_abs_delta": fmt(rd),
            "joint_error_abs_delta": fmt(jd),
            "reference_heading": ref.get("prediction_heading", ""),
            "candidate_heading": row.get("prediction_heading", ""),
            "reference_range": ref.get("prediction_range", ""),
            "candidate_range": row.get("prediction_range", ""),
        })
    hs = summarize(heading)
    rs = summarize(rng)
    js = summarize(joint)
    summary = {
        "comparison": candidate_name,
        "row_count": len(candidate_rows),
        "same_identity_count": same_identity,
        "same_prediction_count": same_prediction,
        "same_prediction_fraction": same_prediction / len(candidate_rows) if candidate_rows else 0.0,
        "heading_delta_mean": hs["mean"],
        "heading_delta_median": hs["median"],
        "heading_delta_p95": hs["p95"],
        "heading_delta_max": hs["max"],
        "range_delta_mean": rs["mean"],
        "range_delta_median": rs["median"],
        "range_delta_p95": rs["p95"],
        "joint_error_delta_mean": js["mean"],
        "joint_error_delta_median": js["median"],
        "joint_error_delta_p95": js["p95"],
        "status_pairs": {"|".join(k): v for k, v in status_pairs.items()},
    }
    return summary, pair_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run", action="append", required=True, help="name=csv")
    parser.add_argument("--load-diagnostics", default="")
    args = parser.parse_args()
    out = Path(args.output_dir)
    runs = []
    for item in args.run:
        name, path = item.split("=", 1)
        runs.append((name, read_csv(path)))
    reference_name, reference_rows = runs[0]
    summaries = []
    pair_rows = []
    for name, rows in runs[1:]:
        summary, pairs = compare_runs(reference_rows, rows, f"{reference_name}_vs_{name}")
        summaries.append(summary)
        pair_rows.extend(pairs[:500])

    max_heading = max((s["heading_delta_max"] or 0.0 for s in summaries), default=0.0)
    max_range = max((s["range_delta_p95"] or 0.0 for s in summaries), default=0.0)
    min_same_fraction = min((s["same_prediction_fraction"] for s in summaries), default=1.0)
    if min_same_fraction == 1.0 and max_heading == 0.0 and max_range == 0.0:
        verdict = "deterministic-repeatability-pass"
        reason = "all_repeated_forwards_identical"
    else:
        verdict = "deterministic-repeatability-fail"
        reason = "same_config_repeated_forwards_change_predictions"

    load_diag = {}
    if args.load_diagnostics and Path(args.load_diagnostics).exists():
        load_diag = json.loads(Path(args.load_diagnostics).read_text(encoding="utf-8"))

    metrics = {
        "verdict": verdict,
        "reason": reason,
        "reference_run": reference_name,
        "repeat_count": len(runs),
        "comparison_count": len(summaries),
        "min_same_prediction_fraction": min_same_fraction,
        "max_heading_delta": max_heading,
        "max_range_delta_p95": max_range,
        "load_diagnostics": load_diag,
        "comparisons": summaries,
    }
    write_json(out / "metrics" / "deterministic_repeatability_metrics.json", metrics)
    write_csv(out / "tables" / "repeatability_delta_summary.csv", [
        {
            **{k: fmt(v) if isinstance(v, float) else v for k, v in s.items() if k != "status_pairs"},
            "status_pairs": json.dumps(s["status_pairs"], sort_keys=True),
        }
        for s in summaries
    ], [
        "comparison",
        "row_count",
        "same_identity_count",
        "same_prediction_count",
        "same_prediction_fraction",
        "heading_delta_mean",
        "heading_delta_median",
        "heading_delta_p95",
        "heading_delta_max",
        "range_delta_mean",
        "range_delta_median",
        "range_delta_p95",
        "joint_error_delta_mean",
        "joint_error_delta_median",
        "joint_error_delta_p95",
        "status_pairs",
    ])
    write_csv(out / "tables" / "repeatability_pair_delta_sample.csv", pair_rows, [
        "comparison",
        "canonical_pair_id",
        "heading_pred_abs_delta",
        "range_pred_abs_delta",
        "joint_error_abs_delta",
        "reference_heading",
        "candidate_heading",
        "reference_range",
        "candidate_range",
    ])
    lines = [
        "# A-v3.2c Deterministic Repeatability Audit",
        "",
        f"verdict: `{verdict}`",
        f"reason: `{reason}`",
        "",
        f"- repeat_count: {len(runs)}",
        f"- min_same_prediction_fraction: {min_same_fraction:.6f}",
        f"- max_heading_delta: {max_heading:.6f}",
        f"- max_range_delta_p95: {max_range:.6f}",
        "",
        "## Load Diagnostics",
        "",
        f"- missing_key_count: {load_diag.get('missing_key_count', '')}",
        f"- unexpected_key_count: {load_diag.get('unexpected_key_count', '')}",
        f"- missing_key_sample: `{load_diag.get('missing_key_sample', [])}`",
        f"- unexpected_key_sample: `{load_diag.get('unexpected_key_sample', [])}`",
        "",
        "## Comparisons",
    ]
    for s in summaries:
        lines.append(
            f"- {s['comparison']}: same_prediction_fraction={s['same_prediction_fraction']:.6f}, "
            f"heading_delta_mean={fmt(s['heading_delta_mean'])}, heading_delta_max={fmt(s['heading_delta_max'])}, "
            f"range_delta_mean={fmt(s['range_delta_mean'])}, joint_error_delta_mean={fmt(s['joint_error_delta_mean'])}"
        )
    lines.extend([
        "",
        "No training, finetuning, threshold tuning, full eval, B/C gate, submission packaging, fuzzy join, or silent deduplication was run.",
    ])
    (out / "reports" / "deterministic_repeatability_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out / "reports" / "deterministic_repeatability_audit_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
