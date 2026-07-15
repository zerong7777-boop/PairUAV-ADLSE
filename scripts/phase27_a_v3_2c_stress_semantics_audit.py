"""Audit whether A-v3.2c stress variants are semantic stress probes."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


DEFAULT_OUT = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_v3_2c_stress_semantics_audit"
DEFAULT_SURFACE = "/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_v3_2c_bounded_outcome_consistency_reacquired_state"
DEFAULT_RUNNER = "/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav/scripts/phase27_a_v3_2c_fixed_manifest_eval_runner.py"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, columns):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


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
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    def q(frac):
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * frac
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return clean[int(pos)]
        return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)
    return {
        "mean": sum(clean) / len(clean),
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "max": clean[-1],
    }


def fmt(value):
    return "" if value is None else f"{value:.12g}"


def config_semantics(config_dir):
    rows = []
    payloads = {}
    for path in sorted(Path(config_dir).glob("*.json")):
        payload = read_json(path)
        payloads[path.stem] = payload
        rows.append({"variant": path.stem, "config": json.dumps(payload, sort_keys=True)})
    all_keys = sorted({k for payload in payloads.values() for k in payload})
    invariant_keys = []
    variant_keys = []
    for key in all_keys:
        vals = {json.dumps(payloads[name].get(key), sort_keys=True) for name in payloads}
        if len(vals) == 1:
            invariant_keys.append(key)
        else:
            variant_keys.append(key)
    return rows, {"invariant_keys": invariant_keys, "variant_keys": variant_keys, "payloads": payloads}


def runner_usage(runner_path):
    text = Path(runner_path).read_text(encoding="utf-8")
    markers = {
        "json_load_count": text.count("json.load"),
        "variant_config_occurrences": text.count("variant_config"),
        "file_hash_variant_config_only": "file_hash(variant_config)" in text,
        "opens_variant_config": "open(variant_config" in text or "Path(variant_config).open" in text,
        "uses_variant_id_in_model_forward": "variant_id" in text[text.find("def run_model_forward_tiny"):text.find("def main")],
    }
    usage_lines = []
    for i, line in enumerate(text.splitlines(), 1):
        if "variant_config" in line or "variant_id" in line or "json.load" in line:
            usage_lines.append({"line": i, "text": line.strip()})
    return markers, usage_lines


def compare_predictions(surface_dir):
    surface = Path(surface_dir)
    baseline = read_csv(surface / "tables" / "baseline_on_fixed_manifest_bounded.csv")
    base_by_id = {row["canonical_pair_id"]: row for row in baseline}
    variants = []
    summary_rows = []
    pair_rows = []
    for path in sorted((surface / "tables").glob("stress*_on_fixed_manifest_bounded.csv")):
        variant = path.name.replace("_on_fixed_manifest_bounded.csv", "")
        variants.append(variant)
        rows = read_csv(path)
        heading_deltas = []
        range_deltas = []
        joint_error_deltas = []
        same_pred = 0
        same_identity = 0
        status_pairs = Counter()
        for row in rows:
            cid = row.get("canonical_pair_id", "")
            base = base_by_id.get(cid, {})
            if all(row.get(k, "") == base.get(k, "") for k in ("source_image_key", "target_image_key", "source_image_path", "target_image_path", "checkpoint_path")):
                same_identity += 1
            hd = angle_delta(row.get("prediction_heading"), base.get("prediction_heading"))
            rd = abs_delta(row.get("prediction_range"), base.get("prediction_range"))
            jd = abs_delta(row.get("joint_error"), base.get("joint_error"))
            heading_deltas.append(hd)
            range_deltas.append(rd)
            joint_error_deltas.append(jd)
            if hd == 0 and rd == 0:
                same_pred += 1
            status_pairs[(base.get("row_status", ""), row.get("row_status", ""))] += 1
            pair_rows.append({
                "variant": variant,
                "canonical_pair_id": cid,
                "same_identity": str(bool(base)).lower(),
                "heading_pred_abs_delta": fmt(hd),
                "range_pred_abs_delta": fmt(rd),
                "joint_error_abs_delta": fmt(jd),
                "baseline_heading": base.get("prediction_heading", ""),
                "stress_heading": row.get("prediction_heading", ""),
                "baseline_range": base.get("prediction_range", ""),
                "stress_range": row.get("prediction_range", ""),
            })
        hsum = summarize(heading_deltas)
        rsum = summarize(range_deltas)
        jsum = summarize(joint_error_deltas)
        summary_rows.append({
            "variant": variant,
            "row_count": str(len(rows)),
            "same_identity_count": str(same_identity),
            "same_prediction_count": str(same_pred),
            "same_prediction_fraction": fmt(same_pred / len(rows) if rows else 0.0),
            "heading_delta_mean": fmt(hsum["mean"]),
            "heading_delta_median": fmt(hsum["median"]),
            "heading_delta_p95": fmt(hsum["p95"]),
            "heading_delta_max": fmt(hsum["max"]),
            "range_delta_mean": fmt(rsum["mean"]),
            "range_delta_median": fmt(rsum["median"]),
            "range_delta_p95": fmt(rsum["p95"]),
            "joint_error_delta_mean": fmt(jsum["mean"]),
            "joint_error_delta_median": fmt(jsum["median"]),
            "joint_error_delta_p95": fmt(jsum["p95"]),
            "status_pairs": json.dumps({"|".join(k): v for k, v in status_pairs.items()}, sort_keys=True),
        })
    return variants, summary_rows, pair_rows


def write_report(out, metrics, config_rows, runner_lines, summary_rows):
    lines = [
        "# A-v3.2c Stress-Semantics Audit",
        "",
        f"verdict: `{metrics['verdict']}`",
        f"reason: `{metrics['reason']}`",
        "",
        "## Config Semantics",
        "",
        f"- variant_config_keys_that_change: `{metrics['config_semantics']['variant_keys']}`",
        f"- invariant_config_keys: `{metrics['config_semantics']['invariant_keys']}`",
        "",
        "## Runner Usage",
        "",
        f"- json_load_count: `{metrics['runner_usage']['markers']['json_load_count']}`",
        f"- opens_variant_config: `{metrics['runner_usage']['markers']['opens_variant_config']}`",
        f"- file_hash_variant_config_only: `{metrics['runner_usage']['markers']['file_hash_variant_config_only']}`",
        "",
        "## Prediction Delta Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['variant']}: same_identity={row['same_identity_count']}/{row['row_count']}, "
            f"same_prediction_fraction={row['same_prediction_fraction']}, "
            f"heading_delta_mean={row['heading_delta_mean']}, "
            f"range_delta_mean={row['range_delta_mean']}, "
            f"joint_error_delta_mean={row['joint_error_delta_mean']}"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Current stress configs only vary metadata such as `variant_id`; they do not encode a consumed perturbation parameter.",
        "- The fixed-manifest runner hashes the config but does not read config contents into model, data, augmentation, or metric behavior.",
        "- Therefore these stress-labeled variants are not valid semantic stress probes.",
        "- Large prediction deltas under nominally identical inputs indicate repeated forward instability or uncontrolled stochasticity, not a controlled stress mechanism.",
        "",
        "No training, finetuning, threshold tuning, full eval, B/C gate, submission packaging, fuzzy join, or silent deduplication was run.",
    ])
    (out / "reports" / "stress_semantics_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-dir", default=DEFAULT_SURFACE)
    parser.add_argument("--runner", default=DEFAULT_RUNNER)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    config_rows, config_metrics = config_semantics(Path(args.surface_dir) / "configs")
    runner_markers, runner_lines = runner_usage(args.runner)
    variants, summary_rows, pair_rows = compare_predictions(args.surface_dir)

    only_metadata_changes = set(config_metrics["variant_keys"]).issubset({"variant_id"}) and not runner_markers["opens_variant_config"]
    nonzero_delta_variants = [
        row["variant"] for row in summary_rows
        if safe_float(row["heading_delta_mean"]) not in (None, 0.0) or safe_float(row["range_delta_mean"]) not in (None, 0.0)
    ]
    if only_metadata_changes and nonzero_delta_variants:
        verdict = "stress-semantics-invalid-uncontrolled-repeat-forward"
        reason = "configs_do_not_define_consumed_stress_but_predictions_change"
    elif only_metadata_changes:
        verdict = "stress-semantics-invalid-noop"
        reason = "configs_do_not_define_consumed_stress_and_predictions_are_same"
    else:
        verdict = "stress-semantics-needs-manual-review"
        reason = "configs_or_runner_show_nontrivial_variant_inputs"

    metrics = {
        "verdict": verdict,
        "reason": reason,
        "surface_dir": args.surface_dir,
        "runner": args.runner,
        "variants": variants,
        "config_semantics": config_metrics,
        "runner_usage": {"markers": runner_markers, "lines": runner_lines},
        "prediction_delta_summary": summary_rows,
    }
    write_json(out / "metrics" / "stress_semantics_audit_metrics.json", metrics)
    write_csv(out / "tables" / "stress_config_semantics.csv", config_rows, ["variant", "config"])
    write_csv(out / "tables" / "stress_prediction_delta_summary.csv", summary_rows, [
        "variant",
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
    write_csv(out / "tables" / "stress_pair_delta_sample.csv", pair_rows[:2000], [
        "variant",
        "canonical_pair_id",
        "same_identity",
        "heading_pred_abs_delta",
        "range_pred_abs_delta",
        "joint_error_abs_delta",
        "baseline_heading",
        "stress_heading",
        "baseline_range",
        "stress_range",
    ])
    write_report(out, metrics, config_rows, runner_lines, summary_rows)
    print((out / "reports" / "stress_semantics_audit_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
