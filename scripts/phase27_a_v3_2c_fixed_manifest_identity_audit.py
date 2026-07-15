"""Audit tiny fixed-manifest runner outputs."""
import argparse
import csv
import json
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


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_outputs(manifest_rows, outcome_sets):
    manifest_count = len(manifest_rows)
    manifest_hashes = {r.get("manifest_hash", "") for r in manifest_rows}
    audit_rows = []
    blocked = []
    for variant_id, rows in outcome_sets:
        keys = [(r.get("canonical_pair_id", ""), r.get("variant_id", "")) for r in rows]
        status_counts = Counter(r.get("row_status", "") for r in rows)
        missing_identity = sum(1 for r in rows if not r.get("canonical_pair_id"))
        missing_source_target = sum(1 for r in rows if not r.get("source_image_key") or not r.get("target_image_key"))
        duplicate_count = len(keys) - len(set(keys))
        hash_mismatch = sum(1 for r in rows if r.get("manifest_hash", "") not in manifest_hashes)
        row_count_match = len(rows) == manifest_count
        passed = row_count_match and missing_identity == 0 and missing_source_target == 0 and duplicate_count == 0 and hash_mismatch == 0 and all(r.get("row_status") for r in rows)
        if not passed:
            blocked.append(variant_id)
        audit_rows.append(
            {
                "variant_id": variant_id,
                "manifest_row_count": str(manifest_count),
                "output_row_count": str(len(rows)),
                "row_count_match": str(row_count_match).lower(),
                "missing_canonical_pair_id_count": str(missing_identity),
                "missing_source_target_identity_count": str(missing_source_target),
                "duplicate_canonical_variant_count": str(duplicate_count),
                "manifest_hash_mismatch_count": str(hash_mismatch),
                "row_status_distribution": json.dumps(dict(status_counts), sort_keys=True),
                "pass": str(passed).lower(),
            }
        )
    if not blocked:
        verdict = "fixed-manifest-runner-smoke-pass"
        reason = "all_variants_preserve_identity_row_count_and_hash"
    else:
        verdict = "fixed-manifest-runner-blocked-metadata-loss"
        reason = "one_or_more_variants_failed_identity_audit"
    metrics = {
        "manifest_row_count": manifest_count,
        "variant_count": len(outcome_sets),
        "passing_variant_count": sum(1 for r in audit_rows if r["pass"] == "true"),
        "verdict": verdict,
        "reason": reason,
    }
    return audit_rows, metrics


def write_reports(out, audit_rows, metrics):
    lines = [
        "# A-v3.2c Fixed-Manifest Eval Identity Audit Tiny",
        "",
        f"- manifest_row_count: {metrics['manifest_row_count']}",
        f"- variant_count: {metrics['variant_count']}",
        f"- passing_variant_count: {metrics['passing_variant_count']}",
        f"- verdict: `{metrics['verdict']}`",
        f"- reason: `{metrics['reason']}`",
        "",
        "No model training, finetuning, 10k eval, outcome-consistency audit, B/C gate, fuzzy join, or silent deduplication was run.",
    ]
    (out / "reports" / "fixed_manifest_eval_identity_audit_tiny.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "reports" / "runner_go_no_go_verdict.md").write_text(
        "# A-v3.2c Runner Go/No-Go Verdict\n\n"
        + f"verdict: `{metrics['verdict']}`\n"
        + f"reason: `{metrics['reason']}`\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--outcome", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outcome_sets = []
    for item in args.outcome:
        name, path = item.split("=", 1)
        outcome_sets.append((name, read_csv(path)))
    out = Path(args.output_dir)
    audit_rows, metrics = audit_outputs(read_csv(args.fixed_manifest), outcome_sets)
    write_csv(out / "tables" / "fixed_manifest_eval_identity_audit_tiny.csv", audit_rows, [
        "variant_id",
        "manifest_row_count",
        "output_row_count",
        "row_count_match",
        "missing_canonical_pair_id_count",
        "missing_source_target_identity_count",
        "duplicate_canonical_variant_count",
        "manifest_hash_mismatch_count",
        "row_status_distribution",
        "pass",
    ])
    write_json(out / "metrics" / "fixed_manifest_eval_identity_audit_tiny.json", metrics)
    write_reports(out, audit_rows, metrics)


if __name__ == "__main__":
    main()

