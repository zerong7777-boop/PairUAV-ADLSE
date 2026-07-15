from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .common import (
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAIN_JSON_ROOT,
    DEFAULT_VAL_JSON_ROOT,
    circular_abs_error_deg,
    ensure_run_root,
    iter_json_paths,
    load_pair_json,
    summarize_numeric,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase91 G1 coordinate convention audit.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--json-root", type=Path, action="append", default=None)
    parser.add_argument("--max-examples", type=int, default=200)
    return parser.parse_args()


def load_records(roots: list[Path]) -> list[dict]:
    records = []
    for root in roots:
        for path in iter_json_paths(root):
            row = load_pair_json(path)
            row["source_root"] = str(root)
            records.append(row)
    return records


def audit(records: list[dict], max_examples: int) -> tuple[dict, list[dict]]:
    by_direct_key: dict[tuple[str, str], dict] = {}
    duplicate_keys = 0
    for row in records:
        key = (row["image_a"], row["image_b"])
        if key in by_direct_key:
            duplicate_keys += 1
        by_direct_key[key] = row

    seen_pairs = set()
    reverse_examples = []
    raw_range_sym = []
    abs_range_sym = []
    signed_range_antisym = []
    heading_inverse = []

    for row in records:
        key = (row["image_a"], row["image_b"])
        reverse_key = (row["image_b"], row["image_a"])
        pair_key = tuple(sorted([key, reverse_key]))
        if pair_key in seen_pairs:
            continue
        reverse = by_direct_key.get(reverse_key)
        if reverse is None:
            continue
        seen_pairs.add(pair_key)
        raw_range_err = abs(row["range_value"] - reverse["range_value"])
        abs_range_err = abs(abs(row["range_value"]) - abs(reverse["range_value"]))
        signed_antisym_err = abs(row["range_value"] + reverse["range_value"])
        heading_err = circular_abs_error_deg(reverse["heading_deg"], row["heading_deg"] + 180.0)
        raw_range_sym.append(raw_range_err)
        abs_range_sym.append(abs_range_err)
        signed_range_antisym.append(signed_antisym_err)
        heading_inverse.append(heading_err)
        if len(reverse_examples) < max_examples:
            reverse_examples.append(
                {
                    "sample_id": row["sample_id"],
                    "reverse_sample_id": reverse["sample_id"],
                    "image_a": row["image_a"],
                    "image_b": row["image_b"],
                    "heading_deg": row["heading_deg"],
                    "reverse_heading_deg": reverse["heading_deg"],
                    "heading_inverse_abs_error_deg": heading_err,
                    "range_value": row["range_value"],
                    "reverse_range_value": reverse["range_value"],
                    "raw_range_symmetry_abs_error": raw_range_err,
                    "abs_range_symmetry_abs_error": abs_range_err,
                    "signed_range_antisymmetry_abs_error": signed_antisym_err,
                }
            )

    reverse_count = len(heading_inverse)
    heading_mean = summarize_numeric(heading_inverse)["mean"]
    raw_mean = summarize_numeric(raw_range_sym)["mean"]
    abs_mean = summarize_numeric(abs_range_sym)["mean"]
    anti_mean = summarize_numeric(signed_range_antisym)["mean"]

    if reverse_count == 0:
        verdict = "ambiguous"
        convention = "no_reverse_pairs_found"
        allow_reversal_loss = False
    elif heading_mean is not None and heading_mean <= 1.0 and (
        (raw_mean is not None and raw_mean <= 1e-4) or (anti_mean is not None and anti_mean <= 1e-4)
    ):
        verdict = "pass"
        convention = "unsigned_range_symmetric" if raw_mean <= 1e-4 else "signed_range_antisymmetric"
        allow_reversal_loss = True
    elif heading_mean is not None and heading_mean <= 5.0 and abs_mean is not None and abs_mean <= 1e-4:
        verdict = "ambiguous"
        convention = "heading_inverse_with_abs_range_symmetry_only"
        allow_reversal_loss = False
    else:
        verdict = "fail"
        convention = "reverse_pairs_do_not_match_expected_polar_inverse"
        allow_reversal_loss = False

    payload = {
        "phase": "phase91_g1_coordinate_convention_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "record_count": len(records),
        "unique_direct_pair_count": len(by_direct_key),
        "duplicate_direct_pair_count": duplicate_keys,
        "reverse_pair_count": reverse_count,
        "heading_inverse_abs_error_deg": summarize_numeric(heading_inverse),
        "raw_range_symmetry_abs_error": summarize_numeric(raw_range_sym),
        "abs_range_symmetry_abs_error": summarize_numeric(abs_range_sym),
        "signed_range_antisymmetry_abs_error": summarize_numeric(signed_range_antisym),
        "convention_verdict": verdict,
        "convention_type": convention,
        "allow_reversal_loss": allow_reversal_loss,
    }
    return payload, reverse_examples


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    ensure_run_root(run_root)
    roots = args.json_root or [DEFAULT_TRAIN_JSON_ROOT, DEFAULT_VAL_JSON_ROOT]
    records = load_records(roots)
    payload, examples = audit(records, max_examples=args.max_examples)
    payload["json_roots"] = [str(root) for root in roots]

    write_json(run_root / "audits" / "coordinate_convention_audit.json", payload)
    write_csv(run_root / "diagnostics" / "reverse_pair_examples.csv", examples)

    md = [
        "# Phase91 G1 Coordinate Convention Audit",
        "",
        f"- json_roots: `{', '.join(str(root) for root in roots)}`",
        f"- record_count: {payload['record_count']}",
        f"- reverse_pair_count: {payload['reverse_pair_count']}",
        f"- convention_verdict: `{payload['convention_verdict']}`",
        f"- convention_type: `{payload['convention_type']}`",
        f"- allow_reversal_loss: {payload['allow_reversal_loss']}",
        "",
        "## Heading Inverse Error",
        "",
        "```json",
        json.dumps(payload["heading_inverse_abs_error_deg"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Range Convention Diagnostics",
        "",
        "```json",
        json.dumps(
            {
                "raw_range_symmetry_abs_error": payload["raw_range_symmetry_abs_error"],
                "abs_range_symmetry_abs_error": payload["abs_range_symmetry_abs_error"],
                "signed_range_antisymmetry_abs_error": payload["signed_range_antisymmetry_abs_error"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "Reversal consistency loss is disabled unless `convention_verdict == pass`.",
    ]
    write_text(run_root / "audits" / "coordinate_convention_audit.md", "\n".join(md))
    print(json.dumps({"verdict": payload["convention_verdict"], "reverse_pair_count": payload["reverse_pair_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

