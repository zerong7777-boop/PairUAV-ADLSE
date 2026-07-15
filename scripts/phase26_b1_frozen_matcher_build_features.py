#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from reloc3r.datasets.pairuav_matcher_features import (
    MATCHER_FEATURE_NAMES,
    apply_normalization,
    compute_normalization,
    extract_matcher_features,
    read_manifest_rows,
    sample_to_match_path,
)


def dedupe(rows):
    seen = set()
    result = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        result.append(row)
    return result


def build_raw_rows(rows, cache_root):
    output = []
    for row in dedupe(rows):
        sample_id = row["sample_id"]
        match_path = sample_to_match_path(cache_root, sample_id)
        raw = extract_matcher_features(match_path, image_width=512, image_height=512)
        output.append(
            {
                "sample_id": sample_id,
                "match_path": str(match_path),
                "raw_features": raw,
                "fallback_used": bool(raw.get("fallback_used", 0.0) >= 0.5),
            }
        )
    return output


def write_jsonl(path, rows, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            features = apply_normalization(row["raw_features"], stats)
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "features": features,
                        "raw_features": row["raw_features"],
                        "match_path": row["match_path"],
                        "fallback_used": row["fallback_used"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def coverage(rows):
    total = len(rows)
    fallback = sum(1 for row in rows if row["fallback_used"])
    covered = total - fallback
    return {
        "rows": total,
        "covered": covered,
        "fallback": fallback,
        "coverage_rate": covered / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--schema-report", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    schema = json.loads(Path(args.schema_report).read_text(encoding="utf-8"))
    if schema.get("schema_verdict") != "usable":
        raise SystemExit(f"Schema report is not usable: {schema.get('schema_verdict')}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest_rows(args.train_manifest, split="train")
    eval_rows = read_manifest_rows(args.eval_manifest)
    raw_train = build_raw_rows(train_rows, args.cache_root)
    raw_eval = build_raw_rows(eval_rows, args.cache_root)
    stats = compute_normalization(raw_train)

    write_jsonl(output_root / "train_features.jsonl", raw_train, stats)
    write_jsonl(output_root / "eval_features.jsonl", raw_eval, stats)

    feature_schema = {
        "feature_names": MATCHER_FEATURE_NAMES,
        "feature_dim": len(MATCHER_FEATURE_NAMES),
        "normalization": stats,
        "normalization_source": "train_manifest",
        "cache_root": str(Path(args.cache_root)),
        "schema_report": str(Path(args.schema_report)),
    }
    (output_root / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "train": coverage(raw_train),
        "eval": coverage(raw_eval),
        "train_manifest": str(Path(args.train_manifest)),
        "eval_manifest": str(Path(args.eval_manifest)),
        "output_root": str(output_root),
    }
    (output_root / "coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
