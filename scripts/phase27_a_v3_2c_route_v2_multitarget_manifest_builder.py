"""Build a stratified multi-target fixed manifest for route-v2 A audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "manifest_version",
    "manifest_hash",
    "manifest_row_id",
    "canonical_pair_id",
    "source_image_key",
    "target_image_key",
    "source_image_path",
    "target_image_path",
    "pair_direction",
    "target_key",
    "group_id",
    "scene_key",
    "split_key",
    "gt_heading",
    "gt_range",
    "json_path",
    "pair_key",
    "candidate_state",
    "candidate_flags_json",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("canonical_pair_id", ""): row for row in rows if row.get("canonical_pair_id", "")}


def parse_flags(value: str) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"FLAG_PARSE_ERROR": "True"}
    return {str(k): str(v) for k, v in parsed.items()}


def is_true(flags: dict[str, str], key: str) -> bool:
    return flags.get(key, "").lower() == "true"


def flag_state(flags: dict[str, str]) -> str:
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


def reacquire_state(source_row: dict[str, str], evidence_row: dict[str, str] | None, training_row: dict[str, str] | None) -> str:
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
    return flag_state(parse_flags(source_row.get("candidate_flags_json", "")))


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    columns = [col for col in MANIFEST_COLUMNS if col != "manifest_hash"]
    h = hashlib.sha256()
    for row in rows:
        h.update("\t".join(str(row.get(col, "")) for col in columns).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def target_id(row: dict[str, str]) -> str:
    return row.get("target_key_metadata") or row.get("group_id") or row.get("target_key") or "unknown_target"


def build(args: argparse.Namespace):
    source_rows = read_csv(args.source_manifest)
    full_by_id = index_by_id(read_csv(args.full_dev_surface))
    evidence_by_id = index_by_id(read_csv(args.evidence_manifest))
    training_by_id = index_by_id(read_csv(args.training_manifest))
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_target[target_id(row)].append(row)

    selected = []
    skipped_missing_gt = 0
    skipped_missing_images = 0
    for target in sorted(by_target):
        count = 0
        for row in by_target[target]:
            if count >= args.per_target:
                break
            cid = row.get("canonical_pair_id", "")
            full = full_by_id.get(cid)
            if not full or not full.get("analysis_only_gt_angle") or not full.get("analysis_only_gt_distance"):
                skipped_missing_gt += 1
                continue
            source_key = row.get("source_image_key") or row.get("source_key", "")
            target_key = row.get("target_image_key") or row.get("target_key", "")
            if not source_key or not target_key:
                skipped_missing_images += 1
                continue
            evidence = evidence_by_id.get(cid)
            training = training_by_id.get(cid)
            flags = parse_flags(row.get("candidate_flags_json", ""))
            selected.append(
                {
                    "manifest_version": "phase27_a_v3_2c_route_v2_multitarget_v1",
                    "manifest_hash": "",
                    "manifest_row_id": str(len(selected)),
                    "canonical_pair_id": cid,
                    "source_image_key": source_key,
                    "target_image_key": target_key,
                    "source_image_path": source_key,
                    "target_image_path": target_key,
                    "pair_direction": row.get("pair_direction", "ordered_source_target"),
                    "target_key": target,
                    "group_id": row.get("group_id", target),
                    "scene_key": row.get("scene_key", ""),
                    "split_key": row.get("split_key", ""),
                    "gt_heading": full.get("analysis_only_gt_angle", ""),
                    "gt_range": full.get("analysis_only_gt_distance", ""),
                    "json_path": row.get("json_path", ""),
                    "pair_key": row.get("pair_key", ""),
                    "candidate_state": row.get("candidate_state", ""),
                    "candidate_flags_json": row.get("candidate_flags_json", ""),
                    "reacquired_state": reacquire_state(row, evidence, training),
                    "flag_state": flag_state(flags),
                    "evidence_base_regime": evidence.get("base_regime", "") if evidence else "",
                    "evidence_joined": "1" if evidence else "0",
                    "training_joined": "1" if training else "0",
                    "ready_control_preservation": flags.get("READY_CONTROL_PRESERVATION", ""),
                    "ready_heading_hard_training": flags.get("READY_HEADING_HARD_TRAINING", ""),
                    "ready_range_hard_training": flags.get("READY_RANGE_HARD_TRAINING", ""),
                    "not_ready": flags.get("NOT_READY", ""),
                    "analysis_only": flags.get("ANALYSIS_ONLY", ""),
                }
            )
            count += 1
    digest = manifest_hash(selected)
    for row in selected:
        row["manifest_hash"] = digest

    target_distribution = Counter(row["target_key"] for row in selected)
    state_distribution = Counter(row["reacquired_state"] for row in selected)
    metrics = {
        "verdict": "route-v2-multitarget-manifest-ready" if len(target_distribution) >= 2 and len(state_distribution) >= 2 else "route-v2-multitarget-manifest-weak",
        "source_manifest": str(args.source_manifest),
        "full_dev_surface": str(args.full_dev_surface),
        "evidence_manifest": str(args.evidence_manifest),
        "training_manifest": str(args.training_manifest),
        "per_target": args.per_target,
        "row_count": len(selected),
        "unique_canonical_pair_id_count": len({row["canonical_pair_id"] for row in selected}),
        "target_count": len(target_distribution),
        "target_distribution": dict(target_distribution),
        "state_count": len(state_distribution),
        "state_distribution": dict(state_distribution),
        "skipped_missing_gt": skipped_missing_gt,
        "skipped_missing_images": skipped_missing_images,
        "manifest_hash": digest,
    }
    return selected, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--full-dev-surface", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--per-target", type=int, default=512)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    args = parser.parse_args()
    rows, metrics = build(args)
    write_csv(args.output_manifest, rows, MANIFEST_COLUMNS)
    write_json(args.metrics_json, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
