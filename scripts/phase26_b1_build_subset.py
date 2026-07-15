#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from reloc3r.datasets.pairuav import PairUAV


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_row(row):
    group = str(row["image_a"]).split("/")[0]
    return f"{group}/{row['json_id']}"


def find_json(root, sample_id):
    group, json_id = sample_id.split("/", 1)
    path = Path(root) / group / f"{json_id}.json"
    return path if path.is_file() else None


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def boolish(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def phase11_label(row):
    if boolish(row.get("high_sim_high_error", "0")):
        return "semantic_decoupled_high_sim_high_error"
    if boolish(row.get("low_sim_low_error", "0")):
        return "semantic_control_low_sim_low_error"
    if row.get("within_controlled_scope"):
        return "semantic_control_other"
    return "semantic_unlabeled"


def phase14_label(row):
    residual = row.get("residual_target_dominance_label", "")
    if residual and residual != "middle":
        return "alignment_unit_sensitive"
    if residual == "middle":
        return "alignment_unit_control_middle"
    return "alignment_unit_unlabeled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase11-controlled-csv", required=True)
    parser.add_argument("--phase14-surface-csv", required=True)
    parser.add_argument("--train-json-root", required=True)
    parser.add_argument("--val-json-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-background-train", type=int, default=512)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_out = output_root / "train_json"
    eval_out = output_root / "eval_json"

    phase11_by_id = {sample_id_from_row(row): row for row in read_csv(args.phase11_controlled_csv)}
    phase14_by_id = {sample_id_from_row(row): row for row in read_csv(args.phase14_surface_csv)}
    eval_ids = sorted(set(phase11_by_id) | set(phase14_by_id))

    rows = []
    selected_eval = []
    for sample_id in eval_ids:
        src = find_json(args.train_json_root, sample_id) or find_json(args.val_json_root, sample_id)
        if src is None:
            continue
        group, json_id = sample_id.split("/", 1)
        dst = eval_out / group / f"{json_id}.json"
        link_or_copy(src, dst)
        p11 = phase11_label(phase11_by_id.get(sample_id, {}))
        p14 = phase14_label(phase14_by_id.get(sample_id, {}))
        rows.append(
            {
                "split": "eval",
                "sample_id": sample_id,
                "source_json": str(src),
                "target_json": str(dst),
                "phase11_label": p11,
                "phase14_label": p14,
            }
        )
        selected_eval.append(sample_id)

    if not selected_eval:
        raise SystemExit("No Phase11/Phase14 overlap samples found in train/val JSON roots")

    copied_train = 0
    for group_dir in sorted(Path(args.train_json_root).iterdir()):
        if not group_dir.is_dir():
            continue
        for src in sorted(group_dir.glob("*.json")):
            if copied_train >= args.max_background_train:
                break
            sample_id = f"{group_dir.name}/{src.stem}"
            dst = train_out / group_dir.name / src.name
            link_or_copy(src, dst)
            rows.append(
                {
                    "split": "train",
                    "sample_id": sample_id,
                    "source_json": str(src),
                    "target_json": str(dst),
                    "phase11_label": "background",
                    "phase14_label": "background",
                }
            )
            copied_train += 1
        if copied_train >= args.max_background_train:
            break

    for sample_id in selected_eval:
        src = find_json(args.train_json_root, sample_id) or find_json(args.val_json_root, sample_id)
        group, json_id = sample_id.split("/", 1)
        dst = train_out / group / f"{json_id}.json"
        link_or_copy(src, dst)
        rows.append(
            {
                "split": "train",
                "sample_id": sample_id,
                "source_json": str(src),
                "target_json": str(dst),
                "phase11_label": "eval_overlap",
                "phase14_label": "eval_overlap",
            }
        )

    manifest_path = output_root / "subset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "sample_id", "source_json", "target_json", "phase11_label", "phase14_label"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset = PairUAV(
        json_root=str(eval_out),
        image_root=args.image_root,
        split="dev",
        resolution=(512, 384),
        seed=777,
        require_labels=True,
    )
    official_manifest = output_root / "eval_official_manifest.csv"
    with official_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "gt_angle", "gt_distance"])
        writer.writeheader()
        for sample in dataset.samples:
            writer.writerow(
                {
                    "sample_id": f"{sample['group_id']}/{sample['json_id']}",
                    "gt_angle": sample["heading_deg"],
                    "gt_distance": sample["range_value"],
                }
            )

    eval_id_set = set(selected_eval)
    summary = {
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "subset_manifest_eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "official_eval_rows": len(dataset.samples),
        "phase11_eval_overlap": sum(1 for sid in eval_id_set if sid in phase11_by_id),
        "phase14_eval_overlap": sum(1 for sid in eval_id_set if sid in phase14_by_id),
        "semantic_decoupled_high_sim_high_error": sum(
            1 for sid in eval_id_set if phase11_label(phase11_by_id.get(sid, {})) == "semantic_decoupled_high_sim_high_error"
        ),
        "semantic_control_low_sim_low_error": sum(
            1 for sid in eval_id_set if phase11_label(phase11_by_id.get(sid, {})) == "semantic_control_low_sim_low_error"
        ),
        "alignment_unit_sensitive": sum(
            1 for sid in eval_id_set if phase14_label(phase14_by_id.get(sid, {})) == "alignment_unit_sensitive"
        ),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "official_manifest": str(official_manifest),
    }
    (output_root / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
