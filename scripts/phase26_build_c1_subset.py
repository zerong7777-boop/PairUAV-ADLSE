#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_phase13(row):
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


def add_row(rows, split, sample_id, src, dst, label):
    rows.append(
        {
            "split": split,
            "sample_id": sample_id,
            "source_json": str(src),
            "target_json": str(dst),
            "residual_target_dominance_label": label,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase13-residual-csv", required=True)
    parser.add_argument("--train-json-root", required=True)
    parser.add_argument("--val-json-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-background-train", type=int, default=512)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_out = output_root / "train_json"
    eval_out = output_root / "eval_json"
    rows = []

    phase13 = read_csv(args.phase13_residual_csv)
    selected_eval = []
    for row in phase13:
        sample_id = sample_id_from_phase13(row)
        src = find_json(args.train_json_root, sample_id) or find_json(args.val_json_root, sample_id)
        if src is None:
            continue
        group, json_id = sample_id.split("/", 1)
        dst = eval_out / group / f"{json_id}.json"
        link_or_copy(src, dst)
        label = row.get("residual_target_dominance_label") or row.get("target_dominance_label") or "unknown"
        add_row(rows, "eval", sample_id, src, dst, label)
        selected_eval.append(sample_id)

    if not selected_eval:
        raise SystemExit("No Phase13 overlap samples found in train/val JSON roots")

    copied_train = 0
    for group_dir in sorted(Path(args.train_json_root).iterdir()):
        if not group_dir.is_dir():
            continue
        for src in sorted(group_dir.glob("*.json")):
            if copied_train >= args.max_background_train:
                break
            sample_id = f"{group_dir.name}/{src.stem}"
            group, json_id = sample_id.split("/", 1)
            dst = train_out / group / f"{json_id}.json"
            link_or_copy(src, dst)
            add_row(rows, "train", sample_id, src, dst, "background")
            copied_train += 1
        if copied_train >= args.max_background_train:
            break

    for eval_sample_id in selected_eval:
        src = find_json(args.train_json_root, eval_sample_id) or find_json(args.val_json_root, eval_sample_id)
        group, json_id = eval_sample_id.split("/", 1)
        dst = train_out / group / f"{json_id}.json"
        link_or_copy(src, dst)
        add_row(rows, "train", eval_sample_id, src, dst, "phase13_overlap")

    manifest_path = output_root / "subset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "sample_id", "source_json", "target_json", "residual_target_dominance_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    label_counts = {}
    for row in rows:
        if row["split"] != "eval":
            continue
        label = row["residual_target_dominance_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "phase13_eval_overlap": len(selected_eval),
        "eval_label_counts": label_counts,
        "output_root": str(output_root),
        "manifest": str(manifest_path),
    }
    (output_root / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
