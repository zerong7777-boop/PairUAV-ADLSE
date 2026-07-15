import csv
import json
import subprocess
import sys
from pathlib import Path


def test_surface_builder_joins_manifest_source_stats_and_anchor_errors(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "pair_id": "0001/00_01",
        "group_id": "0001",
        "json_path": str(tmp_path / "00_01.json"),
        "heading_deg": 10.0,
        "range_value": 100.0,
        "split": "train",
    }) + "\n", encoding="utf-8")

    source_stats = tmp_path / "source_stats.csv"
    with source_stats.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pair_id",
            "roma_mean",
            "roma_certainty_mean",
            "vggt_mean",
            "vggt_conf_mean",
            "mast3r_mean",
            "mast3r_conf_mean",
        ])
        writer.writeheader()
        writer.writerow({
            "pair_id": "0001/00_01",
            "roma_mean": "0.1",
            "roma_certainty_mean": "0.8",
            "vggt_mean": "0.2",
            "vggt_conf_mean": "0.9",
            "mast3r_mean": "0.3",
            "mast3r_conf_mean": "0.7",
        })

    anchor = tmp_path / "anchor.csv"
    with anchor.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "pred_heading_deg", "pred_range_value"])
        writer.writeheader()
        writer.writerow({"pair_id": "0001/00_01", "pred_heading_deg": "20.0", "pred_range_value": "110.0"})

    out = tmp_path / "surface.csv"
    subprocess.run([
        sys.executable,
        "scripts/build_pairuav_source_policy_surface.py",
        "--manifest-jsonl", str(manifest),
        "--source-stats-csv", str(source_stats),
        "--anchor-prediction-csv", str(anchor),
        "--output-csv", str(out),
    ], check=True)

    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "0001/00_01"
    assert float(rows[0]["anchor_angle_error"]) == 10.0
    assert float(rows[0]["anchor_range_error"]) == 10.0
    assert rows[0]["angle_policy_hard"] in {"0", "1"}
