import csv
import json
import subprocess
import sys
from pathlib import Path


def test_policy_packet_builder_outputs_capped_weights(tmp_path):
    json_root = tmp_path / "json"
    json_root.mkdir()
    sample = {
        "group_id": "0001",
        "json_id": "00_01",
        "image_a": "a.jpg",
        "image_b": "b.jpg",
        "heading_deg": 20.0,
        "range_value": 100.0,
    }
    (json_root / "00_01.json").write_text(json.dumps(sample), encoding="utf-8")

    source_csv = tmp_path / "source.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "roma_mean",
                "roma_max",
                "mast3r_mean",
                "mast3r_max",
                "vggt_mean",
                "vggt_max",
                "roma_angle_error",
                "roma_distance_error",
                "vggt_angle_error",
                "vggt_distance_error",
                "mast3r_vggt_angle_error",
                "mast3r_vggt_distance_error",
                "roma_mast3r_vggt_angle_error",
                "roma_mast3r_vggt_distance_error",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "pair_id": "0001/00_01",
            "roma_mean": "0.10",
            "roma_max": "0.90",
            "mast3r_mean": "0.20",
            "mast3r_max": "0.80",
            "vggt_mean": "0.30",
            "vggt_max": "0.70",
            "roma_angle_error": "4.0",
            "roma_distance_error": "8.0",
            "vggt_angle_error": "9.0",
            "vggt_distance_error": "5.0",
            "mast3r_vggt_angle_error": "7.0",
            "mast3r_vggt_distance_error": "3.0",
            "roma_mast3r_vggt_angle_error": "2.0",
            "roma_mast3r_vggt_distance_error": "6.0",
        })

    output = tmp_path / "packet.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_pairuav_source_policy_packet.py",
            "--json-root",
            str(json_root),
            "--source-table-csv",
            str(source_csv),
            "--output-jsonl",
            str(output),
            "--policy-version",
            "unit-test",
            "--require-source-row",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["pair_id"] == "0001/00_01"
    assert 0.5 <= row["sample_weight"] <= 2.0
    assert 0.5 <= row["angle_weight"] <= 2.5
    assert 0.5 <= row["range_weight"] <= 2.5
    assert "roma_full_angle_helpful" in row["bucket_tags"]
    assert "mast3r_vggt_distance_helpful" in row["bucket_tags"]
    assert row["policy_version"] == "unit-test"


def test_policy_packet_audit_rejects_duplicate_pair_id(tmp_path):
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        "\n".join([
            json.dumps({"pair_id": "0001/00_01", "split": "train", "sample_weight": 1.0, "angle_weight": 1.0, "range_weight": 1.0, "bucket_tags": ["easy_anchor"]}),
            json.dumps({"pair_id": "0001/00_01", "split": "train", "sample_weight": 1.0, "angle_weight": 1.0, "range_weight": 1.0, "bucket_tags": ["easy_anchor"]}),
        ]),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_pairuav_policy_packet.py",
            "--packet-jsonl",
            str(packet),
            "--output-json",
            str(tmp_path / "audit.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "duplicate pair_id" in result.stderr
