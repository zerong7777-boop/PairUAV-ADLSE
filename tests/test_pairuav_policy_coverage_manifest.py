import json
import subprocess
from pathlib import Path


def _write_sample(root, group, stem, heading, distance):
    d = root / group
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "group_id": group,
        "json_id": stem,
        "image_a": f"{group}/a.jpg",
        "image_b": f"{group}/b.jpg",
        "heading_deg": heading,
        "range_value": distance,
    }
    (d / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_builder_is_group_balanced_and_train_only(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    for group in ("0001", "0002", "0003", "0004"):
        for idx in range(4):
            _write_sample(train, group, f"{idx:02d}", heading=idx * 30.0, distance=10.0 + idx)
    _write_sample(val, "0001", "99", heading=0.0, distance=10.0)

    out = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    subprocess.run(
        [
            "/usr/bin/python3",
            "scripts/build_pairuav_policy_coverage_manifest.py",
            "--train-json-root",
            str(train),
            "--val-json-root",
            str(val),
            "--output-jsonl",
            str(out),
            "--audit-json",
            str(audit),
            "--target-rows",
            "8",
            "--seed",
            "7",
        ],
        check=True,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert len({row["pair_id"] for row in rows}) == 8
    assert all(row["split"] == "train" for row in rows)
    summary = json.loads(audit.read_text(encoding="utf-8"))
    assert summary["rows"] == 8
    assert summary["val_overlap"] == 0
    assert summary["group_count"] >= 2
