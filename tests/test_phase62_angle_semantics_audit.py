from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "phase62_angle_semantics_audit.py"
    spec = importlib.util.spec_from_file_location("phase62_angle_semantics_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_pair(root: Path, group: str, stem: str, heading: float, distance: float) -> None:
    path = root / group / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "heading_num": heading,
                "range_num": distance,
                "image_a": f"{group}/image-{stem.split('_')[0]}.jpeg",
                "image_b": f"{group}/image-{stem.split('_')[1]}.jpeg",
            }
        ),
        encoding="utf-8",
    )


def test_angle_semantics_audit_detects_reverse_and_self_pairs(tmp_path: Path) -> None:
    module = load_script()
    train_root = tmp_path / "train_json"
    val_root = tmp_path / "val_json"
    for root in (train_root, val_root):
        write_pair(root, "0001", "01_02", 30.0, 4.0)
        write_pair(root, "0001", "02_01", -30.0, -4.0)
        write_pair(root, "0001", "01_01", 0.0, 0.0)

    result = module.run_audit(train_root, val_root, tmp_path / "out")
    payload = result["payload"]
    train = payload["splits"]["train"]
    val = payload["splits"]["val"]

    assert train["reverse_pair_count"] == 1
    assert train["reverse_heading_abs_error_max"] == 0.0
    assert train["reverse_distance_abs_error_max"] == 0.0
    assert train["self_pair_count"] == 1
    assert train["num_nodes"] == 2
    assert val["reverse_pair_count"] == 1
    assert payload["transform_assessment"]["reverse_pair_negation_supported"] is True
    assert payload["transform_assessment"]["rotation_or_crop_label_transform_supported"] == "unknown"
    assert Path(result["json_path"]).exists()
    assert Path(result["report_path"]).exists()
