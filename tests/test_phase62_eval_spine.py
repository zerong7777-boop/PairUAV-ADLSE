from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "phase62_build_eval_spine.py"
    spec = importlib.util.spec_from_file_location("phase62_build_eval_spine", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_eval_spine_outputs_metrics_and_hard_manifest(tmp_path: Path) -> None:
    module = load_script()
    val_root = tmp_path / "val_json"
    (val_root / "0001").mkdir(parents=True)
    (val_root / "0001" / "01_02.json").write_text(
        json.dumps({"heading_num": 10.0, "range_num": 2.0}),
        encoding="utf-8",
    )
    (val_root / "0001" / "02_03.json").write_text(
        json.dumps({"heading_num": -20.0, "range_num": -3.0}),
        encoding="utf-8",
    )

    prediction_csv = tmp_path / "pred.csv"
    with prediction_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "split",
                "group_id",
                "json_path",
                "target_heading",
                "target_distance",
                "rank1_heading",
                "rank1_distance",
                "rank1_angle_abs_error",
                "rank1_distance_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "0001/01_02",
                "split": "val",
                "group_id": "0001",
                "json_path": "",
                "target_heading": 10.0,
                "target_distance": 2.0,
                "rank1_heading": 11.0,
                "rank1_distance": 2.5,
                "rank1_angle_abs_error": "",
                "rank1_distance_abs_error": "",
            }
        )
        writer.writerow(
            {
                "pair_id": "0001/02_03",
                "split": "val",
                "group_id": "0001",
                "json_path": "",
                "target_heading": -20.0,
                "target_distance": -3.0,
                "rank1_heading": -18.0,
                "rank1_distance": -1.0,
                "rank1_angle_abs_error": "",
                "rank1_distance_abs_error": "",
            }
        )

    result = module.build_eval_spine(
        prediction_csv=prediction_csv,
        val_json_root=val_root,
        output_dir=tmp_path / "out",
        hard_quantile=0.5,
    )

    metrics = result["metrics"]
    assert metrics["num_rows"] == 2
    assert metrics["angle_mae"] == 1.5
    assert metrics["distance_mae"] == 1.25
    assert metrics["hard_rows"] == 1

    hard_lines = Path(result["hard_manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(hard_lines) == 1
    hard = json.loads(hard_lines[0])
    assert hard["pair_id"] == "0001/02_03"
    assert hard["json_path"].endswith("0001/02_03.json")
