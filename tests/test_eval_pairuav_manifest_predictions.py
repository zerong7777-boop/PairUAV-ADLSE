import csv
from pathlib import Path

import torch

from scripts.eval_pairuav_manifest_predictions import (
    angle_abs_error_deg,
    write_prediction_csv,
)


def test_angle_abs_error_wraps():
    assert angle_abs_error_deg(359.0, 1.0) == 2.0
    assert angle_abs_error_deg(1.0, 359.0) == 2.0


def test_write_prediction_csv_preserves_pair_id(tmp_path):
    out = tmp_path / "pred.csv"
    write_prediction_csv(
        out,
        pair_ids=["a/1"],
        group_ids=["a"],
        json_paths=["/x/a/1.json"],
        pred_heading=torch.tensor([12.0]),
        pred_distance=torch.tensor([101.0]),
        target_heading=torch.tensor([10.0]),
        target_distance=torch.tensor([100.0]),
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["pair_id"] == "a/1"
    assert float(rows[0]["rank1_angle_abs_error"]) == 2.0
    assert float(rows[0]["rank1_distance_abs_error"]) == 1.0
