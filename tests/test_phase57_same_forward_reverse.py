import csv

import torch

from scripts.phase57_eval_same_forward_reverse import (
    circular_mean_deg,
    heading_deg_from_vec,
    select_reverse_prediction_tensors,
    wrap_angle_deg,
    write_same_forward_reverse_csv,
)


def test_inverse_heading_wrap_policy():
    assert wrap_angle_deg(-181.0) == 179.0
    assert wrap_angle_deg(181.0) == -179.0


def test_circular_mean_crosses_wrap_boundary():
    value = circular_mean_deg(179.0, -179.0)
    assert abs(abs(value) - 180.0) < 1e-6


def test_heading_deg_from_vec():
    heading = heading_deg_from_vec(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert heading.tolist() == [90.0, 0.0]


def test_select_reverse_prediction_tensors_true_swap_uses_swapped_pred2():
    pred1 = {
        "heading_vec": torch.tensor([[1.0, 0.0]]),
        "range_value": torch.tensor([1.0]),
    }
    pred2_swapped = {
        "heading_vec": torch.tensor([[0.0, 1.0]]),
        "range_value": torch.tensor([2.0]),
    }

    heading_vec, range_value = select_reverse_prediction_tensors(
        "true_swap_pred2_inverse",
        pred1,
        pred2_swapped=pred2_swapped,
    )

    assert torch.equal(heading_vec, pred2_swapped["heading_vec"])
    assert torch.equal(range_value, pred2_swapped["range_value"])


def test_write_same_forward_reverse_csv_uses_neg_deg_inverse(tmp_path):
    out = tmp_path / "pred.csv"
    metrics = write_same_forward_reverse_csv(
        out,
        pair_ids=["g/01_02"],
        group_ids=["g"],
        json_paths=["/tmp/g/01_02.json"],
        pred_heading=torch.tensor([12.0]),
        pred_distance=torch.tensor([101.0]),
        reverse_heading=torch.tensor([-10.0]),
        reverse_distance=torch.tensor([-99.0]),
        target_heading=torch.tensor([10.0]),
        target_distance=torch.tensor([100.0]),
    )
    row = next(csv.DictReader(out.open(encoding="utf-8")))
    assert float(row["reverse_forward_heading"]) == 10.0
    assert float(row["reverse_forward_distance"]) == 99.0
    assert float(row["same_forward_heading_disagreement"]) == 2.0
    assert metrics["same_forward_avg_angle_mae"] < metrics["rank1_angle_mae"]
