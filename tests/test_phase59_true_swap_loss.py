import math

import torch

from reloc3r.loss import PairUAVOfficialMetricAwareTrueSwapLoss


def _heading_vec(deg):
    rad = math.radians(float(deg))
    return [math.cos(rad), math.sin(rad)]


def _view(heading_deg, range_value):
    vec = _heading_vec(heading_deg)
    return {
        "heading_deg": torch.tensor([float(heading_deg)], dtype=torch.float32),
        "heading_cos": torch.tensor([vec[0]], dtype=torch.float32),
        "heading_sin": torch.tensor([vec[1]], dtype=torch.float32),
        "range_value": torch.tensor([float(range_value)], dtype=torch.float32),
    }


def _prediction(heading_deg, range_value):
    return {
        "heading_vec": torch.tensor([_heading_vec(heading_deg)], dtype=torch.float32),
        "range_value": torch.tensor([[float(range_value)]], dtype=torch.float32),
    }


def test_true_swap_loss_is_near_zero_for_perfect_forward_and_swapped_inverse():
    criterion = PairUAVOfficialMetricAwareTrueSwapLoss(
        absolute_heading_weight=0.0,
        absolute_range_weight=0.0,
        swapped_supervised_weight=1.0,
        true_swap_consistency_weight=1.0,
        inverse_heading_policy="neg_deg",
        inverse_range_policy="neg",
    )
    gt1 = _view(30.0, 10.0)
    gt2 = _view(30.0, 10.0)
    pose1 = _prediction(-30.0, -10.0)
    pose2 = _prediction(30.0, 10.0)
    swapped_pose1 = _prediction(30.0, 10.0)
    swapped_pose2 = _prediction(-30.0, -10.0)

    loss, details = criterion(
        gt1,
        gt2,
        pose1,
        pose2,
        swapped_pose1=swapped_pose1,
        swapped_pose2=swapped_pose2,
    )

    assert float(loss) < 1e-5
    assert details["pairuav_true_swap_supervised_angle_abs_deg"] < 1e-5
    assert details["pairuav_true_swap_heading_consistency"] < 1e-5
    assert details["pairuav_true_swap_range_consistency"] < 1e-5


def test_true_swap_loss_penalizes_wrong_swapped_prediction():
    criterion = PairUAVOfficialMetricAwareTrueSwapLoss(
        absolute_heading_weight=0.0,
        absolute_range_weight=0.0,
        swapped_supervised_weight=1.0,
        true_swap_consistency_weight=1.0,
        inverse_heading_policy="neg_deg",
        inverse_range_policy="neg",
    )
    gt1 = _view(30.0, 10.0)
    gt2 = _view(30.0, 10.0)
    pose1 = _prediction(-30.0, -10.0)
    pose2 = _prediction(30.0, 10.0)
    swapped_pose1 = _prediction(30.0, 10.0)
    swapped_pose2 = _prediction(50.0, 10.0)

    loss, details = criterion(
        gt1,
        gt2,
        pose1,
        pose2,
        swapped_pose1=swapped_pose1,
        swapped_pose2=swapped_pose2,
    )

    assert torch.isfinite(loss)
    assert details["pairuav_true_swap_supervised_angle_abs_deg"] > 70.0
    assert details["pairuav_true_swap_heading_consistency"] > 0.1


def test_true_swap_loss_requires_swapped_predictions():
    criterion = PairUAVOfficialMetricAwareTrueSwapLoss()
    gt1 = _view(0.0, 0.0)
    gt2 = _view(0.0, 0.0)
    pose1 = _prediction(0.0, 0.0)
    pose2 = _prediction(0.0, 0.0)

    try:
        criterion(gt1, gt2, pose1, pose2)
    except ValueError as exc:
        assert "swapped_pose2" in str(exc)
    else:
        raise AssertionError("expected missing swapped predictions to raise ValueError")
