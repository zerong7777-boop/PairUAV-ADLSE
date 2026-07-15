import math

import torch

from reloc3r.loss import PairUAVOfficialMetricAwareBidirectionalLoss


def _heading_vec(deg):
    rad = math.radians(float(deg))
    return [math.cos(rad), math.sin(rad)]


def _view(heading_deg, range_value):
    vec = _heading_vec(heading_deg)
    values = torch.tensor([float(heading_deg)], dtype=torch.float32)
    ranges = torch.tensor([float(range_value)], dtype=torch.float32)
    return {
        "heading_deg": values,
        "heading_cos": torch.tensor([vec[0]], dtype=torch.float32),
        "heading_sin": torch.tensor([vec[1]], dtype=torch.float32),
        "range_value": ranges,
    }


def _prediction(heading_deg, range_value):
    return {
        "heading_vec": torch.tensor([_heading_vec(heading_deg)], dtype=torch.float32),
        "range_value": torch.tensor([[float(range_value)]], dtype=torch.float32),
    }


def _criterion(**kwargs):
    return PairUAVOfficialMetricAwareBidirectionalLoss(
        absolute_heading_weight=0.0,
        absolute_range_weight=0.0,
        inverse_weight=1.0,
        self_consistency_weight=1.0,
        **kwargs,
    )


def test_perfect_main_inverse_neg_deg_policy_is_near_zero():
    criterion = _criterion(inverse_heading_policy="neg_deg", inverse_range_policy="neg")
    gt1 = _view(160.0, -20.0)
    gt2 = _view(160.0, -20.0)
    pose2 = _prediction(160.0, -20.0)
    pose1 = _prediction(-160.0, 20.0)

    loss, details = criterion(gt1, gt2, pose1, pose2)

    assert float(loss) < 1e-5
    assert details["pairuav_main_angle_abs_deg"] < 1e-5
    assert details["pairuav_inverse_angle_abs_deg"] < 1e-5
    assert details["pairuav_self_heading_consistency"] < 1e-5
    assert details["pairuav_self_range_consistency"] < 1e-5


def test_perfect_main_inverse_neg_vec_policy_is_near_zero():
    criterion = _criterion(inverse_heading_policy="neg_vec", inverse_range_policy="neg")
    gt1 = _view(160.0, -20.0)
    gt2 = _view(160.0, -20.0)
    pose2 = _prediction(160.0, -20.0)
    pose1 = _prediction(-20.0, 20.0)

    loss, details = criterion(gt1, gt2, pose1, pose2)

    assert float(loss) < 1e-5
    assert details["pairuav_inverse_angle_abs_deg"] < 1e-5
    assert details["pairuav_self_heading_consistency"] < 1e-5


def test_wrapped_angle_error_handles_181_degree_equivalence():
    criterion = _criterion(inverse_heading_policy="same", inverse_range_policy="same")
    gt1 = _view(-179.0, 10.0)
    gt2 = _view(-179.0, 10.0)
    pose2 = _prediction(181.0, 10.0)
    pose1 = _prediction(181.0, 10.0)

    loss, details = criterion(gt1, gt2, pose1, pose2)

    assert torch.isfinite(loss)
    assert details["pairuav_main_angle_abs_deg"] < 1e-4


def test_loss_is_finite_near_zero_targets():
    criterion = _criterion(inverse_heading_policy="neg_deg", inverse_range_policy="neg")
    gt1 = _view(0.0, 0.0)
    gt2 = _view(0.0, 0.0)
    pose2 = _prediction(0.5, 0.25)
    pose1 = _prediction(-0.5, -0.25)

    loss, details = criterion(gt1, gt2, pose1, pose2)

    assert torch.isfinite(loss)
    assert math.isfinite(details["pairuav_total"])
