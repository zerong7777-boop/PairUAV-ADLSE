import math

import torch

from reloc3r.loss import PairUAVGeometryPolicyLoss, PairUAVOfficialMetricAwareLoss


def test_geometry_policy_loss_composes_with_pairuav_official_loss(tmp_path):
    policy_csv = tmp_path / "policy.csv"
    policy_csv.write_text(
        "\n".join(
            [
                "pair_id,angle_weight,teacher_aux_allowed,teacher_heading_split_fusion",
                "g1/0001,2.0,0,",
                "g1/0002,1.0,0,",
            ]
        ),
        encoding="utf-8",
    )
    view1 = {}
    view2 = {
        "pair_id": ["g1/0001", "g1/0002"],
        "heading_deg": torch.tensor([0.0, 90.0]),
        "range_value": torch.tensor([10.0, 20.0]),
    }
    pose1 = {}
    pose2 = {
        "heading_vec": torch.tensor(
            [
                [math.cos(math.radians(10.0)), math.sin(math.radians(10.0))],
                [math.cos(math.radians(100.0)), math.sin(math.radians(100.0))],
            ],
            dtype=torch.float32,
        ),
        "range_value": torch.tensor([11.0, 18.0]),
    }

    criterion = PairUAVOfficialMetricAwareLoss() + 0.5 * PairUAVGeometryPolicyLoss(
        policy_csv,
        teacher_aux_weight=0.0,
    )
    loss, details = criterion(view1, view2, pose1, pose2)

    assert torch.isfinite(loss)
    assert "pairuav_official_like" in details
    assert "pairuav_geometry_policy_loss" in details
    assert details["pairuav_geometry_policy_angle_weight_mean"] == 1.5


def test_geometry_policy_loss_is_available_to_eval_train_criterion(tmp_path):
    policy_csv = tmp_path / "policy.csv"
    policy_csv.write_text(
        "pair_id,angle_weight\n"
        "g1/0001,1.0\n",
        encoding="utf-8",
    )
    namespace = {}
    exec("from reloc3r.loss import *", namespace)

    criterion = eval(
        f"PairUAVOfficialMetricAwareLoss() + 0.25*PairUAVGeometryPolicyLoss(r'{policy_csv}')",
        namespace,
    )

    assert "PairUAVGeometryPolicyLoss" in repr(criterion)
