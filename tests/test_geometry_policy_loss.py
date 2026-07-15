import csv

import pytest
import torch

from reloc3r.geometry_policy_loss import GeometryPolicyTable, PairUAVGeometryPolicyLoss


def _write_policy_csv(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "angle_policy_weight",
                "teacher_aux_allowed",
                "teacher_heading_split_fusion",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "g1/0_1",
                "angle_policy_weight": "2.0",
                "teacher_aux_allowed": "true",
                "teacher_heading_split_fusion": "8.0",
            }
        )
        writer.writerow(
            {
                "pair_id": "g2/0_2",
                "angle_policy_weight": "0.5",
                "teacher_aux_allowed": "false",
                "teacher_heading_split_fusion": "80.0",
            }
        )


def _batch(pair_id="g1/0_1", heading_deg=359.0):
    group_id, json_id = pair_id.split("/", 1)
    heading = torch.tensor([heading_deg], dtype=torch.float32)
    return {
        "group_id": [group_id],
        "json_id": [json_id],
        "sample_id": [pair_id],
        "heading_deg": heading,
        "heading_cos": torch.cos(torch.deg2rad(heading)),
        "heading_sin": torch.sin(torch.deg2rad(heading)),
        "range_value": torch.tensor([10.0], dtype=torch.float32),
    }


def _pred_from_deg(deg):
    heading = torch.tensor([deg], dtype=torch.float32, requires_grad=True)
    return {"pred_heading_deg": heading}


def test_geometry_policy_table_loads_csv(tmp_path):
    csv_path = tmp_path / "policy.csv"
    _write_policy_csv(csv_path)

    table = GeometryPolicyTable.from_csv(csv_path)
    entry = table.get("g1/0_1")

    assert entry.pair_id == "g1/0_1"
    assert entry.angle_policy_weight == 2.0
    assert entry.teacher_aux_allowed is True
    assert entry.teacher_heading_split_fusion == 8.0


def test_loss_returns_finite_scalar_and_backpropagates(tmp_path):
    csv_path = tmp_path / "policy.csv"
    _write_policy_csv(csv_path)
    loss_fn = PairUAVGeometryPolicyLoss(GeometryPolicyTable.from_csv(csv_path), teacher_aux_weight=0.25)
    pred = _pred_from_deg(1.0)

    loss, details = loss_fn.compute_loss({}, _batch("g1/0_1", 359.0), {}, pred)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert pred["pred_heading_deg"].grad is not None
    assert torch.isfinite(pred["pred_heading_deg"].grad).all()
    assert 1.5 < details["pairuav_geometry_policy_angle_abs_deg"] < 2.5


def test_hard_and_helpful_policy_weights_scale_angle_loss(tmp_path):
    csv_path = tmp_path / "policy.csv"
    _write_policy_csv(csv_path)
    loss_fn = PairUAVGeometryPolicyLoss(GeometryPolicyTable.from_csv(csv_path), teacher_aux_weight=0.0)

    hard_loss, hard_details = loss_fn.compute_loss({}, _batch("g1/0_1", 10.0), {}, _pred_from_deg(20.0))
    helpful_loss, helpful_details = loss_fn.compute_loss({}, _batch("g2/0_2", 10.0), {}, _pred_from_deg(20.0))

    assert hard_details["pairuav_geometry_policy_angle_weight_mean"] == 2.0
    assert helpful_details["pairuav_geometry_policy_angle_weight_mean"] == 0.5
    assert torch.isclose(hard_loss, helpful_loss * 4.0, rtol=1e-5, atol=1e-5)


def test_teacher_auxiliary_is_disabled_by_policy(tmp_path):
    csv_path = tmp_path / "policy.csv"
    _write_policy_csv(csv_path)
    loss_fn = PairUAVGeometryPolicyLoss(GeometryPolicyTable.from_csv(csv_path), teacher_aux_weight=1.0)

    pred = _pred_from_deg(10.0)
    loss, details = loss_fn.compute_loss({}, _batch("g2/0_2", 10.0), {}, pred)

    assert torch.isclose(loss, torch.zeros_like(loss))
    assert details["pairuav_geometry_policy_teacher_aux_count"] == 0


def test_missing_pair_id_raises_helpful_training_error(tmp_path):
    csv_path = tmp_path / "policy.csv"
    _write_policy_csv(csv_path)
    loss_fn = PairUAVGeometryPolicyLoss(GeometryPolicyTable.from_csv(csv_path))
    loss_fn.train()

    with pytest.raises(KeyError, match="Missing geometry policy entry.*missing/0_9"):
        loss_fn.compute_loss({}, _batch("missing/0_9", 0.0), {}, _pred_from_deg(0.0))
