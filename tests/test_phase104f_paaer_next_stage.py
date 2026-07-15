import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


def _import_script(repo_root: Path, relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, repo_root / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paaer_range_fc_only_policy_trains_only_fc_range():
    from reloc3r.trainable_policy import should_train_parameter

    trainable = [
        "pose_head.fc_range.weight",
        "pose_head.fc_range.bias",
    ]
    frozen = [
        "pose_head.proj.weight",
        "pose_head.res_conv.0.res_conv1.weight",
        "pose_head.more_mlps.0.weight",
        "pose_head.fc_heading.weight",
        "pose_head.phase104e_heading_fc.weight",
        "pose_head.task_tokens",
        "pose_head.task_cross_attn.in_proj_weight",
        "pose_head.heading_token_mlp.0.weight",
        "enc_blocks.0.attn.qkv.weight",
        "dec_blocks.11.cross_attn.projq.weight",
    ]
    for name in trainable:
        assert should_train_parameter(name, "paaer_range_fc_only", dec_depth=12)
    for name in frozen:
        assert not should_train_parameter(name, "paaer_range_fc_only", dec_depth=12)


def test_paaer_range_path_only_policy_trains_range_feature_path_and_fc():
    from reloc3r.trainable_policy import should_train_parameter

    trainable = [
        "pose_head.proj.weight",
        "pose_head.proj.bias",
        "pose_head.res_conv.0.res_conv1.weight",
        "pose_head.res_conv.1.res_conv3.bias",
        "pose_head.more_mlps.0.weight",
        "pose_head.more_mlps.2.bias",
        "pose_head.fc_range.weight",
        "pose_head.fc_range.bias",
    ]
    frozen = [
        "pose_head.fc_heading.weight",
        "pose_head.phase104e_heading_fc.weight",
        "pose_head.task_tokens",
        "pose_head.task_cross_attn.in_proj_weight",
        "pose_head.heading_token_mlp.0.weight",
        "enc_blocks.0.attn.qkv.weight",
        "dec_blocks.11.cross_attn.projq.weight",
    ]
    for name in trainable:
        assert should_train_parameter(name, "paaer_range_path_only", dec_depth=12)
    for name in frozen:
        assert not should_train_parameter(name, "paaer_range_path_only", dec_depth=12)


def _toy_batch():
    gt1 = {"heading_deg": torch.tensor([0.0, 90.0]), "range_value": torch.tensor([10.0, 20.0])}
    gt2 = {"heading_deg": torch.tensor([0.0, 90.0]), "range_value": torch.tensor([10.0, 20.0])}
    pose1 = {}
    heading_vec = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    range_value = torch.tensor([12.0, 16.0], requires_grad=True)
    pose2 = {"heading_vec": heading_vec, "range_value": range_value}
    return gt1, gt2, pose1, pose2


def test_heading_only_loss_has_no_range_grad():
    from reloc3r.loss import PairUAVHeadingOnlyMetricAwareLoss

    gt1, gt2, pose1, pose2 = _toy_batch()
    loss, details = PairUAVHeadingOnlyMetricAwareLoss()(gt1, gt2, pose1, pose2)
    loss.backward()
    assert pose2["heading_vec"].grad is not None
    assert pose2["range_value"].grad is None
    assert "pairuav_heading_only_distance_abs_log" in details


def test_range_only_loss_has_no_heading_grad():
    from reloc3r.loss import PairUAVRangeOnlyMetricAwareLoss

    gt1, gt2, pose1, pose2 = _toy_batch()
    loss, details = PairUAVRangeOnlyMetricAwareLoss()(gt1, gt2, pose1, pose2)
    loss.backward()
    assert pose2["range_value"].grad is not None
    assert pose2["heading_vec"].grad is None
    assert "pairuav_range_only_angle_abs_deg_log" in details


def test_teacher_cache_builder_writes_strict_alignment_fields(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _import_script(repo_root, "scripts/phase104f_build_teacher_cache.py", "phase104f_build_teacher_cache")

    json_root = tmp_path / "json"
    (json_root / "0001").mkdir(parents=True)
    (json_root / "0001" / "02_03.json").write_text(
        json.dumps(
            {
                "group_id": "0001",
                "json_id": "02_03",
                "image_a": "/data/src.jpg",
                "image_b": "/data/tgt.jpg",
                "heading_deg": 10.0,
                "range_value": 5.0,
            }
        ),
        encoding="utf-8",
    )
    pred = tmp_path / "pred.txt"
    pred.write_text("12.5 99.0\n", encoding="utf-8")
    out = tmp_path / "teacher.csv"

    rows = module.build_cache(json_root=json_root, prediction_path=pred, output_csv=out)
    assert rows[0]["sample_id"] == "0001/02_03"
    assert rows[0]["src_image_path"] == "/data/src.jpg"
    assert rows[0]["tgt_image_path"] == "/data/tgt.jpg"
    assert rows[0]["gt_heading"] == "10.000000"
    assert rows[0]["gt_range"] == "5.000000"
    assert rows[0]["teacher_heading_deg"] == "12.500000"
    assert float(rows[0]["teacher_heading_cos"]) == pytest.approx(math.cos(math.radians(12.5)), abs=1e-5)
    assert rows[0]["teacher_heading_error"] == "2.500000"
    assert out.exists()


def test_heading_teacher_cache_loss_checks_gt_alignment_and_logs_quality(tmp_path):
    from reloc3r.loss import PairUAVHeadingTeacherCacheLoss

    csv_path = tmp_path / "teacher.csv"
    csv_path.write_text(
        "sample_id,src_image_path,tgt_image_path,json_path,gt_heading,gt_range,teacher_heading_deg,teacher_heading_cos,teacher_heading_sin,teacher_heading_error\n"
        "0001/02_03,/a.jpg,/b.jpg,/x.json,10.000000,5.000000,0.000000,1.000000,0.000000,10.000000\n",
        encoding="utf-8",
    )
    gt1 = {"sample_id": ["0001/02_03"], "heading_deg": torch.tensor([10.0]), "range_value": torch.tensor([5.0])}
    gt2 = {"sample_id": ["0001/02_03"], "heading_deg": torch.tensor([10.0]), "range_value": torch.tensor([5.0])}
    pose1 = {}
    pose2 = {
        "heading_vec": torch.tensor([[0.0, 1.0]], requires_grad=True),
        "range_value": torch.tensor([7.0], requires_grad=True),
        "phase104e_base_heading_vec": torch.tensor([[0.0, 1.0]]),
    }

    loss, details = PairUAVHeadingTeacherCacheLoss(str(csv_path), heading_distill_weight=0.05)(gt1, gt2, pose1, pose2)
    loss.backward()
    assert pose2["heading_vec"].grad is not None
    assert pose2["range_value"].grad is None
    assert details["pairuav_heading_teacher_kd_count"] == 1
    assert "pairuav_heading_teacher_error_mean" in details
    assert "pairuav_heading_teacher_better_than_base_ratio" in details
    assert "pairuav_heading_teacher_better_than_current_student_ratio" in details
    assert "pairuav_heading_teacher_kd_loss_mean" in details

    bad_gt2 = {"sample_id": ["0001/02_03"], "heading_deg": torch.tensor([11.0]), "range_value": torch.tensor([5.0])}
    with pytest.raises(ValueError, match="gt_heading"):
        PairUAVHeadingTeacherCacheLoss(str(csv_path), heading_distill_weight=0.05)(gt1, bad_gt2, pose1, pose2)


def test_safe_axis_graft_aborts_on_prediction_count_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _import_script(repo_root, "scripts/phase104f_safe_axis_graft_eval.py", "phase104f_safe_axis_graft_eval")

    json_root = tmp_path / "json"
    (json_root / "0001").mkdir(parents=True)
    for stem in ("02_03", "04_05"):
        (json_root / "0001" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "group_id": "0001",
                    "json_id": stem,
                    "image_a": f"/data/{stem}_a.jpg",
                    "image_b": f"/data/{stem}_b.jpg",
                    "heading_deg": 10.0,
                    "range_value": 5.0,
                }
            ),
            encoding="utf-8",
        )
    good_pred = tmp_path / "good.txt"
    good_pred.write_text("1 2\n3 4\n", encoding="utf-8")
    short_pred = tmp_path / "short.txt"
    short_pred.write_text("1 2\n", encoding="utf-8")

    manifest = module.load_manifest(json_root)
    module.validate_prediction_file(good_pred, manifest)
    with pytest.raises(ValueError, match="line count"):
        module.validate_prediction_file(short_pred, manifest)
