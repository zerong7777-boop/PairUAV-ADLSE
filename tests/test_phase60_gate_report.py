import json

from scripts.phase60_gate_report import build_gate_report, load_eval_metrics


def test_phase60_gate_report_promotes_strong_angle(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"angle_mae_deg": 0.118, "distance_mae": 0.043, "samples": 811}), encoding="utf-8")
    report = build_gate_report(
        run_name="good",
        metrics=load_eval_metrics(metrics_path),
        base_checkpoint="/x/rank1.pth",
        run_dir="/x/run",
        eval_dir="/x/eval",
    )
    assert report["gates"]["g2_strong_angle"]["pass"] is True
    assert report["gates"]["g3_distance_protection"]["pass"] is True
    assert report["decision"] == "promote_to_scale_review"


def test_phase60_gate_report_holds_tiny_non_regression():
    report = build_gate_report(
        run_name="tiny",
        metrics={"angle_mae_deg": 0.131900, "distance_mae": 0.0429, "samples": 811},
        base_checkpoint="/x/rank1.pth",
        run_dir="/x/run",
        eval_dir="/x/eval",
    )
    assert report["gates"]["g0_angle_non_regression"]["pass"] is True
    assert report["gates"]["g1_direct_angle"]["pass"] is False
    assert report["gates"]["g3_distance_protection"]["pass"] is True
    assert report["decision"] == "hold_for_repeat_checkpoint_policy"


def test_phase60_gate_report_kills_regression():
    report = build_gate_report(
        run_name="bad",
        metrics={"angle_mae_deg": 0.1325, "distance_mae": 0.0425, "samples": 811},
        base_checkpoint="/x/rank1.pth",
        run_dir="/x/run",
        eval_dir="/x/eval",
    )
    assert report["gates"]["g0_angle_non_regression"]["pass"] is False
    assert report["decision"] == "kill_or_redesign"
