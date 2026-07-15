import json

from scripts.phase59_gate_report import (
    BASELINE_ANGLE_MAE,
    BASELINE_DISTANCE_MAE,
    build_gate_report,
    load_eval_metrics,
)


def _write_metrics(path, angle, distance, samples=811):
    path.write_text(
        json.dumps(
            {
                "angle_mae_deg": angle,
                "distance_mae": distance,
                "samples": samples,
                "final_score_proxy": 0.0,
                "distance_mode": "endpoint_range_span",
            }
        ),
        encoding="utf-8",
    )


def test_load_eval_metrics_reads_required_fields(tmp_path):
    metrics_path = tmp_path / "val_metrics_endpoint_range_span.json"
    _write_metrics(metrics_path, angle=0.12, distance=0.04)

    metrics = load_eval_metrics(metrics_path)

    assert metrics["angle_mae_deg"] == 0.12
    assert metrics["distance_mae"] == 0.04
    assert metrics["samples"] == 811


def test_build_gate_report_marks_g1_g2_g3():
    report = build_gate_report(
        run_name="good",
        metrics={"angle_mae_deg": 0.118, "distance_mae": 0.043, "samples": 811},
        base_checkpoint="/x/rank1.pth",
        run_dir="/x/run",
        eval_dir="/x/eval",
    )

    assert report["baseline"]["angle_mae_deg"] == BASELINE_ANGLE_MAE
    assert report["baseline"]["distance_mae"] == BASELINE_DISTANCE_MAE
    assert report["delta_vs_rank1_pct"]["angle_mae_deg"] < -10.0
    assert report["gates"]["g1_direct_angle"]["pass"] is True
    assert report["gates"]["g2_strong_angle"]["pass"] is True
    assert report["gates"]["g3_distance_protection"]["pass"] is True
    assert report["decision"] == "promote_to_scale_review"


def test_build_gate_report_kills_distance_regression():
    report = build_gate_report(
        run_name="bad_distance",
        metrics={"angle_mae_deg": 0.120, "distance_mae": 0.050, "samples": 811},
        base_checkpoint="/x/rank1.pth",
        run_dir="/x/run",
        eval_dir="/x/eval",
    )

    assert report["gates"]["g1_direct_angle"]["pass"] is True
    assert report["gates"]["g3_distance_protection"]["pass"] is False
    assert report["decision"] == "hold_or_kill_distance_regression"
