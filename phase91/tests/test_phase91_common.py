import math

from phase91.common import circular_abs_error_deg, circular_diff_deg, summarize_numeric
from phase91.eval_phase91_metrics import (
    compute_phase91_metrics,
    displacement_epe,
    phase91_metric_contract,
)
from phase91.matched_baseline_inventory import build_baseline_inventory, compatibility_label
from phase91.matched_baseline_replay import build_replay_rows


def test_circular_diff_wrap_boundary():
    assert circular_diff_deg(179.0, -179.0) == -2.0
    assert circular_diff_deg(-179.0, 179.0) == 2.0
    assert circular_abs_error_deg(181.0, -179.0) == 0.0


def test_summarize_numeric():
    stats = summarize_numeric([1.0, 2.0, 3.0])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0


def test_phase91_metric_contract_scores_wrap_and_displacement():
    payload = compute_phase91_metrics(
        heading_prediction_deg=[181.0, -179.0],
        heading_target_deg=[-179.0, 179.0],
        range_prediction=[10.0, 12.0],
        range_target=[10.0, 10.0],
        range_min=0.0,
        range_max=20.0,
    )

    assert payload["samples"] == 2
    assert payload["angle_mae_deg"] == 1.0
    assert payload["distance_mae"] == 1.0
    assert payload["angle_rel_error"] == 1.0 / 180.0
    assert payload["distance_rel_error"] == 1.0 / 20.0
    assert payload["final_score_proxy"] == ((1.0 / 180.0) + (1.0 / 20.0)) / 2.0
    assert payload["heading_induced_displacement_error"] > 0.0
    assert payload["heading_induced_displacement_error"] < payload["displacement_epe"]
    assert math.isclose(payload["range_induced_displacement_error"], 1.0)
    assert payload["displacement_epe"] > payload["heading_induced_displacement_error"]


def test_displacement_epe_uses_polar_recomposition():
    assert displacement_epe([0.0], [10.0], [90.0], [10.0]) == 14.142135623730951


def test_phase91_metric_contract_names_required_outputs():
    contract = phase91_metric_contract()
    for key in (
        "angle_mae_deg",
        "distance_mae",
        "final_score_proxy",
        "displacement_epe",
        "heading_induced_displacement_error",
        "range_induced_displacement_error",
    ):
        assert key in contract["metrics"]


def test_matched_baseline_inventory_marks_required_rows_missing_or_not_comparable():
    rows = build_baseline_inventory(
        imported_rows=[
            {
                "run_name": "phase88_lab_Wbounded_H8_full_5000_lr5e-6_bs4_val811",
                "metric_file": "val_metrics_range_span.json",
                "metric_path": "/tmp/phase88/val_metrics_range_span.json",
            }
        ]
    )
    by_id = {row["baseline_id"]: row for row in rows}

    assert set(by_id) == {"B0", "B1", "B2", "B3", "B4"}
    assert by_id["B0"]["compatibility"] == "missing"
    assert by_id["B4"]["compatibility"] == "imported_not_comparable"


def test_compatibility_label_requires_exact_phase91_match():
    assert compatibility_label("phase91_h8_matched_smoke_train512_val256") == "rerun_matched"
    assert compatibility_label("phase88_lab_Wbounded_H8_full_5000_lr5e-6_bs4_val811") == "imported_not_comparable"


def test_matched_baseline_replay_rows_cover_b0_to_b4(tmp_path):
    rows = build_replay_rows(
        run_root=tmp_path,
        max_train_steps=2500,
        eval_max_samples=811,
        batch_size=4,
        eval_batch_size=8,
        lr=5e-6,
        checkpoint="/ckpt/backbone.pth",
    )

    assert [row["baseline_id"] for row in rows] == ["B0", "B1", "B2", "B3", "B4"]
    assert rows[0]["output_mode"] == "pairuav_heading_range"
    assert rows[3]["output_mode"] == "pairuav_early_split_heading_range"
    assert "MAX_TRAIN_STEPS=2500" in rows[0]["train_env"]
    assert rows[0]["eval_max_samples"] == 811
