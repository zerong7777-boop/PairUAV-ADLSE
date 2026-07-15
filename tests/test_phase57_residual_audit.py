from scripts.phase57_residual_audit import (
    compute_base_metrics,
    compute_signal_metrics,
    enrich_row,
    rank_auc,
    wrapped_signed_angle_error_deg,
)


def test_wrapped_signed_angle_error_handles_equivalence():
    assert wrapped_signed_angle_error_deg(181.0, -179.0) == 0.0
    assert wrapped_signed_angle_error_deg(-179.0, 181.0) == 0.0
    assert wrapped_signed_angle_error_deg(359.0, 1.0) == -2.0


def test_compute_base_metrics_counts_tails():
    rows = [
        enrich_row({"rank1_heading": "10", "target_heading": "9", "rank1_distance_abs_error": "0.2"}),
        enrich_row({"rank1_heading": "20", "target_heading": "20.25", "rank1_distance_abs_error": "0.4"}),
        enrich_row({"rank1_heading": "30", "target_heading": "32.5", "rank1_distance_abs_error": "0.6"}),
    ]
    metrics = compute_base_metrics(rows)

    assert metrics["rows"] == 3
    assert metrics["angle_ge_0p5"] == 2
    assert metrics["angle_ge_1p0"] == 2
    assert metrics["angle_ge_2p0"] == 1
    assert round(metrics["distance_mae"], 6) == 0.4


def test_rank_auc_perfect_and_tied():
    assert rank_auc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == 1.0
    assert rank_auc([1.0, 1.0], [False, True]) == 0.5


def test_compute_signal_metrics_finds_synthetic_tail_predictor():
    rows = []
    for idx in range(20):
        err = 2.0 if idx >= 15 else 0.1
        rows.append(
            enrich_row(
                {
                    "rank1_heading": str(err),
                    "target_heading": "0",
                    "rank1_distance": str(idx),
                    "synthetic_tail_score": str(idx),
                }
            )
        )

    metrics = compute_signal_metrics(rows, ["synthetic_tail_score"])[0]

    assert metrics["valid_rows"] == 20
    assert metrics["tail_ge_0p5_positives"] == 5
    assert metrics["tail_ge_0p5_auc"] == 1.0
    assert metrics["tail_ge_0p5_top_decile_lift"] > 1.0
