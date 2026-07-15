from pathlib import Path

from scripts.phase57_fit_angle_correction import (
    apply_distance_bin_bias,
    apply_global_bias,
    base_metrics,
    evaluate_calibrated_apply,
    evaluate_cv,
    fit_distance_bin_bias,
    fit_global_bias,
    signed_error,
    wrap_angle_deg,
    write_outputs,
)


def _row(pair_id, fold_id, pred, target, distance=100.0):
    return {
        "pair_id": pair_id,
        "group_id": pair_id.split("/")[0],
        "json_path": f"/tmp/{pair_id}.json",
        "fold_id": fold_id,
        "target_heading": str(target),
        "target_distance": str(distance),
        "rank1_heading": str(pred),
        "rank1_distance": str(distance),
        "rank1_heading_float": float(pred),
        "rank1_distance_float": float(distance),
        "target_heading_float": float(target),
        "target_distance_float": float(distance),
        "rank1_signed_error": signed_error(pred, target),
        "rank1_abs_error": abs(signed_error(pred, target)),
        "abs_rank1_distance": abs(float(distance)),
    }


def _reverse_row(pair_id, fold_id, pred, reverse_forward, target, distance=100.0):
    row = _row(pair_id, fold_id, pred, target, distance)
    row["reverse_forward_heading_float"] = float(reverse_forward)
    row["same_forward_avg_heading_float"] = (float(pred) + float(reverse_forward)) / 2.0
    row["same_forward_heading_disagreement_float"] = abs(signed_error(pred, reverse_forward))
    return row


def test_wrap_angle_deg():
    assert wrap_angle_deg(181.0) == -179.0
    assert signed_error(181.0, -179.0) == 0.0


def test_global_bias_improves_simple_biased_rows():
    rows = [_row("g/1", 0, 12.0, 10.0), _row("g/2", 1, 22.0, 20.0)]
    params = fit_global_bias(rows)
    corrected = [dict(row, corrected=apply_global_bias(row, params)) for row in rows]

    assert params["bias_deg"] == 2.0
    assert base_metrics(rows, "rank1_heading_float")["angle_mae"] == 2.0
    assert base_metrics(corrected, "corrected")["angle_mae"] == 0.0


def test_distance_bin_bias_changes_only_heading():
    rows = [_row("g/1", 0, 12.0, 10.0, 40.0), _row("g/2", 1, 30.0, 20.0, 120.0)]
    params = fit_distance_bin_bias(rows, prior_strength=0.0)
    corrected = apply_distance_bin_bias(rows[0], params)

    assert corrected == 10.0
    assert rows[0]["rank1_distance_float"] == 40.0


def test_evaluate_cv_uses_nonheldout_folds_for_fitting():
    folds = {
        0: [_row("a/1", 0, 12.0, 10.0), _row("a/2", 0, 22.0, 20.0)],
        1: [_row("b/1", 1, 32.0, 30.0), _row("b/2", 1, 42.0, 40.0)],
    }

    result = evaluate_cv(folds)
    global_row = next(row for row in result["method_metrics"] if row["method"] == "m1_global_bias")

    assert global_row["corrected_angle_mae"] == 0.0
    assert global_row["distance_mae_delta"] == 0.0


def test_evaluate_cv_adds_reverse_average_when_reverse_columns_exist():
    folds = {
        0: [_reverse_row("a/1", 0, 12.0, 10.0, 10.0)],
        1: [_reverse_row("b/1", 1, 32.0, 30.0, 30.0)],
    }

    result = evaluate_cv(folds)
    methods = {row["method"]: row for row in result["method_metrics"]}

    assert "m4_same_forward_average" in methods
    assert methods["m4_same_forward_average"]["corrected_angle_mae"] == 1.0
    assert methods["m4_same_forward_average"]["distance_mae_delta"] == 0.0


def test_write_outputs_creates_report(tmp_path: Path):
    folds = {
        0: [_row("a/1", 0, 12.0, 10.0)],
        1: [_row("b/1", 1, 32.0, 30.0)],
    }
    result = evaluate_cv(folds)
    write_outputs(tmp_path, result)

    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "method_metrics.csv").is_file()
    assert (tmp_path / "corrected_predictions" / "m0_noop.csv").is_file()


def test_evaluate_calibrated_apply_uses_external_calibration():
    calib = [_row("c/1", 0, 12.0, 10.0), _row("c/2", 0, 22.0, 20.0)]
    eval_rows = [_row("e/1", 0, 32.0, 30.0)]

    result = evaluate_calibrated_apply(calib, eval_rows)
    global_row = next(row for row in result["method_metrics"] if row["method"] == "m1_global_bias")

    assert global_row["corrected_angle_mae"] == 0.0
    assert global_row["distance_mae_delta"] == 0.0
