import csv

from scripts.phase59_route_c_observability_audit import evaluate_cv, read_prediction_csv


def _write_rows(path, rows):
    fieldnames = [
        "pair_id",
        "group_id",
        "target_heading",
        "target_distance",
        "rank1_heading",
        "rank1_distance",
        "same_forward_avg_heading",
        "same_forward_heading_disagreement",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_route_c_selector_recovers_deployable_gain(tmp_path):
    rows = []
    for idx in range(80):
        target = 0.0
        if idx % 2 == 0:
            rank1 = 2.0
            alt = 0.1
            disagreement = 10.0
        else:
            rank1 = 0.2
            alt = 1.5
            disagreement = 0.1
        rows.append(
            {
                "pair_id": f"{idx:04d}/00_01",
                "group_id": f"{idx:04d}",
                "target_heading": target,
                "target_distance": 0.0,
                "rank1_heading": rank1,
                "rank1_distance": 1.0,
                "same_forward_avg_heading": alt,
                "same_forward_heading_disagreement": disagreement,
            }
        )
    csv_path = tmp_path / "predictions.csv"
    _write_rows(csv_path, rows)

    loaded = read_prediction_csv(csv_path)
    result = evaluate_cv(loaded, folds=5)
    selector = [row for row in result["method_metrics"] if row["method"] == "m3_cv_feature_selector"][0]

    assert selector["angle_mae_rel_improvement"] > 0.15
    assert selector["oracle_conversion"] > 0.20
    assert result["decision"] == "promote_selector_probe"


def test_route_c_reports_observability_gap_when_selector_is_weak(tmp_path):
    rows = []
    for idx in range(80):
        target = 0.0
        rank1 = 2.0 if idx % 2 == 0 else 0.2
        alt = 0.1 if idx % 2 == 0 else 1.5
        rows.append(
            {
                "pair_id": f"{idx:04d}/00_01",
                "group_id": f"{idx:04d}",
                "target_heading": target,
                "target_distance": 0.0,
                "rank1_heading": rank1,
                "rank1_distance": 1.0,
                "same_forward_avg_heading": alt,
                "same_forward_heading_disagreement": 1.0,
            }
        )
    csv_path = tmp_path / "predictions.csv"
    _write_rows(csv_path, rows)

    loaded = read_prediction_csv(csv_path)
    result = evaluate_cv(loaded, folds=5)
    selector = [row for row in result["method_metrics"] if row["method"] == "m3_cv_feature_selector"][0]
    oracle = [row for row in result["method_metrics"] if row["method"] == "m2_oracle_alt_if_better"][0]

    assert oracle["angle_mae_rel_improvement"] > 0.40
    assert selector["angle_mae_rel_improvement"] < oracle["angle_mae_rel_improvement"]
