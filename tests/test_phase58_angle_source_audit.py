import csv

from scripts.phase58_angle_source_audit import (
    SourceSpec,
    angle_abs_error_deg,
    evaluate_source,
    parse_source_arg,
    read_rank1_csv,
    read_source_csv,
    render_report,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_angle_abs_error_wraps_at_180():
    assert angle_abs_error_deg(-179.0, 179.0) == 2.0
    assert angle_abs_error_deg(179.0, -179.0) == 2.0


def test_evaluate_source_reports_overlap_win_rate_and_oracle_bound(tmp_path):
    rank1_csv = tmp_path / "rank1.csv"
    source_csv = tmp_path / "source.csv"
    _write_csv(
        rank1_csv,
        ["pair_id", "target_heading", "target_distance", "rank1_heading", "rank1_distance"],
        [
            {
                "pair_id": "p1",
                "target_heading": "179",
                "target_distance": "10",
                "rank1_heading": "-179",
                "rank1_distance": "10.5",
            },
            {
                "pair_id": "p2",
                "target_heading": "10",
                "target_distance": "20",
                "rank1_heading": "11",
                "rank1_distance": "19",
            },
            {
                "pair_id": "p3",
                "target_heading": "-170",
                "target_distance": "30",
                "rank1_heading": "-160",
                "rank1_distance": "31",
            },
        ],
    )
    _write_csv(
        source_csv,
        ["pair_id", "split", "target_heading_deg", "pred_heading_deg"],
        [
            {"pair_id": "p1", "split": "val", "target_heading_deg": "179", "pred_heading_deg": "178"},
            {"pair_id": "p2", "split": "train", "target_heading_deg": "10", "pred_heading_deg": "12"},
            {"pair_id": "p2", "split": "val", "target_heading_deg": "10", "pred_heading_deg": "40"},
            {"pair_id": "bad", "split": "val", "target_heading_deg": "nan", "pred_heading_deg": "nan"},
        ],
    )

    rank1 = read_rank1_csv(rank1_csv)
    source = read_source_csv(
        SourceSpec(
            name="toy",
            path=source_csv,
            split="val",
            heading_col="pred_heading_deg",
            target_col="target_heading_deg",
        )
    )
    metrics, pair_rows = evaluate_source("toy", rank1, source)

    assert metrics["overlap_rows"] == 2
    assert metrics["rank1_angle_mae"] == 1.5
    assert metrics["source_angle_mae"] == 15.5
    assert metrics["source_better_count"] == 1
    assert metrics["source_win_rate"] == 0.5
    assert metrics["oracle_min_angle_mae"] == 1.0
    assert metrics["oracle_gain_pct"] == 33.333333
    assert metrics["source_angle_ge_2p0"] == 1
    assert [row["pair_id"] for row in pair_rows] == ["p1", "p2"]


def test_parse_source_arg_uses_key_value_format(tmp_path):
    spec = parse_source_arg(
        f"name=split_v2,path={tmp_path / 'pred.csv'},split=val,heading=pred_heading_deg,target=target_heading_deg"
    )
    assert spec.name == "split_v2"
    assert spec.path == tmp_path / "pred.csv"
    assert spec.split == "val"
    assert spec.heading_col == "pred_heading_deg"
    assert spec.target_col == "target_heading_deg"


def test_report_does_not_call_small_direct_gain_material():
    report = render_report(
        [
            {
                "source": "tiny_gain",
                "overlap_rows": 811,
                "rank1_angle_mae": 0.131944,
                "source_angle_mae": 0.129717,
                "source_delta_pct": -1.687746,
                "source_win_rate": 0.549938,
                "oracle_min_angle_mae": 0.117889,
                "oracle_gain_pct": 10.652308,
                "source_angle_ge_1p0": 4,
                "oracle_angle_ge_1p0": 3,
            }
        ],
        source_meta={
            "tiny_gain": {
                "path": "/tmp/tiny.csv",
                "rows_total": 811,
                "rows_loaded": 811,
                "rows_skipped": 0,
                "rows_split_filtered": 0,
                "duplicate_rows": 0,
                "heading_col": "pred",
                "target_col": "target",
            }
        },
        rank1_meta={"path": "/tmp/rank1.csv", "rows_loaded": 811},
    )
    assert "No audited source has >=5% direct angle MAE gain" in report
    assert "selector/policy route may be worth testing" in report
