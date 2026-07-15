import csv
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from scripts.phase59_route_b_matcher_residual_head import read_prediction_csv, run


def _write_match(path: Path, residual_signal: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keypoints0=np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32),
        keypoints1=np.asarray([[residual_signal, 0.0], [10.0 + residual_signal, 0.0], [residual_signal, 10.0]], dtype=np.float32),
        matches=np.asarray([0, 1, 2], dtype=np.int64),
        match_confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
    )


def _write_predictions(path: Path, rows: list[dict]):
    fields = ["pair_id", "group_id", "target_heading", "target_distance", "rank1_heading", "rank1_distance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_read_prediction_csv_computes_residual(tmp_path):
    csv_path = tmp_path / "pred.csv"
    _write_predictions(
        csv_path,
        [
            {
                "pair_id": "0001/01_02",
                "group_id": "0001",
                "target_heading": 1.0,
                "target_distance": 0.0,
                "rank1_heading": 0.25,
                "rank1_distance": 0.0,
            }
        ],
    )
    rows = read_prediction_csv(csv_path)
    assert rows[0]["target_residual_deg"] == 0.75
    assert rows[0]["rank1_angle_abs_error_float"] == 0.75


def test_route_b_matcher_head_recovers_synthetic_residual(tmp_path):
    cache_root = tmp_path / "matches"
    train_rows = []
    eval_rows = []
    for idx in range(80):
        group = f"{idx:04d}"
        pair = f"{group}/01_02"
        residual = 1.0 if idx % 2 == 0 else -1.0
        _write_match(cache_root / group / "image-01_image-02_matches.npz", residual_signal=residual * 8.0)
        row = {
            "pair_id": pair,
            "group_id": group,
            "target_heading": residual,
            "target_distance": 0.0,
            "rank1_heading": 0.0,
            "rank1_distance": 0.0,
        }
        if idx < 60:
            train_rows.append(row)
        else:
            eval_rows.append(row)
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    _write_predictions(train_csv, train_rows)
    _write_predictions(eval_csv, eval_rows)

    args = SimpleNamespace(
        train_prediction_csv=[train_csv],
        eval_prediction_csv=eval_csv,
        cache_root=cache_root,
        output_dir=tmp_path / "out",
        folds=5,
        g1_angle_mae=0.3,
        g3_distance_mae=0.1,
    )

    result = run(args)

    assert result["train_matcher_coverage"]["coverage_rate"] == 1.0
    assert result["eval_corrected"]["angle_mae"] < result["eval_baseline"]["angle_mae"]
    assert result["eval_corrected"]["angle_mae"] <= 0.3
    assert result["decision"] == "promote_to_scale_review"
