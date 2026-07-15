from pathlib import Path

from scripts.phase57_make_split_manifests import build_split_manifests, normalize_surface_row, write_split_manifests


def _row(group, stem, split):
    return {
        "pair_id": f"{group}/{stem}",
        "group_id": group,
        "split": split,
        "source_json": f"/src/{group}/{stem}.json",
        "materialized_json": f"/surface/{split}/{group}/{stem}.json",
        "target_heading": 10.0,
        "target_distance": 20.0,
    }


def test_normalize_surface_row_prefers_materialized_json():
    row = normalize_surface_row(_row("0001", "01_02", "train"))

    assert row["pair_id"] == "0001/01_02"
    assert row["group_id"] == "0001"
    assert row["json_path"] == "/surface/train/0001/01_02.json"
    assert row["source_split"] == "train"


def test_build_split_manifests_group_disjoint_folds():
    rows = [
        _row("0001", "01_02", "train"),
        _row("0001", "02_03", "train"),
        _row("0002", "01_02", "train"),
        _row("0003", "01_02", "train"),
        _row("0004", "01_02", "val"),
    ]
    bundle = build_split_manifests(rows, folds=2, seed=57)

    assert len(bundle["files"]["all_labeled.jsonl"]) == 5
    assert len(bundle["files"]["fixed_val811.jsonl"]) == 1
    assert bundle["summary"]["train_rows"] == 4
    assert bundle["summary"]["fixed_val_rows"] == 1

    for fold_id in range(2):
        calib = bundle["files"][f"cv_fold_{fold_id:02d}_calib.jsonl"]
        holdout = bundle["files"][f"cv_fold_{fold_id:02d}_holdout.jsonl"]
        calib_groups = {row["group_id"] for row in calib}
        holdout_groups = {row["group_id"] for row in holdout}
        assert holdout
        assert not (calib_groups & holdout_groups)
        assert {row["role"] for row in holdout} == {"holdout"}
        assert {row["fold_id"] for row in holdout} == {fold_id}


def test_write_split_manifests(tmp_path: Path):
    bundle = build_split_manifests([_row("0001", "01_02", "train"), _row("0002", "01_02", "val")], folds=2, seed=57)
    summary = write_split_manifests(bundle, tmp_path)

    assert (tmp_path / "split_summary.json").is_file()
    assert (tmp_path / "all_labeled.jsonl").is_file()
    assert (tmp_path / "fixed_val811.jsonl").is_file()
    assert summary["written_files"]["all_labeled.jsonl"] == 2
