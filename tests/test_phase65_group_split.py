import pytest

from scripts.phase65_make_group_split import group_id_from_sample_id, make_group_split


def test_group_id_from_sample_id():
    assert group_id_from_sample_id("0839/01_27") == "0839"
    with pytest.raises(ValueError):
        group_id_from_sample_id("bad")


def test_make_group_split_has_no_group_overlap():
    sample_ids = [
        "a/00_01",
        "a/00_02",
        "b/00_01",
        "b/00_02",
        "c/00_01",
        "d/00_01",
    ]
    split = make_group_split(sample_ids, val_fraction=0.34, seed=7)
    train_groups = set(split["train_groups"])
    val_groups = set(split["val_groups"])
    assert train_groups
    assert val_groups
    assert not train_groups.intersection(val_groups)
    assert len(split["train_indices"]) + len(split["val_indices"]) == len(sample_ids)
    for index in split["train_indices"]:
        assert group_id_from_sample_id(sample_ids[index]) in train_groups
    for index in split["val_indices"]:
        assert group_id_from_sample_id(sample_ids[index]) in val_groups
