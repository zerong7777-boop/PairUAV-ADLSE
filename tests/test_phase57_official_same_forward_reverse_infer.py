import zipfile

import torch

from scripts.phase57_official_same_forward_reverse_infer import (
    apply_heading_correction,
    apply_forward_heading_correction,
    circular_mean_deg_tensor,
    materialize_max_samples_json_root,
    select_first_json_paths_fast,
    wrap_angle_deg_tensor,
    write_zip,
)


def test_wrap_angle_deg_tensor():
    values = wrap_angle_deg_tensor(torch.tensor([181.0, -181.0, 360.0, -360.0]))
    assert values.tolist() == [-179.0, 179.0, 0.0, 0.0]


def test_circular_mean_deg_tensor_crosses_boundary():
    value = circular_mean_deg_tensor(torch.tensor([179.0]), torch.tensor([-179.0]))
    assert abs(abs(float(value[0])) - 180.0) < 1e-5


def test_p90_gated_correction_uses_average_only_under_threshold():
    final, reverse_forward, disagreement, avg, used = apply_heading_correction(
        torch.tensor([10.0, 10.0]),
        torch.tensor([-10.05, -11.0]),
        method="m5_reverse_disagreement_gated_avg_p90",
        threshold=0.11338043199998538,
    )

    assert torch.allclose(reverse_forward, torch.tensor([10.05, 11.0]), atol=1e-5)
    assert torch.allclose(disagreement, torch.tensor([0.05, 1.0]), atol=1e-5)
    assert used.tolist() == [True, False]
    assert abs(float(final[0]) - float(avg[0])) < 1e-5
    assert float(final[1]) == 10.0


def test_true_swap_average_uses_forward_candidate_without_inverse_again():
    final, disagreement, avg, used = apply_forward_heading_correction(
        torch.tensor([179.0, 10.0]),
        torch.tensor([-179.0, 12.0]),
        method="m6_true_swap_average",
        threshold=0.0,
    )

    assert torch.allclose(disagreement, torch.tensor([2.0, 2.0]), atol=1e-5)
    assert used.tolist() == [True, True]
    assert abs(abs(float(final[0])) - 180.0) < 1e-5
    assert float(final[1]) == 11.0
    assert torch.allclose(final, wrap_angle_deg_tensor(avg), atol=1e-5)


def test_true_swap_p90_gate_keeps_rank1_when_disagreement_is_large():
    final, disagreement, avg, used = apply_forward_heading_correction(
        torch.tensor([10.0, 10.0]),
        torch.tensor([10.05, 11.0]),
        method="m7_true_swap_disagreement_gated_avg_p90",
        threshold=0.11338043199998538,
    )

    assert torch.allclose(disagreement, torch.tensor([0.05, 1.0]), atol=1e-5)
    assert used.tolist() == [True, False]
    assert abs(float(final[0]) - float(avg[0])) < 1e-5
    assert float(final[1]) == 10.0


def test_write_zip_contains_root_result_txt(tmp_path):
    result = tmp_path / "result.txt"
    result.write_text("1.000000 2.000000\n", encoding="utf-8")

    zip_path = write_zip(result)

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["result.txt"]
        assert archive.read("result.txt").decode("utf-8") == "1.000000 2.000000\n"


def test_materialize_max_samples_json_root_uses_numeric_order(tmp_path):
    source = tmp_path / "source"
    (source / "0002").mkdir(parents=True)
    (source / "0001").mkdir(parents=True)
    (source / "0001" / "0003.json").write_text("{}", encoding="utf-8")
    (source / "0001" / "0001.json").write_text("{}", encoding="utf-8")
    (source / "0002" / "0000.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "out"
    report = materialize_max_samples_json_root(source, out, 2)

    assert report["written"] == 2
    assert (out / "0001" / "0001.json").is_file()
    assert (out / "0001" / "0003.json").is_file()
    assert not (out / "0002" / "0000.json").exists()


def test_select_first_json_paths_fast_stops_in_numeric_order(tmp_path):
    source = tmp_path / "source"
    (source / "0002").mkdir(parents=True)
    (source / "0001").mkdir(parents=True)
    (source / "0001" / "0004.json").write_text("{}", encoding="utf-8")
    (source / "0001" / "0002.json").write_text("{}", encoding="utf-8")
    (source / "0002" / "0000.json").write_text("{}", encoding="utf-8")

    selected = select_first_json_paths_fast(source, 2)

    assert [path.name for path in selected] == ["0002.json", "0004.json"]
