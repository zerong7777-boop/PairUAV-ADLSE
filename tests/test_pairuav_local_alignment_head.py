import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_module(module_name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pose_head_module = _load_module("pose_head_under_test_local_alignment", Path("reloc3r") / "pose_head.py")
misc_module = _load_module("misc_under_test_local_alignment", Path("reloc3r") / "utils" / "misc.py")

compute_local_alignment_summary = pose_head_module.compute_local_alignment_summary
LocalAlignmentPairUAVHead = pose_head_module.LocalAlignmentPairUAVHead
transpose_to_landscape = misc_module.transpose_to_landscape


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 8


class LocalAlignmentPairUAVHeadTest(unittest.TestCase):
    def test_compute_local_alignment_summary_returns_eight_similarity_features(self):
        decout = [torch.eye(3).unsqueeze(0)]
        paired_decout = [torch.eye(3).unsqueeze(0)]

        summary = compute_local_alignment_summary(decout, paired_decout)

        self.assertEqual(summary.shape, (1, 8))
        self.assertTrue(torch.isfinite(summary).all())
        self.assertAlmostEqual(summary[0, 0].item(), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(summary[0, 2].item(), 1.0, places=5)
        self.assertAlmostEqual(summary[0, 3].item(), 0.0, places=5)

    def test_compute_local_alignment_summary_missing_paired_decout_falls_back_to_zeros(self):
        decout = [torch.randn(2, 4, 8)]

        summary = compute_local_alignment_summary(decout, None)

        self.assertEqual(summary.shape, (2, 8))
        self.assertTrue(torch.equal(summary, torch.zeros_like(summary)))

    def test_local_alignment_head_output_contract_and_missing_paired_fallback(self):
        torch.manual_seed(11)
        head = LocalAlignmentPairUAVHead(_Net(), num_resconv_block=1, alignment_dropout=0.0)
        decout = [torch.randn(2, 4, 8)]

        out = head(decout, (32, 32))

        self.assertEqual(set(out.keys()), {"heading_vec", "range_value"})
        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(out["range_value"]).all())

    def test_transpose_to_landscape_slices_nested_list_tuple_tensor_kwargs(self):
        calls = []

        class _Head(torch.nn.Module):
            def forward(self, decout, img_shape, meta=None, pair=None):
                calls.append((img_shape, meta, pair))
                return {"range_value": torch.zeros(decout[-1].shape[0], 1)}

        wrapped = transpose_to_landscape(_Head(), activate=True)
        decout = [torch.randn(3, 4, 2)]
        true_shape = torch.tensor([[32, 64], [64, 32], [32, 64]])
        meta = [torch.tensor([10, 20, 30]), "keep"]
        pair = (torch.arange(12).view(3, 4), None)

        wrapped(decout, true_shape, meta=meta, pair=pair)

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][1][0], torch.tensor([10, 30])))
        self.assertEqual(calls[0][1][1], "keep")
        self.assertTrue(torch.equal(calls[0][2][0], torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]])))
        self.assertIsNone(calls[0][2][1])
        self.assertTrue(torch.equal(calls[1][1][0], torch.tensor([20])))
        self.assertTrue(torch.equal(calls[1][2][0], torch.tensor([[4, 5, 6, 7]])))

    def test_reloc3r_relpose_supports_local_alignment_output_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_local_alignment",
                Path("reloc3r") / "reloc3r_relpose.py",
            )
            model = module.Reloc3rRelpose(
                img_size=32,
                patch_size=16,
                enc_embed_dim=8,
                enc_depth=0,
                enc_num_heads=1,
                dec_embed_dim=8,
                dec_depth=0,
                dec_num_heads=1,
                output_mode="pairuav_local_alignment_heading_range",
                alignment_dropout=0.0,
            )
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        self.assertEqual(model.pose_head.__class__.__name__, "LocalAlignmentPairUAVHead")


if __name__ == "__main__":
    unittest.main()
