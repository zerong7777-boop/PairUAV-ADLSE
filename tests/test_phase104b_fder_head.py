import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


def _load_pose_head_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "pose_head.py"
    if not module_path.exists():
        module_path = Path(__file__).resolve().parents[1] / ".tmp_uavm_stage" / "pose_head.py"
    spec = importlib.util.spec_from_file_location("pose_head_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pose_head_under_test"] = module
    spec.loader.exec_module(module)
    return module


pose_head = _load_pose_head_module()


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 32


def _fake_decout(batch_size=2, grid_size=4, dec_embed_dim=32):
    token_count = grid_size * grid_size
    return [
        torch.randn(batch_size, token_count, dec_embed_dim * 2),
        torch.randn(batch_size, token_count, dec_embed_dim),
        torch.randn(batch_size, token_count, dec_embed_dim),
        torch.randn(batch_size, token_count, dec_embed_dim),
    ]


class Phase104bFDERHeadTest(unittest.TestCase):
    def test_fixed_alpha_starts_as_identity_heading_residual_and_protects_range(self):
        torch.manual_seed(11)
        head = pose_head.Phase104bFDERPairUAVHead(
            _Net(),
            use_sample_router=False,
            heading_residual_max_delta_deg=1.0,
        )
        out = head(_fake_decout(), (64, 64))

        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))
        self.assertEqual(out["phase104b_heading_layer_weights"].shape, (2, 3))
        self.assertEqual(out["phase104b_range_diag_layer_weights"].shape, (2, 3))
        self.assertEqual(out["phase104b_heading_gate"].shape, (2, 1))
        self.assertEqual(out["phase104b_heading_residual_delta_deg"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["heading_vec"], out["phase104b_heading_base_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(out["phase104b_heading_residual_delta_deg"], torch.zeros(2, 1), atol=1e-6))
        self.assertTrue(torch.allclose(out["range_value"], out["phase104b_protected_range_value"], atol=1e-6))
        self.assertTrue(torch.allclose(out["phase104b_heading_layer_weights"].sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(out["phase104b_range_diag_layer_weights"].sum(dim=-1), torch.ones(2), atol=1e-6))

    def test_sample_router_produces_per_sample_weights_and_bounded_delta(self):
        torch.manual_seed(17)
        head = pose_head.Phase104bFDERPairUAVHead(
            _Net(),
            use_sample_router=True,
            heading_residual_max_delta_deg=1.0,
        )
        out = head(_fake_decout(), (64, 64))

        self.assertEqual(out["phase104b_heading_layer_weights"].shape, (2, 3))
        self.assertEqual(out["phase104b_heading_router_entropy"].shape, (2,))
        self.assertEqual(out["phase104b_per_layer_heading_evidence_norm"].shape, (2, 3))
        self.assertEqual(out["phase104b_per_layer_contribution"].shape, (2, 3))
        self.assertTrue(torch.isfinite(out["heading_vec"]).all())
        self.assertTrue(torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertLessEqual(out["phase104b_heading_residual_delta_rad"].abs().max().item(), math.radians(1.0) + 1e-6)


if __name__ == "__main__":
    unittest.main()
