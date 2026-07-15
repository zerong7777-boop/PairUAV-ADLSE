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


pose_head_module = _load_module("pose_head_under_test_frozen_matcher", Path("reloc3r") / "pose_head.py")
FrozenMatcherFusionPairUAVHead = pose_head_module.FrozenMatcherFusionPairUAVHead
PairUAVHead = pose_head_module.PairUAVHead


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 8


class FrozenMatcherFusionPairUAVHeadTest(unittest.TestCase):
    def test_fusion_head_output_contract(self):
        torch.manual_seed(21)
        head = FrozenMatcherFusionPairUAVHead(_Net(), num_resconv_block=1, matcher_feature_dim=13)
        decout = [torch.randn(2, 4, 8)]
        features = torch.randn(2, 13)
        mask = torch.ones(2)

        out = head(decout, (32, 32), matcher_features=features, matcher_feature_mask=mask)

        self.assertEqual(set(out.keys()), {"heading_vec", "range_value"})
        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_zero_initialized_fusion_matches_baseline_loaded_state(self):
        torch.manual_seed(22)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        fusion = FrozenMatcherFusionPairUAVHead(_Net(), num_resconv_block=1, matcher_feature_dim=13)
        fusion.load_state_dict(baseline.state_dict(), strict=False)
        decout = [torch.randn(2, 4, 8)]
        features = torch.randn(2, 13)

        base_out = baseline(decout, (32, 32))
        fusion_out = fusion(decout, (32, 32), matcher_features=features, matcher_feature_mask=torch.ones(2))

        self.assertTrue(torch.allclose(base_out["heading_vec"], fusion_out["heading_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(base_out["range_value"], fusion_out["range_value"], atol=1e-6))

    def test_matcher_mask_suppresses_enabled_residual(self):
        torch.manual_seed(23)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        fusion = FrozenMatcherFusionPairUAVHead(_Net(), num_resconv_block=1, matcher_feature_dim=13)
        fusion.load_state_dict(baseline.state_dict(), strict=False)
        fusion.matcher_adapter_scale.data.fill_(2.0)
        decout = [torch.randn(2, 4, 8)]
        features = torch.randn(2, 13)

        base_out = baseline(decout, (32, 32))
        fusion_out = fusion(decout, (32, 32), matcher_features=features, matcher_feature_mask=torch.zeros(2))

        self.assertTrue(torch.allclose(base_out["heading_vec"], fusion_out["heading_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(base_out["range_value"], fusion_out["range_value"], atol=1e-6))

    def test_nonzero_matcher_residual_can_change_output(self):
        torch.manual_seed(24)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        fusion = FrozenMatcherFusionPairUAVHead(_Net(), num_resconv_block=1, matcher_feature_dim=13)
        fusion.load_state_dict(baseline.state_dict(), strict=False)
        fusion.matcher_adapter_scale.data.fill_(2.0)
        decout = [torch.randn(2, 4, 8)]
        features = torch.randn(2, 13)

        base_out = baseline(decout, (32, 32))
        fusion_out = fusion(decout, (32, 32), matcher_features=features, matcher_feature_mask=torch.ones(2))

        changed_heading = not torch.allclose(base_out["heading_vec"], fusion_out["heading_vec"], atol=1e-6)
        changed_range = not torch.allclose(base_out["range_value"], fusion_out["range_value"], atol=1e-6)
        self.assertTrue(changed_heading or changed_range)

    def test_reloc3r_relpose_supports_frozen_matcher_output_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_frozen_matcher",
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
                output_mode="pairuav_frozen_matcher_fusion_heading_range",
                matcher_feature_dim=13,
            )
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        self.assertEqual(model.pose_head.__class__.__name__, "FrozenMatcherFusionPairUAVHead")


if __name__ == "__main__":
    unittest.main()
