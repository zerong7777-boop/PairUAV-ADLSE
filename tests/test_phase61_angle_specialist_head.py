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


pose_head_module = _load_module("pose_head_under_test_phase61", Path("reloc3r") / "pose_head.py")
AngleSpecialistPairUAVHead = pose_head_module.AngleSpecialistPairUAVHead
PairUAVHead = pose_head_module.PairUAVHead


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 8


class AngleSpecialistPairUAVHeadTest(unittest.TestCase):
    def test_zero_scale_matches_base_head_after_partial_state_load(self):
        torch.manual_seed(6101)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        specialist = AngleSpecialistPairUAVHead(
            _Net(),
            num_resconv_block=1,
            angle_specialist_hidden_dim=16,
            angle_specialist_init_scale=0.0,
        )
        specialist.load_state_dict(baseline.state_dict(), strict=False)
        decout = [torch.randn(2, 4, 8)]

        base_out = baseline(decout, (32, 32))
        specialist_out = specialist(decout, (32, 32), paired_decout=decout)

        self.assertTrue(torch.allclose(base_out["heading_vec"], specialist_out["heading_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(base_out["range_value"], specialist_out["range_value"], atol=1e-6))
        self.assertEqual(specialist_out["angle_specialist_residual"].shape, (2, 2))
        self.assertEqual(specialist_out["angle_specialist_gate"].shape, (2, 1))

    def test_nonzero_specialist_residual_can_change_heading_without_changing_range(self):
        torch.manual_seed(6102)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        specialist = AngleSpecialistPairUAVHead(
            _Net(),
            num_resconv_block=1,
            angle_specialist_hidden_dim=16,
            angle_specialist_init_scale=1.0,
        )
        specialist.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            specialist.angle_specialist_gate.weight.zero_()
            specialist.angle_specialist_gate.bias.fill_(10.0)
            specialist.angle_specialist_delta_heading.weight.zero_()
            specialist.angle_specialist_delta_heading.bias.copy_(torch.tensor([0.2, -0.1]))
        decout = [torch.randn(2, 4, 8)]

        base_out = baseline(decout, (32, 32))
        specialist_out = specialist(decout, (32, 32), paired_decout=decout)

        self.assertFalse(torch.allclose(base_out["heading_vec"], specialist_out["heading_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(base_out["range_value"], specialist_out["range_value"], atol=1e-6))
        self.assertTrue(torch.allclose(specialist_out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_reloc3r_relpose_supports_angle_specialist_output_mode_and_freeze_policy(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_phase61",
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
                output_mode="pairuav_angle_specialist_heading_range",
                angle_specialist_hidden_dim=16,
            )
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        self.assertEqual(model.pose_head.__class__.__name__, "AngleSpecialistPairUAVHead")
        summary = model.freeze_except_angle_specialist()
        self.assertGreater(summary["trainable_params"], 0)
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("angle_specialist" in name for name in trainable))
        self.assertTrue(any("angle_specialist_delta_heading" in name for name in trainable))


if __name__ == "__main__":
    unittest.main()
