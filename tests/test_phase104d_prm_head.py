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


pose_head = _load_module("pose_head_under_test_phase104d", Path("reloc3r") / "pose_head.py")


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


def _view(heading_deg, range_value):
    heading_deg = torch.tensor(heading_deg, dtype=torch.float32)
    range_value = torch.tensor(range_value, dtype=torch.float32)
    heading_rad = torch.deg2rad(heading_deg)
    return {
        "heading_deg": heading_deg,
        "heading_cos": torch.cos(heading_rad),
        "heading_sin": torch.sin(heading_rad),
        "range_value": range_value,
    }


def _pred(heading_deg, range_value):
    heading_deg = torch.tensor(heading_deg, dtype=torch.float32)
    range_value = torch.tensor(range_value, dtype=torch.float32)
    heading_rad = torch.deg2rad(heading_deg)
    return {
        "heading_vec": torch.stack((torch.cos(heading_rad), torch.sin(heading_rad)), dim=-1),
        "range_value": range_value.view(-1, 1),
    }


class Phase104dPRMHeadTest(unittest.TestCase):
    def test_prm_r0_outputs_structurally_typed_slots_and_protected_range(self):
        torch.manual_seed(10441)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(), (64, 64))

        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))
        self.assertEqual(out["phase104d_heading_base_vec"].shape, (2, 2))
        self.assertEqual(out["phase104d_protected_range_value"].shape, (2, 1))
        self.assertEqual(out["phase104d_heading_gate"].shape, (2, 1))
        self.assertEqual(out["phase104d_heading_residual_delta_deg"].shape, (2, 1))
        self.assertEqual(out["phase104d_slot_norm"].shape, (2, 5))
        self.assertEqual(out["phase104d_slot_source_mass"].shape, (2, 5, 5))
        self.assertEqual(out["phase104d_slot_attention_entropy"].shape, (2, 5))
        self.assertNotIn("phase104d_bearing_bin_logits", out)
        self.assertNotIn("phase104d_scale_bin_logits", out)

        self.assertTrue(torch.allclose(out["range_value"], out["phase104d_protected_range_value"], atol=1e-6))
        self.assertTrue(torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertGreater(out["phase104d_heading_gate"].mean().item(), 0.02)
        self.assertLess(out["phase104d_heading_gate"].mean().item(), 0.8)
        self.assertGreater(out["phase104d_heading_residual_delta_deg"].abs().max().item(), 1e-8)
        self.assertLessEqual(out["phase104d_heading_residual_delta_deg"].abs().max().item(), 5.0 + 1e-6)

        source_mass = out["phase104d_slot_source_mass"]
        early, mid, late, agreement, disagreement = range(5)
        bearing, _layout, scale, overlap, ambiguity = range(5)
        self.assertLess(source_mass[:, bearing, early].abs().max().item(), 1e-6)
        self.assertGreater(source_mass[:, bearing, mid].mean().item(), 0.05)
        self.assertGreater(source_mass[:, bearing, late].mean().item(), 0.05)
        self.assertGreater(source_mass[:, scale, late].mean().item(), 0.99)
        self.assertLess(source_mass[:, scale, mid].abs().max().item(), 1e-6)
        self.assertGreater((source_mass[:, overlap, agreement] + source_mass[:, overlap, disagreement]).mean().item(), 0.99)
        self.assertLess((source_mass[:, overlap, mid] + source_mass[:, overlap, late]).abs().max().item(), 1e-6)
        self.assertGreater(source_mass[:, ambiguity, disagreement].mean().item(), 0.99)

    def test_prm_r1_outputs_auxiliary_logits(self):
        torch.manual_seed(10442)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=True,
            heading_bin_count=12,
            range_bin_count=8,
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(), (64, 64))

        self.assertEqual(out["phase104d_bearing_bin_logits"].shape, (2, 12))
        self.assertEqual(out["phase104d_scale_bin_logits"].shape, (2, 8))
        self.assertEqual(out["phase104d_ambiguity_pred"].shape, (2, 1))

    def test_prm_r2_direct_memory_heading_uses_slots_without_range_leak(self):
        torch.manual_seed(10447)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            heading_readout_mode="direct_memory",
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        decout = _fake_decout()
        out = head(decout, (64, 64))
        masked = head(decout, (64, 64), phase104d_mask_slot=0)

        self.assertEqual(out["phase104d_heading_memory_gate"].shape, (2, 1))
        self.assertEqual(out["phase104d_heading_memory_vec"].shape, (2, 2))
        self.assertTrue(torch.allclose(out["range_value"], out["phase104d_protected_range_value"], atol=1e-6))
        self.assertGreater(out["phase104d_heading_memory_gate"].mean().item(), 0.05)
        self.assertLess(out["phase104d_heading_memory_gate"].mean().item(), 0.9)
        self.assertGreater(out["phase104d_base_memory_delta_deg"].mean().item(), 1e-4)
        self.assertGreater((out["heading_vec"] - masked["heading_vec"]).abs().max().item(), 1e-6)

        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = torch.nn.functional.smooth_l1_loss(out["heading_vec"], target)
        loss.backward()
        self.assertIsNone(head.fc_range.weight.grad)
        self.assertIsNone(head.fc_range.bias.grad)
        self.assertIsNone(head.proj.weight.grad)
        self.assertIsNone(head.proj.bias.grad)

    def test_prm_r3_bounded_memory_delta_keeps_heading_correction_bounded(self):
        torch.manual_seed(10448)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            heading_readout_mode="bounded_memory_delta",
            heading_residual_max_delta_deg=20.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        decout = _fake_decout()
        out = head(decout, (64, 64))
        masked = head(decout, (64, 64), phase104d_mask_slot=0)

        self.assertEqual(out["phase104d_heading_memory_delta_deg"].shape, (2, 1))
        self.assertEqual(out["phase104d_heading_bounded_memory_delta_deg"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["range_value"], out["phase104d_protected_range_value"], atol=1e-6))
        self.assertGreater(out["phase104d_heading_memory_gate"].mean().item(), 0.05)
        self.assertLessEqual(out["phase104d_heading_residual_delta_deg"].abs().max().item(), 20.0 + 1e-6)
        self.assertGreater((out["heading_vec"] - masked["heading_vec"]).abs().max().item(), 1e-6)

    def test_prm_supports_counterfactual_slot_masking(self):
        torch.manual_seed(10444)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(), (64, 64), phase104d_mask_slot=0)

        self.assertEqual(out["phase104d_slot_mask"].shape, (2, 5))
        self.assertTrue(torch.allclose(out["phase104d_slot_mask"][:, 0], torch.zeros(2)))
        self.assertTrue(torch.allclose(out["phase104d_slot_mask"][:, 1:], torch.ones(2, 4)))

    def test_prm_source_banks_are_spatially_compressed_before_attention(self):
        torch.manual_seed(10445)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            source_pool_grid=(2, 2),
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(grid_size=8), (128, 128))

        self.assertEqual(out["phase104d_slot_source_token_count"].shape, (2, 5))
        self.assertEqual(out["phase104d_slot_source_token_count"][0].tolist(), [8, 12, 4, 8, 12])
        self.assertEqual(out["phase104d_slot_source_token_count"][1].tolist(), [8, 12, 4, 8, 12])

    def test_prm_heavy_source_projections_happen_after_spatial_pooling(self):
        torch.manual_seed(10446)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            source_pool_grid=(2, 2),
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        seen_lengths = {}

        def record_length(name):
            def hook(_module, inputs, _output):
                seen_lengths[name] = inputs[0].shape[1]

            return hook

        handles = [
            head.layout_gap_proj.register_forward_hook(record_length("layout_gap")),
            head.overlap_agreement_proj.register_forward_hook(record_length("overlap_agreement")),
            head.overlap_disagreement_proj.register_forward_hook(record_length("overlap_disagreement")),
            head.ambiguity_gap_proj.register_forward_hook(record_length("ambiguity_gap")),
            head.ambiguity_std_proj.register_forward_hook(record_length("ambiguity_std")),
        ]
        try:
            head(_fake_decout(grid_size=8), (128, 128))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(
            seen_lengths,
            {
                "layout_gap": 4,
                "overlap_agreement": 4,
                "overlap_disagreement": 4,
                "ambiguity_gap": 4,
                "ambiguity_std": 4,
            },
        )

    def test_heading_only_loss_does_not_backprop_into_prm_protected_range_path(self):
        torch.manual_seed(10443)
        head = pose_head.Phase104dPolarRelationMemoryHead(
            _Net(),
            use_auxiliary=False,
            heading_residual_max_delta_deg=5.0,
            slot_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(), (64, 64))
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = torch.nn.functional.smooth_l1_loss(out["heading_vec"], target)
        loss.backward()

        self.assertIsNone(head.fc_range.weight.grad)
        self.assertIsNone(head.fc_range.bias.grad)
        self.assertIsNone(head.proj.weight.grad)
        self.assertIsNone(head.proj.bias.grad)
        prm_grads = [
            param.grad
            for name, param in head.named_parameters()
            if ("slot" in name or "heading" in name) and param.requires_grad and param.grad is not None
        ]
        self.assertTrue(prm_grads)
        self.assertGreater(sum(float(g.abs().sum()) for g in prm_grads), 0.0)

    def test_reloc3r_relpose_registers_phase104d_modes(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_phase104d",
                Path("reloc3r") / "reloc3r_relpose.py",
            )
            for output_mode in (
                "pairuav_phase104d_prm_r0_heading_range",
                "pairuav_phase104d_prm_r1_aux_heading_range",
                "pairuav_phase104d_prm_r2_direct_heading_range",
                "pairuav_phase104d_prm_r3_bounded_delta_heading_range",
            ):
                model = module.Reloc3rRelpose(
                    img_size=32,
                    patch_size=16,
                    enc_embed_dim=8,
                    enc_depth=0,
                    enc_num_heads=1,
                    dec_embed_dim=8,
                    dec_depth=0,
                    dec_num_heads=1,
                    output_mode=output_mode,
                    phase104_task_token_num_heads=1,
                )
                self.assertEqual(model.pose_head.__class__.__name__, "Phase104dPolarRelationMemoryHead")
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))


class Phase104dPRMLossTest(unittest.TestCase):
    def test_prm_aux_loss_is_conditional_and_uses_detached_targets(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            loss_module = _load_module("loss_under_test_phase104d", Path("reloc3r") / "loss.py")
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        loss_fn = loss_module.PairUAVOfficialMetricAwareLoss()
        gt2 = _view([10.0, 45.0], [20.0, 40.0])
        pred = _pred([12.0, 60.0], [21.0, 35.0])
        base_loss, base_details = loss_fn.compute_loss({}, gt2, {}, pred)
        self.assertNotIn("phase104d_aux_loss", base_details)

        pred_with_aux = dict(pred)
        pred_with_aux.update(
            {
                "phase104d_heading_base_vec": _pred([20.0, 80.0], [0.0, 0.0])["heading_vec"].detach(),
                "phase104d_protected_range_value": torch.tensor([[25.0], [30.0]]),
                "phase104d_bearing_bin_logits": torch.randn(2, 12, requires_grad=True),
                "phase104d_scale_bin_logits": torch.randn(2, 8, requires_grad=True),
                "phase104d_ambiguity_pred": torch.tensor([[0.2], [0.7]], requires_grad=True),
            }
        )
        total_loss, details = loss_fn.compute_loss({}, gt2, {}, pred_with_aux)

        self.assertIn("phase104d_aux_loss", details)
        self.assertIn("phase104d_heading_bin_acc", details)
        self.assertIn("phase104d_scale_bin_acc", details)
        self.assertGreater(float(total_loss), float(base_loss))
        total_loss.backward()
        self.assertIsNotNone(pred_with_aux["phase104d_bearing_bin_logits"].grad)
        self.assertIsNotNone(pred_with_aux["phase104d_scale_bin_logits"].grad)
        self.assertIsNotNone(pred_with_aux["phase104d_ambiguity_pred"].grad)


if __name__ == "__main__":
    unittest.main()
