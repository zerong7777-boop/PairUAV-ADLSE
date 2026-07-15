import importlib.util
import math
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


pose_head = _load_module("pose_head_under_test_phase104c", Path("reloc3r") / "pose_head.py")


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


class Phase104cOfferHeadTest(unittest.TestCase):
    def test_offer_head_outputs_typed_observability_and_active_residual(self):
        torch.manual_seed(10401)
        head = pose_head.Phase104cObservabilityFactorRouterHead(
            _Net(),
            use_observability_router=True,
            heading_residual_max_delta_deg=5.0,
            router_hidden_dim=64,
            task_token_num_heads=4,
        )
        out = head(_fake_decout(), (64, 64))

        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))
        self.assertEqual(out["phase104c_heading_base_vec"].shape, (2, 2))
        self.assertEqual(out["phase104c_protected_range_value"].shape, (2, 1))
        self.assertEqual(out["phase104c_heading_gate"].shape, (2, 1))
        self.assertEqual(out["phase104c_heading_residual_delta_deg"].shape, (2, 1))
        self.assertEqual(out["phase104c_heading_observability"].shape, (2, 1))
        self.assertEqual(out["phase104c_range_observability"].shape, (2, 1))
        self.assertEqual(out["phase104c_heading_evidence_weights"].shape, (2, 4))
        self.assertEqual(out["phase104c_range_diag_evidence_weights"].shape, (2, 4))
        self.assertEqual(out["phase104c_typed_evidence_norm"].shape, (2, 4))
        self.assertEqual(out["phase104c_evidence_layer_attention"].shape, (2, 12))

        self.assertTrue(torch.allclose(out["range_value"], out["phase104c_protected_range_value"], atol=1e-6))
        self.assertTrue(torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(out["phase104c_heading_evidence_weights"].sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(out["phase104c_range_diag_evidence_weights"].sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertGreater(out["phase104c_heading_gate"].mean().item(), 0.02)
        self.assertLess(out["phase104c_heading_gate"].mean().item(), 0.8)
        self.assertGreater(out["phase104c_heading_residual_delta_deg"].abs().max().item(), 1e-8)
        self.assertLessEqual(out["phase104c_heading_residual_delta_deg"].abs().max().item(), 5.0 + 1e-6)
        self.assertTrue(((out["phase104c_heading_observability"] >= 0.0) & (out["phase104c_heading_observability"] <= 1.0)).all())
        self.assertTrue(((out["phase104c_range_observability"] >= 0.0) & (out["phase104c_range_observability"] <= 1.0)).all())
        self.assertGreater(out["phase104c_heading_evidence_weights"][:, 0].mean().item(), 0.40)
        self.assertGreater(out["phase104c_heading_evidence_weights"][:, 2].mean().item(), 0.25)
        self.assertLess(out["phase104c_heading_evidence_weights"][:, 1].mean().item(), 0.15)
        self.assertGreater(out["phase104c_range_diag_evidence_weights"][:, 1].mean().item(), 0.45)

    def test_heading_only_loss_does_not_backprop_into_protected_range_path(self):
        torch.manual_seed(10402)
        head = pose_head.Phase104cObservabilityFactorRouterHead(
            _Net(),
            use_observability_router=True,
            heading_residual_max_delta_deg=5.0,
            router_hidden_dim=64,
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
        heading_grads = [
            param.grad
            for name, param in head.named_parameters()
            if "heading" in name and param.requires_grad and param.grad is not None
        ]
        self.assertTrue(heading_grads)
        self.assertGreater(sum(float(g.abs().sum()) for g in heading_grads), 0.0)

    def test_reloc3r_relpose_registers_phase104c_modes(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_phase104c",
                Path("reloc3r") / "reloc3r_relpose.py",
            )
            for output_mode in (
                "pairuav_phase104c_offer_fixed_heading_range",
                "pairuav_phase104c_offer_heading_range",
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
                self.assertEqual(model.pose_head.__class__.__name__, "Phase104cObservabilityFactorRouterHead")
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))


class Phase104cObservabilityLossTest(unittest.TestCase):
    def test_observability_loss_is_conditional_and_supervised_by_detached_proxy(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            loss_module = _load_module("loss_under_test_phase104c", Path("reloc3r") / "loss.py")
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        loss_fn = loss_module.PairUAVOfficialMetricAwareLoss()
        gt2 = _view([10.0, 45.0], [20.0, 40.0])
        pred = _pred([12.0, 60.0], [21.0, 35.0])
        base_loss, base_details = loss_fn.compute_loss({}, gt2, {}, pred)
        self.assertNotIn("phase104c_observability_loss", base_details)

        pred_with_obs = dict(pred)
        pred_with_obs.update(
            {
                "phase104c_heading_base_vec": _pred([20.0, 80.0], [0.0, 0.0])["heading_vec"].detach(),
                "phase104c_protected_range_value": torch.tensor([[25.0], [30.0]]),
                "phase104c_heading_observability": torch.tensor([[0.9], [0.9]], requires_grad=True),
                "phase104c_range_observability": torch.tensor([[0.8], [0.8]], requires_grad=True),
            }
        )
        total_loss, details = loss_fn.compute_loss({}, gt2, {}, pred_with_obs)

        self.assertIn("phase104c_observability_loss", details)
        self.assertIn("phase104c_heading_observability_target_mean", details)
        self.assertIn("phase104c_range_observability_target_mean", details)
        self.assertGreater(float(total_loss), float(base_loss))
        total_loss.backward()
        self.assertIsNotNone(pred_with_obs["phase104c_heading_observability"].grad)
        self.assertIsNotNone(pred_with_obs["phase104c_range_observability"].grad)


if __name__ == "__main__":
    unittest.main()
