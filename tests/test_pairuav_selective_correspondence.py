import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


def _load_module(module_name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_npz(path):
    np.savez(
        path,
        keypoints0=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [8.0, 8.0]], dtype=np.float32),
        keypoints1=np.array([[1.0, 1.0], [14.0, 2.0], [0.0, 9.0], [9.0, 12.0]], dtype=np.float32),
        matches=np.array([0, 1, -1, 3], dtype=np.int64),
        match_confidence=np.array([0.5, 0.25, 0.0, 0.75], dtype=np.float32),
    )


features_module = _load_module(
    "pairuav_matcher_features_under_test_bscr",
    Path("reloc3r") / "datasets" / "pairuav_matcher_features.py",
)
pose_head_module = _load_module("pose_head_under_test_bscr", Path("reloc3r") / "pose_head.py")

BSCR_GLOBAL_FEATURE_NAMES = features_module.BSCR_GLOBAL_FEATURE_NAMES
extract_bscr_packet = features_module.extract_bscr_packet
build_bscr_feature_manifest = features_module.build_bscr_feature_manifest
load_bscr_feature_manifest = features_module.load_bscr_feature_manifest
bscr_tensor_for_sample = features_module.bscr_tensor_for_sample
SelectiveCorrespondencePairUAVHead = pose_head_module.SelectiveCorrespondencePairUAVHead
PairUAVHead = pose_head_module.PairUAVHead


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 8


class SelectiveCorrespondenceFeatureTest(unittest.TestCase):
    def test_extract_bscr_packet_has_fixed_structured_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_matches.npz"
            _write_npz(path)

            packet = extract_bscr_packet(path, image_size=(20, 20), grid_size=4, topk=16)

            self.assertEqual(packet["global_stats"].shape, (len(BSCR_GLOBAL_FEATURE_NAMES),))
            self.assertEqual(packet["spatial_bins"].shape, (4, 4, 4))
            self.assertEqual(packet["topk_anchors"].shape, (16, 5))
            self.assertEqual(packet["quality_mask"].shape, (1,))
            self.assertEqual(float(packet["fallback_used"]), 0.0)
            self.assertTrue(np.isfinite(packet["global_stats"]).all())
            self.assertGreater(packet["global_stats"][0], 0.0)

    def test_missing_bscr_packet_returns_zero_fallback(self):
        packet = extract_bscr_packet("/missing/file.npz", image_size=(20, 20), grid_size=4, topk=16)

        self.assertEqual(packet["global_stats"].shape, (len(BSCR_GLOBAL_FEATURE_NAMES),))
        self.assertEqual(packet["spatial_bins"].shape, (4, 4, 4))
        self.assertEqual(packet["topk_anchors"].shape, (16, 5))
        self.assertEqual(float(packet["fallback_used"]), 1.0)
        self.assertEqual(float(packet["quality_mask"][0]), 0.0)

    def test_bscr_manifest_loader_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            match_dir = cache_root / "0001"
            match_dir.mkdir(parents=True)
            _write_npz(match_dir / "image-01_image-02_matches.npz")
            manifest = root / "manifest.csv"
            manifest.write_text("sample_id,split\n0001/01_02,train\n", encoding="utf-8")
            output = root / "features.jsonl"
            stats = root / "summary.json"

            summary = build_bscr_feature_manifest(manifest, cache_root, output, stats_json=stats, split="train")
            loaded = load_bscr_feature_manifest(output)
            tensors = bscr_tensor_for_sample("0001/01_02", loaded)
            missing = bscr_tensor_for_sample("0001/03_04", loaded)

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["covered"], 1)
            self.assertEqual(tensors["bscr_spatial_bins"].shape, (4, 4, 4))
            self.assertEqual(tensors["bscr_topk_anchors"].shape, (16, 5))
            self.assertEqual(float(tensors["bscr_fallback_used"][0]), 0.0)
            self.assertEqual(float(missing["bscr_fallback_used"][0]), 1.0)


class SelectiveCorrespondenceHeadTest(unittest.TestCase):
    def _bscr_inputs(self, batch_size):
        return {
            "bscr_global_stats": torch.randn(batch_size, len(BSCR_GLOBAL_FEATURE_NAMES)),
            "bscr_spatial_bins": torch.randn(batch_size, 4, 4, 4),
            "bscr_topk_anchors": torch.randn(batch_size, 16, 5),
            "bscr_quality_mask": torch.ones(batch_size, 1),
            "bscr_fallback_used": torch.zeros(batch_size, 1),
        }

    def test_selective_head_output_contract_and_range_protection(self):
        torch.manual_seed(31)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        head = SelectiveCorrespondencePairUAVHead(_Net(), num_resconv_block=1)
        head.load_state_dict(baseline.state_dict(), strict=False)
        decout = [torch.randn(2, 4, 8)]

        base_out = baseline(decout, (32, 32))
        out = head(decout, (32, 32), **self._bscr_inputs(2))

        self.assertEqual({"heading_vec", "range_value", "bscr_gate", "bscr_heading_residual"}, set(out.keys()))
        self.assertTrue(torch.allclose(base_out["range_value"], out["range_value"], atol=1e-6))
        self.assertTrue(torch.allclose(base_out["heading_vec"], out["heading_vec"], atol=1e-6))
        self.assertEqual(out["bscr_gate"].shape, (2, 1))
        self.assertEqual(out["bscr_heading_residual"].shape, (2, 2))

    def test_force_gate_off_suppresses_residual_even_when_scale_enabled(self):
        torch.manual_seed(32)
        baseline = PairUAVHead(_Net(), num_resconv_block=1)
        head = SelectiveCorrespondencePairUAVHead(_Net(), num_resconv_block=1, force_gate_off=True)
        head.load_state_dict(baseline.state_dict(), strict=False)
        head.bscr_residual_scale.data.fill_(2.0)
        decout = [torch.randn(2, 4, 8)]

        base_out = baseline(decout, (32, 32))
        out = head(decout, (32, 32), **self._bscr_inputs(2))

        self.assertTrue(torch.allclose(base_out["heading_vec"], out["heading_vec"], atol=1e-6))
        self.assertTrue(torch.allclose(out["bscr_gate"], torch.zeros_like(out["bscr_gate"])))

    def test_reloc3r_relpose_supports_bscr_output_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        try:
            sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(PyTorchModelHubMixin=object))
            module = _load_module(
                "reloc3r_relpose_under_test_bscr",
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
                output_mode="pairuav_selective_correspondence_heading_range",
            )
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

        self.assertEqual(model.pose_head.__class__.__name__, "SelectiveCorrespondencePairUAVHead")


if __name__ == "__main__":
    unittest.main()
