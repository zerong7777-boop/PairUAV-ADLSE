import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module(module_name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


loader_module = _load_module("phase64_token_shards_under_test", Path("reloc3r") / "phase64_token_shards.py")


def _write_fixture(root):
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard_000000.npz"
    np.savez_compressed(
        shard_path,
        sample_id=np.asarray(["0001/01_02", "0001/02_03"], dtype=object),
        target_heading=np.asarray([10.0, -20.0], dtype=np.float32),
        target_distance=np.asarray([5.0, 6.0], dtype=np.float32),
        rank1_heading=np.asarray([9.9, -19.8], dtype=np.float32),
        rank1_distance=np.asarray([5.1, 6.1], dtype=np.float32),
        rank1_angle_abs_error=np.asarray([0.1, 0.2], dtype=np.float32),
        tokens=np.zeros((2, 4, 18), dtype=np.float32),
        token_mask=np.ones((2, 4), dtype=np.float32),
        hypothesis_features=np.zeros((2, 3, 9), dtype=np.float32),
        global_stats=np.zeros((2, 18), dtype=np.float32),
        fallback_used=np.zeros((2,), dtype=np.float32),
        valid_matches=np.asarray([4, 4], dtype=np.int32),
        total_matches=np.asarray([4, 4], dtype=np.int32),
        match_path=np.asarray(["a.npz", "b.npz"], dtype=object),
    )
    manifest = {
        "format": "phase64_token_shards_v1",
        "rows": 2,
        "shards": [
            {
                "path": str(shard_path),
                "name": "shard_000000.npz",
                "start": 0,
                "rows": 2,
                "covered": 2,
                "fallback": 0,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class Phase64TokenLoaderTest(unittest.TestCase):
    def test_numpy_loader_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_fixture(Path(tmp))
            dataset = loader_module.Phase64TokenShardDataset(manifest)
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(sample["tokens"].shape, (4, 18))
            self.assertEqual(sample["hypothesis_features"].shape, (3, 9))
            self.assertAlmostEqual(float(sample["residual_target"]), 0.1, places=3)
            batch = dataset.batch_numpy([0, 1])
            self.assertEqual(batch["tokens"].shape, (2, 4, 18))
            summary = loader_module.validate_manifest(manifest)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["tokens_shape"], [4, 18])


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not importable in this environment")
class Phase64TokenModelTest(unittest.TestCase):
    def test_model_starts_at_rank1_parity(self):
        import torch

        model_module = _load_module("phase64_token_angle_model_under_test", Path("reloc3r") / "phase64_token_angle_model.py")
        model = model_module.Phase64TokenAngleSpecialist(
            token_dim=18,
            hypothesis_dim=9,
            global_dim=18,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_residual_deg=0.3,
        )
        outputs = model(
            tokens=torch.zeros(2, 4, 18),
            token_mask=torch.ones(2, 4),
            hypothesis_features=torch.zeros(2, 3, 9),
            global_stats=torch.zeros(2, 18),
            rank1_heading=torch.tensor([10.0, -20.0]),
            rank1_distance=torch.tensor([5.0, 6.0]),
        )
        self.assertTrue(torch.allclose(outputs["corrected_heading"], torch.tensor([10.0, -20.0]), atol=1e-6))
        self.assertTrue(torch.allclose(outputs["residual"], torch.zeros(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
