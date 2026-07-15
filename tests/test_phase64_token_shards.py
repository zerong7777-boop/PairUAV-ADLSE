import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_script():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "phase64_build_token_shards.py"
    spec = importlib.util.spec_from_file_location("phase64_build_token_shards_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_npz(path, shift=(3.0, 1.0)):
    keypoints0 = np.array(
        [[2.0, 2.0], [10.0, 2.0], [2.0, 10.0], [10.0, 10.0], [6.0, 6.0]],
        dtype=np.float32,
    )
    keypoints1 = keypoints0 + np.asarray(shift, dtype=np.float32)
    np.savez(
        path,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=np.arange(len(keypoints0), dtype=np.int64),
        match_confidence=np.linspace(0.9, 0.5, len(keypoints0), dtype=np.float32),
    )


class Phase64TokenShardTest(unittest.TestCase):
    def test_builder_writes_manifest_and_npz_shapes(self):
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            match_dir = cache_root / "0001"
            match_dir.mkdir(parents=True)
            _write_npz(match_dir / "image-01_image-02_matches.npz")
            _write_npz(match_dir / "image-02_image-03_matches.npz", shift=(1.0, 2.0))

            predictions = root / "predictions.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pair_id",
                        "split",
                        "target_heading",
                        "target_distance",
                        "rank1_heading",
                        "rank1_distance",
                        "rank1_angle_abs_error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "0001/01_02",
                        "split": "train",
                        "target_heading": "10.0",
                        "target_distance": "5.0",
                        "rank1_heading": "10.1",
                        "rank1_distance": "5.1",
                        "rank1_angle_abs_error": "0.1",
                    }
                )
                writer.writerow(
                    {
                        "pair_id": "0001/02_03",
                        "split": "train",
                        "target_heading": "20.0",
                        "target_distance": "6.0",
                        "rank1_heading": "19.9",
                        "rank1_distance": "6.1",
                        "rank1_angle_abs_error": "0.1",
                    }
                )

            rows = module.read_prediction_rows(predictions, split="train")
            arrays = module.build_arrays(
                rows,
                cache_root=cache_root,
                image_size=(20, 20),
                topk=4,
                grid_size=4,
                residual_threshold=0.05,
            )
            shard_path = root / "out" / "shards" / "shard_000000.npz"
            module.write_shard(shard_path, arrays)

            loaded = np.load(shard_path, allow_pickle=True)
            self.assertEqual(loaded["tokens"].shape, (2, 4, len(module.TOKENS.TOKEN_FEATURE_NAMES)))
            self.assertEqual(loaded["token_mask"].shape, (2, 4))
            self.assertEqual(loaded["hypothesis_features"].shape[0], 2)
            self.assertEqual(loaded["global_stats"].shape[0], 2)
            self.assertEqual(float(loaded["fallback_used"].sum()), 0.0)

            args = SimpleNamespace(
                predictions_csv=str(predictions),
                cache_root=str(cache_root),
                output_root=str(root / "out"),
                shard_size=8,
                topk=4,
                grid_size=4,
                image_width=20,
                image_height=20,
                residual_threshold=0.05,
            )
            manifest = module.manifest_summary(
                [
                    {
                        "path": str(shard_path),
                        "name": "shard_000000.npz",
                        "start": 0,
                        "rows": 2,
                        "covered": 2,
                        "fallback": 0,
                        "valid_matches_summary": [5, 5],
                    }
                ],
                rows,
                args,
            )
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["covered"], 2)
            self.assertEqual(manifest["coverage_rate"], 1.0)
            self.assertEqual(manifest["topk"], 4)
            self.assertIn("token_feature_names", manifest)
            json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()
