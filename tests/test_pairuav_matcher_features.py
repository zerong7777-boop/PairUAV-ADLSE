import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reloc3r.datasets.pairuav import PairUAV
from reloc3r.datasets.pairuav_matcher_features import (
    MATCHER_FEATURE_NAMES,
    apply_normalization,
    compute_normalization,
    extract_matcher_features,
    feature_tensor_for_sample,
    load_feature_manifest,
)


class PairUAVMatcherFeaturesTest(unittest.TestCase):
    def _write_npz(self, path):
        np.savez(
            path,
            keypoints0=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32),
            keypoints1=np.array([[1.0, 1.0], [14.0, 2.0], [0.0, 9.0]], dtype=np.float32),
            matches=np.array([0, 1, -1], dtype=np.int64),
            match_confidence=np.array([0.5, 0.25, 0.0], dtype=np.float32),
        )

    def test_extract_matcher_features_fixed_schema_and_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_matches.npz"
            self._write_npz(path)

            features = extract_matcher_features(path, image_width=20, image_height=20)

            self.assertEqual(list(features), MATCHER_FEATURE_NAMES)
            self.assertEqual(len(features), len(MATCHER_FEATURE_NAMES))
            self.assertEqual(features["fallback_used"], 0.0)
            self.assertGreater(features["log1p_match_count"], 0.0)
            self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_missing_matcher_packet_returns_zero_fallback(self):
        features = extract_matcher_features("/missing/file.npz")

        self.assertEqual(features["fallback_used"], 1.0)
        self.assertEqual(features["log1p_match_count"], 0.0)
        self.assertEqual(len(features), len(MATCHER_FEATURE_NAMES))

    def test_train_derived_normalization_is_reusable(self):
        rows = [
            {"raw_features": {name: float(i + 1) for i, name in enumerate(MATCHER_FEATURE_NAMES)}},
            {"raw_features": {name: float(i + 2) for i, name in enumerate(MATCHER_FEATURE_NAMES)}},
        ]

        stats = compute_normalization(rows)
        normalized = apply_normalization(rows[0]["raw_features"], stats)

        self.assertEqual(stats["feature_names"], MATCHER_FEATURE_NAMES)
        self.assertEqual(len(normalized), len(MATCHER_FEATURE_NAMES))
        self.assertTrue(all(np.isfinite(value) for value in normalized))

    def test_manifest_loader_and_sample_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "features.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "0001/01_02",
                        "features": [0.1] * len(MATCHER_FEATURE_NAMES),
                        "fallback_used": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_feature_manifest(manifest)
            features, mask = feature_tensor_for_sample("0001/01_02", loaded)
            missing, missing_mask = feature_tensor_for_sample("0001/03_04", loaded)

            self.assertEqual(len(features), len(MATCHER_FEATURE_NAMES))
            self.assertEqual(mask, 1.0)
            self.assertEqual(missing, [0.0] * len(MATCHER_FEATURE_NAMES))
            self.assertEqual(missing_mask, 0.0)

    def test_pairuav_dataset_attaches_matcher_feature_payload_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir = root / "json" / "0001"
            json_dir.mkdir(parents=True)
            (json_dir / "01_02.json").write_text(
                json.dumps(
                    {
                        "image_a": "0001/image-01.jpeg",
                        "image_b": "0001/image-02.jpeg",
                        "heading_num": 0.0,
                        "range_num": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            feature_manifest = root / "features.jsonl"
            feature_manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "0001/01_02",
                        "features": [0.25] * len(MATCHER_FEATURE_NAMES),
                        "fallback_used": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = PairUAV(
                json_root=json_dir.parent,
                image_root=root / "images",
                split="dev",
                resolution=(512, 384),
                seed=7,
                matcher_feature_manifest=feature_manifest,
            )

            payload = dataset._matcher_feature_payload(dataset.samples[0])
            self.assertEqual(payload["sample_id"], "0001/01_02")
            self.assertEqual(payload["matcher_features"].shape, (len(MATCHER_FEATURE_NAMES),))
            self.assertEqual(float(payload["matcher_feature_mask"]), 1.0)

    def test_pairuav_dataset_default_payload_is_zero_mask_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir = root / "json" / "0001"
            json_dir.mkdir(parents=True)
            (json_dir / "01_02.json").write_text(
                json.dumps(
                    {
                        "image_a": "0001/image-01.jpeg",
                        "image_b": "0001/image-02.jpeg",
                        "heading_num": 0.0,
                        "range_num": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            dataset = PairUAV(
                json_root=json_dir.parent,
                image_root=root / "images",
                split="dev",
                resolution=(512, 384),
                seed=7,
            )

            payload = dataset._matcher_feature_payload(dataset.samples[0])
            self.assertEqual(payload["matcher_features"].tolist(), [0.0] * len(MATCHER_FEATURE_NAMES))
            self.assertEqual(float(payload["matcher_feature_mask"]), 0.0)


if __name__ == "__main__":
    unittest.main()
