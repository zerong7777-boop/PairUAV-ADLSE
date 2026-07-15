import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_evidence_state_v3_feature_schema import (
    audit_feature_columns_v3,
    compute_cheap_image_features,
    extract_identity_layout_features,
    extract_image_index,
    make_feature_layer_flags,
)


class Phase27V3FeatureSchemaTest(unittest.TestCase):
    def test_forbidden_fields_rejected(self):
        audit = audit_feature_columns_v3(
            [
                "group_id",
                "heading_num",
                "range_num",
                "gt_angle",
                "gt_distance",
                "final_score",
                "angle_err",
                "range_err",
                "combined_error",
                "residual",
                "official_metric",
                "leaderboard_score",
            ]
        )
        self.assertFalse(audit["passed"])
        self.assertIn("heading_num", audit["forbidden_columns"])
        self.assertIn("final_score", audit["forbidden_columns"])

    def test_identity_layout_parsing(self):
        features = extract_identity_layout_features(
            {
                "split": "train",
                "json_path": "/x/0839/01_33.json",
                "group_id": "0839",
                "json_id": "01_33",
                "image_a": "0839/image-01.jpeg",
                "image_b": "0839/image-33.jpeg",
                "image_a_name": "image-01.jpeg",
                "image_b_name": "image-33.jpeg",
                "pair_key": "image-01.jpeg|image-33.jpeg",
            }
        )
        self.assertEqual(features["pair_id"], "0839/01_33")
        self.assertEqual(features["image_a_index"], 1)
        self.assertEqual(features["image_b_index"], 33)
        self.assertEqual(features["image_index_gap_abs"], 32)
        self.assertEqual(features["has_identity_features"], 1)

    def test_extract_image_index(self):
        self.assertEqual(extract_image_index("image-02.jpeg"), 2)
        self.assertIsNone(extract_image_index("foo.jpeg"))

    def test_layer_flags(self):
        flags = make_feature_layer_flags(
            {
                "has_identity_features": 1,
                "has_cheap_image_features": 1,
                "has_cached_matcher_features": 0,
            }
        )
        self.assertEqual(flags["feature_layer_mask"], "identity+cheap")
        self.assertEqual(flags["feature_confidence_level"], "cheap")

    def test_cheap_image_features_synthetic(self):
        try:
            from PIL import Image
        except Exception:
            self.skipTest("PIL unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.png"
            path_b = Path(tmpdir) / "b.png"
            Image.new("RGB", (8, 8), (100, 100, 100)).save(path_a)
            Image.new("RGB", (8, 8), (120, 120, 120)).save(path_b)
            features = compute_cheap_image_features(path_a, path_b)
        self.assertEqual(features["image_a_exists"], 1)
        self.assertEqual(features["image_b_exists"], 1)
        self.assertEqual(features["has_cheap_image_features"], 1)
        self.assertIn("grayscale_hist_similarity", features)


if __name__ == "__main__":
    unittest.main()
