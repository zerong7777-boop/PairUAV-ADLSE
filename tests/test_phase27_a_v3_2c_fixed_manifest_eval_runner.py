import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_2c_fixed_manifest_eval_runner import (
    angle_abs_error,
    joint_error,
    manifest_row_to_sample,
    output_row_from_manifest,
    relative_error,
    run_dry_identity,
)


class FixedManifestEvalRunnerTests(unittest.TestCase):
    def test_dry_run_preserves_identity_and_status(self):
        rows = [
            {
                "manifest_version": "v",
                "manifest_hash": "h",
                "canonical_pair_id": "a",
                "source_image_key": "s",
                "target_image_key": "t",
                "gt_heading": "1",
                "gt_range": "2",
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.json"
            config.write_text("{}", encoding="utf-8")
            out = run_dry_identity(rows, "baseline", str(config), "")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["canonical_pair_id"], "a")
        self.assertEqual(out[0]["variant_id"], "baseline")
        self.assertEqual(out[0]["row_status"], "ok")
        self.assertTrue(out[0]["eval_config_hash"])

    def test_metric_helpers(self):
        self.assertEqual(angle_abs_error(179, -179), 2)
        self.assertEqual(relative_error(2, 4), 0.5)
        self.assertIsNone(relative_error(2, 0))
        self.assertAlmostEqual(joint_error(3, 4), 5)

    def test_manifest_row_to_sample_preserves_identity(self):
        sample = manifest_row_to_sample(
            {
                "canonical_pair_id": "0852/01_02",
                "source_image_path": "0852/image-01.jpeg",
                "target_image_path": "0852/image-02.jpeg",
                "gt_heading": "16",
                "gt_range": "-2",
                "group_id": "0852",
                "scene_key": "0852",
                "manifest_hash": "h",
                "manifest_row_id": "1",
                "source_image_key": "s",
                "target_image_key": "t",
            }
        )
        self.assertEqual(sample["canonical_pair_id"], "0852/01_02")
        self.assertEqual(sample["image_a"], "0852/image-01.jpeg")
        self.assertEqual(sample["heading_deg"], 16)

    def test_output_row_with_prediction_writes_metrics(self):
        row = {
            "canonical_pair_id": "a",
            "source_image_key": "s",
            "target_image_key": "t",
            "gt_heading": "10",
            "gt_range": "2",
        }
        out = output_row_from_manifest(row, "baseline", "", "", "ok", prediction_heading=13, prediction_range=5)
        self.assertEqual(out["row_status"], "ok")
        self.assertEqual(out["heading_abs_error"], "3")
        self.assertEqual(out["range_abs_error"], "3")


if __name__ == "__main__":
    unittest.main()
