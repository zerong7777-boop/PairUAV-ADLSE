import unittest

from scripts.phase27_a_v3_2b_fixed_manifest import build_fixed_manifest


class FixedManifestTests(unittest.TestCase):
    def test_builds_clean_manifest_from_candidate_full_overlap(self):
        cand = [
            {"canonical_pair_id": "a", "source_image_key": "S.JPG", "target_image_key": "T.JPG", "target_key": "g", "group_id": "g", "scene_key": "g", "split_key": "train", "READY_CONTROL_PRESERVATION": "False"},
            {"canonical_pair_id": "b", "source_image_key": "S2.JPG", "target_image_key": "T2.JPG"},
            {"canonical_pair_id": "a", "source_image_key": "S3.JPG", "target_image_key": "T3.JPG"},
        ]
        full = [{"canonical_pair_id": "a"}]
        rows, metrics = build_fixed_manifest(cand, full, limit=100)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_pair_id"], "a")
        self.assertEqual(rows[0]["pair_direction"], "ordered_source_target")
        self.assertTrue(rows[0]["manifest_checksum"])
        self.assertEqual(metrics["unique_canonical_pair_id_count"], 1)


if __name__ == "__main__":
    unittest.main()

