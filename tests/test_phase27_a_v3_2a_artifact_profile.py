import unittest

from scripts.phase27_a_v3_2a_artifact_profile import profile_artifact_rows


class ArtifactProfileTests(unittest.TestCase):
    def test_profile_counts_duplicates_and_missing(self):
        rows = [
            {"canonical_pair_id": "a", "target_key": "t1", "group_id": "g1", "scene_key": "s1"},
            {"canonical_pair_id": "a", "target_key": "t1", "group_id": "g1", "scene_key": "s1"},
            {"canonical_pair_id": "", "target_key": "t2"},
        ]
        profiles = {row["key_strategy"]: row for row in profile_artifact_rows("x", rows)}
        canonical = profiles["canonical_pair_id"]
        self.assertEqual(canonical["row_count"], 3)
        self.assertEqual(canonical["non_empty_key_count"], 2)
        self.assertEqual(canonical["unique_key_count"], 1)
        self.assertEqual(canonical["duplicate_key_count"], 1)
        self.assertEqual(canonical["missing_key_count"], 1)
        self.assertIn("a:2", canonical["top_duplicate_keys"])


if __name__ == "__main__":
    unittest.main()
