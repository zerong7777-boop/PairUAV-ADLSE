import unittest

from scripts.phase27_a_v3_2a_duplicate_missing_exports import build_disjoint_universe_summary, build_duplicate_blocked_pairs, build_missing_key_rows


class DuplicateMissingExportsTests(unittest.TestCase):
    def test_duplicate_missing_and_disjoint_exports(self):
        artifacts = {
            "a": [{"canonical_pair_id": "x", "source_image_key": "s"}, {"canonical_pair_id": "x", "source_image_key": "s2"}, {"canonical_pair_id": ""}],
        }
        dup = build_duplicate_blocked_pairs(artifacts)
        self.assertTrue(any(row["join_key"] == "x" for row in dup))
        self.assertTrue(all(row["promotion_allowed"] == "false" for row in dup))
        missing = build_missing_key_rows(artifacts)
        self.assertTrue(any(row["key_strategy"] == "canonical_pair_id" for row in missing))
        disjoint = build_disjoint_universe_summary([{"left_artifact": "a", "right_artifact": "b", "key_strategy": "canonical_pair_id", "left_only_count": 1, "right_only_count": 2, "intersection_count": 0, "promotion_eligible": "false", "reason_codes": "zero_intersection"}])
        self.assertEqual(disjoint[0]["intersection_count"], 0)


if __name__ == "__main__":
    unittest.main()
