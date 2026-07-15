import unittest

from scripts.phase27_a_v3_2a_repair_candidates import infer_repair_candidates


class RepairCandidateTests(unittest.TestCase):
    def test_repair_classes(self):
        pairwise = [
            {"key_strategy": "source_target_pair_composite", "intersection_count": 1},
            {"key_strategy": "path_normalized_source_target_pair", "intersection_count": 3},
            {"key_strategy": "direction_invariant_source_target_pair", "intersection_count": 4},
            {"key_strategy": "canonical_pair_id", "intersection_count": 0},
            {"key_strategy": "row_index_diagnostic_only", "intersection_count": 5},
        ]
        profile = [{"key_strategy": "canonical_pair_id", "duplicate_key_count": 2}]
        classes = {row["repair_class"] for row in infer_repair_candidates(profile, pairwise)}
        self.assertIn("path_normalization_candidate", classes)
        self.assertIn("direction_normalization_candidate", classes)
        self.assertIn("unrepairable_identity_conflict", classes)

    def test_manifest_reacquisition_when_no_overlap(self):
        classes = {row["repair_class"] for row in infer_repair_candidates([], [{"key_strategy": "canonical_pair_id", "intersection_count": 0}])}
        self.assertIn("manifest_reacquisition_required", classes)


if __name__ == "__main__":
    unittest.main()
