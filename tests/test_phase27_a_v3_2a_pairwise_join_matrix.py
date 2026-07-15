import unittest

from scripts.phase27_a_v3_2a_pairwise_join_matrix import compute_pairwise_join_for_strategy


class PairwiseJoinMatrixTests(unittest.TestCase):
    def test_join_matrix_counts_and_promotion(self):
        left = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}, {"canonical_pair_id": "dup"}, {"canonical_pair_id": "dup"}]
        right = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "c"}, {"canonical_pair_id": "dup"}]
        row = compute_pairwise_join_for_strategy("l", left, "r", right, "canonical_pair_id")
        self.assertEqual(row["intersection_count"], 2)
        self.assertEqual(row["one_to_one_count"], 1)
        self.assertEqual(row["many_to_one_count"], 1)
        self.assertEqual(row["duplicate_blocked_count"], 1)
        self.assertEqual(row["promotion_eligible"], "false")

    def test_row_index_is_never_promotion_eligible(self):
        left = [{"source_row_index": "1"}]
        right = [{"source_row_index": "1"}]
        row = compute_pairwise_join_for_strategy("l", left, "r", right, "row_index_diagnostic_only")
        self.assertEqual(row["intersection_count"], 1)
        self.assertEqual(row["promotion_eligible"], "false")


if __name__ == "__main__":
    unittest.main()
