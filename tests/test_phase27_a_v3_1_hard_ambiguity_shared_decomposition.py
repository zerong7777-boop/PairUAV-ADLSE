import unittest

from scripts.phase27_a_v3_1_hard_ambiguity_shared_decomposition import compute_hard_ambiguity_shared_decomposition


class HardAmbiguitySharedDecompositionTests(unittest.TestCase):
    def test_shared_counts_and_diagnostic_only(self):
        rows = [
            {"shared_baseline_stress_joined": "true", "baseline_heading_hard": "true", "baseline_range_hard": "true", "baseline_joint_hard": "true", "multi_modal_ambiguous": "true", "tail_error_unreliable": "true", "semantic_geometric_conflict": "true", "stress_sensitive_ambiguous": "true"},
            {"shared_baseline_stress_joined": "false", "baseline_heading_hard": "true", "multi_modal_ambiguous": "true"},
        ]
        table = {row["subtype"]: row for row in compute_hard_ambiguity_shared_decomposition(rows)}
        self.assertEqual(table["heading_hard_multi_modal"]["shared_count"], 1)
        self.assertEqual(table["range_hard_multi_modal"]["shared_count"], 1)
        self.assertEqual(table["joint_hard_tail_unreliable"]["shared_count"], 1)
        self.assertEqual(table["heading_hard_semantic_conflict"]["shared_count"], 1)
        self.assertEqual(table["joint_hard_stress_ambiguous"]["diagnostic_only"], "true")


if __name__ == "__main__":
    unittest.main()
