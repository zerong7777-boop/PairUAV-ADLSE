import unittest

from scripts.phase27_a_v3_hard_ambiguity_decomposition import compute_hard_ambiguity_decomposition


class HardAmbiguityDecompositionTests(unittest.TestCase):
    def test_expected_intersections_and_zero_rows_present(self):
        rows = [
            {
                "evidence_sufficient_heading_hard": "true",
                "evidence_sufficient_range_hard": "true",
                "evidence_sufficient_joint_hard": "true",
                "multi_modal_ambiguous": "true",
                "semantic_geometric_conflict": "true",
                "stress_sensitive_ambiguous": "true",
                "tail_error_unreliable": "true",
            }
        ]
        table = {row["subtype"]: row for row in compute_hard_ambiguity_decomposition(rows)}
        self.assertEqual(table["heading_hard_multi_modal"]["count"], 1)
        self.assertEqual(table["heading_hard_semantic_conflict"]["count"], 1)
        self.assertEqual(table["heading_hard_stress_ambiguous"]["count"], 1)
        self.assertEqual(table["range_hard_multi_modal"]["count"], 1)
        self.assertEqual(table["joint_hard_tail_unreliable"]["count"], 1)
        self.assertIn("range_hard_stress_ambiguous", table)

    def test_diagnostic_only_always_true(self):
        table = compute_hard_ambiguity_decomposition([{}])
        self.assertTrue(all(row["diagnostic_only"] == "true" for row in table))


if __name__ == "__main__":
    unittest.main()
