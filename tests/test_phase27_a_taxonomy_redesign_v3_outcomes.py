import unittest

from scripts import phase27_a_taxonomy_redesign_v3_outcomes as outcomes


class OutcomeRulesTest(unittest.TestCase):
    def test_heading_and_range_hard_are_independent(self):
        row = {
            "evidence_sufficient_candidate": True,
            "heading_error_score": "0.8",
            "range_error_score": "0.2",
            "full_dev_joined": "1",
            "stress_joined": "1",
        }
        out = outcomes.derive_layer2_outcomes(row)
        self.assertTrue(out["baseline_heading_hard"])
        self.assertFalse(out["baseline_range_hard"])
        self.assertTrue(out["baseline_joint_hard"])

    def test_stress_sensitivity_split(self):
        row = {"stress_sensitivity_score": "0.8", "heading_error_score": "0.2", "range_error_score": "0.9"}
        out = outcomes.derive_stress_sensitivity(row)
        self.assertFalse(out["heading_stress_sensitive"])
        self.assertTrue(out["range_stress_sensitive"])

    def test_unjoined_validation_status(self):
        out = outcomes.derive_layer2_outcomes({"full_dev_joined": "0", "stress_joined": "0"})
        self.assertEqual(out["validation_status"], "unknown_due_to_missing_join")


if __name__ == "__main__":
    unittest.main()
