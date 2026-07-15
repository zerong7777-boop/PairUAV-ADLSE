import unittest

from scripts.phase27_a_v3_1_shared_outcome_consistency import compute_shared_outcome_consistency


class SharedOutcomeConsistencyTests(unittest.TestCase):
    def test_shared_only_counts_and_overlap(self):
        rows = [
            {"shared_baseline_stress_joined": "true", "baseline_heading_hard": "true", "baseline_range_hard": "false", "baseline_joint_hard": "true", "stress_main_heading_sensitive": "true", "stress_main_range_sensitive": "false", "stress_main_joint_sensitive": "true", "target_key": "t", "group_id": "g"},
            {"shared_baseline_stress_joined": "false", "baseline_joint_hard": "true", "stress_main_joint_sensitive": "true"},
        ]
        metrics = compute_shared_outcome_consistency(rows, ["main"])
        self.assertEqual(metrics["shared_rows"], 1)
        self.assertEqual(metrics["baseline_heading_hard_count"], 1)
        self.assertEqual(metrics["baseline_joint_hard_stress_main_joint_overlap_count"], 1)
        self.assertEqual(metrics["target_distribution_on_shared"]["t"], 1)


if __name__ == "__main__":
    unittest.main()
