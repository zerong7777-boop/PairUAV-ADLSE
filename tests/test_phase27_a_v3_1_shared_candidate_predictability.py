import unittest

from scripts.phase27_a_v3_1_shared_candidate_predictability import auc_rank, compute_shared_candidate_predictability, flag_wide_precision_recall, precision_recall_at_k_tie_aware


class SharedCandidatePredictabilityTests(unittest.TestCase):
    def test_shared_only_auc_flag_and_tie_aware(self):
        rows = [
            {"shared_baseline_stress_joined": "true", "ambiguity_candidate": "true", "stress_main_joint_sensitive": "true", "target_key": "t", "group_id": "g"},
            {"shared_baseline_stress_joined": "true", "ambiguity_candidate": "false", "stress_main_joint_sensitive": "false", "target_key": "t", "group_id": "g"},
            {"shared_baseline_stress_joined": "false", "ambiguity_candidate": "true", "stress_main_joint_sensitive": "false"},
        ]
        self.assertEqual(auc_rank(rows, "ambiguity_candidate", "stress_main_joint_sensitive"), 1.0)
        self.assertEqual(flag_wide_precision_recall(rows, "ambiguity_candidate", "stress_main_joint_sensitive")["precision"], 1.0)
        self.assertEqual(precision_recall_at_k_tie_aware(rows, "ambiguity_candidate", "stress_main_joint_sensitive", 1)["precision"], 1.0)
        metrics = compute_shared_candidate_predictability(rows, ["main"])
        self.assertTrue(metrics["by_target_group"])

    def test_tied_scores_auc_half_and_missing_ok(self):
        rows = [
            {"shared_baseline_stress_joined": "true", "ambiguity_candidate": "true", "stress_main_joint_sensitive": "true"},
            {"shared_baseline_stress_joined": "true", "ambiguity_candidate": "true", "stress_main_joint_sensitive": "false"},
            {"shared_baseline_stress_joined": "true", "ambiguity_candidate": "", "stress_main_joint_sensitive": "true"},
        ]
        self.assertEqual(auc_rank(rows, "ambiguity_candidate", "stress_main_joint_sensitive"), 0.5)


if __name__ == "__main__":
    unittest.main()
