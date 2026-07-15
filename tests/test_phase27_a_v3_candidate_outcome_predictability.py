import unittest

from scripts.phase27_a_v3_candidate_outcome_predictability import (
    compute_candidate_to_outcome_predictability,
    compute_predictability_for_pair,
)


class CandidateOutcomePredictabilityTests(unittest.TestCase):
    def test_precision_recall_and_auc_perfect(self):
        rows = [
            {"score": "0.9", "outcome": "true"},
            {"score": "0.8", "outcome": "true"},
            {"score": "0.1", "outcome": "false"},
        ]
        metrics = compute_predictability_for_pair(rows, "score", "outcome")
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["precision_at_50"], 2 / 3)
        self.assertEqual(metrics["recall_at_50"], 1.0)

    def test_auc_tied_scores(self):
        rows = [{"score": "1", "outcome": "true"}, {"score": "1", "outcome": "false"}]
        self.assertEqual(compute_predictability_for_pair(rows, "score", "outcome")["auc"], 0.5)

    def test_top_decile_lift_above_one(self):
        rows = []
        for i in range(100):
            rows.append(
                {
                    "evidence_sufficient_candidate": "true" if i < 10 else "false",
                    "baseline_joint_hard": "true" if i < 10 else "false",
                }
            )
        metrics = compute_candidate_to_outcome_predictability(rows)
        best = metrics["best_pairs"][0]
        self.assertGreater(best["top_decile_lift"], 1.0)
        self.assertTrue(metrics["deciles"])

    def test_missing_scores_do_not_crash(self):
        rows = [{"evidence_sufficient_candidate": "", "baseline_joint_hard": "true"}]
        metrics = compute_candidate_to_outcome_predictability(rows)
        self.assertEqual(metrics["pair_metrics"][0]["valid_score_count"], 0)


if __name__ == "__main__":
    unittest.main()
