import unittest

from scripts.phase27_a_v3_stable_control_join_bias_audit import (
    compute_candidate_distribution_by_join,
    compute_join_bias_extension_report,
    compute_stable_control_stress_audit,
    select_control_rows,
)


class StableControlJoinBiasTests(unittest.TestCase):
    def test_control_selected_by_candidate_or_readiness(self):
        rows = [{"control_candidate": "true"}, {"READY_CONTROL_PRESERVATION": "true"}, {}]
        self.assertEqual(len(select_control_rows(rows)), 2)

    def test_high_stress_control_no_go(self):
        rows = [
            {
                "control_candidate": "true",
                "stress_sensitivity_score": "0.9",
                "baseline_error_score": "0.1",
                "heading_error_score": "0.1",
                "range_error_score": "0.1",
            }
        ]
        metrics = compute_stable_control_stress_audit(rows)
        self.assertEqual(metrics["verdict"], "control-preservation-no-go")
        self.assertIn("p95", metrics["stress_delta_summary"])

    def test_tail_conflict_ambiguity_rates(self):
        rows = [
            {
                "control_candidate": "true",
                "tail_error_high": "true",
                "semantic_geometric_conflict": "true",
                "ambiguity_candidate": "true",
            }
        ]
        metrics = compute_stable_control_stress_audit(rows)
        self.assertEqual(metrics["tail_error_rate"], 1.0)
        self.assertEqual(metrics["conflict_contamination_rate"], 1.0)
        self.assertEqual(metrics["ambiguity_contamination_rate"], 1.0)

    def test_join_distributions_and_blocking_verdict(self):
        rows = [
            {"full_dev_join_status": "joined", "stress_join_status": "missing", "control_candidate": "true"},
            {"full_dev_join_status": "missing", "stress_join_status": "missing", "control_candidate": "false"},
        ]
        dist = compute_candidate_distribution_by_join(rows)
        self.assertIn("joined", dist["full_dev"])
        metrics = compute_join_bias_extension_report(rows)
        self.assertEqual(metrics["shared_join_mask_count"], 0)
        self.assertEqual(metrics["verdict"], "join-bias-blocks-training-policy")


if __name__ == "__main__":
    unittest.main()
