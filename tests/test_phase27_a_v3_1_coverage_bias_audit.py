import unittest

from scripts.phase27_a_v3_1_coverage_bias_audit import compute_coverage_metrics, compute_join_bias_by_target_group


class CoverageBiasAuditTests(unittest.TestCase):
    def test_coverage_counts_and_zero_verdict(self):
        rows = [
            {"baseline_joined": "true", "stress_main_joined": "false", "shared_baseline_stress_joined": "false", "shared_join_status": "missing_stress", "target_key": "t1", "group_id": "g1"},
            {"baseline_joined": "false", "stress_main_joined": "true", "shared_baseline_stress_joined": "false", "shared_join_status": "missing_baseline", "target_key": "t2", "group_id": "g1"},
            {"baseline_joined": "false", "stress_main_joined": "false", "shared_baseline_stress_joined": "false", "shared_join_status": "missing_both", "target_key": "t2", "group_id": "g2"},
        ]
        metrics = compute_coverage_metrics(rows, ["main"])
        self.assertEqual(metrics["baseline_joined_count"], 1)
        self.assertEqual(metrics["stress_joined_count_by_variant"]["main"], 1)
        self.assertEqual(metrics["shared_joined_count"], 0)
        self.assertEqual(metrics["verdict"], "shared-surface-blocked-zero-coverage")

    def test_analysis_only_when_low_nonzero_coverage(self):
        rows = [{"shared_baseline_stress_joined": "true", "shared_join_status": "joined"}] + [{"shared_baseline_stress_joined": "false", "shared_join_status": "missing_both"}] * 9
        self.assertEqual(compute_coverage_metrics(rows, ["main"])["verdict"], "shared-surface-analysis-only")

    def test_join_bias_table(self):
        rows = [{"target_key": "t", "group_id": "g", "shared_baseline_stress_joined": "true", "baseline_joined": "true", "stress_main_joined": "true"}]
        table = compute_join_bias_by_target_group(rows)
        self.assertEqual(table[0]["shared_coverage_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
