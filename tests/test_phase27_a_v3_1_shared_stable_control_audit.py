import unittest

from scripts.phase27_a_v3_1_shared_stable_control_audit import compute_shared_stable_control_metrics, select_shared_control_rows


class SharedStableControlAuditTests(unittest.TestCase):
    def test_shared_controls_and_verdicts(self):
        rows = [
            {"shared_baseline_stress_joined": "true", "control_candidate": "true", "baseline_joint_error_score": "0.1", "baseline_angle_abs_error": "0.1", "baseline_distance_rel_error": "0.1", "stress_main_joint_delta": "0.1"},
            {"shared_baseline_stress_joined": "false", "control_candidate": "true", "stress_main_joint_delta": "9"},
        ]
        self.assertEqual(len(select_shared_control_rows(rows)), 1)
        metrics = compute_shared_stable_control_metrics(rows, ["main"])
        self.assertEqual(metrics["verdict"], "control-anchor-shadow-candidate")

    def test_no_shared_controls_not_validated_and_high_stress_analysis_only(self):
        self.assertEqual(compute_shared_stable_control_metrics([], ["main"])["verdict"], "control-anchor-not-validated")
        rows = [{"shared_baseline_stress_joined": "true", "control_candidate": "true", "baseline_joint_error_score": "0.1", "stress_main_joint_delta": "0.9"}]
        self.assertEqual(compute_shared_stable_control_metrics(rows, ["main"])["verdict"], "control-anchor-analysis-only")


if __name__ == "__main__":
    unittest.main()
