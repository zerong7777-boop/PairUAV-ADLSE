import unittest

from scripts.phase27_a_v3_outcome_consistency_audit import (
    audit_outcome_surface_consistency,
    compute_heading_range_consistency,
    compute_overlap_counts,
    compute_shared_join_mask_counts,
)


class OutcomeConsistencyAuditTests(unittest.TestCase):
    def test_overlap_counts(self):
        rows = [
            {"baseline_joint_hard": "true", "stress_joint_sensitive": "true"},
            {"baseline_joint_hard": "true", "stress_joint_sensitive": "false"},
            {"baseline_joint_hard": "false", "stress_joint_sensitive": "true"},
        ]
        metrics = compute_overlap_counts(rows)
        self.assertEqual(metrics["baseline_joint_hard_count"], 2)
        self.assertEqual(metrics["stress_joint_sensitive_count"], 2)
        self.assertEqual(metrics["baseline_stress_overlap_count"], 1)

    def test_zero_overlap_verdict_unresolved_when_shared_join_available(self):
        rows = [
            {
                "baseline_joint_hard": "true",
                "stress_joint_sensitive": "false",
                "full_dev_join_status": "joined",
                "stress_join_status": "joined",
            },
            {
                "baseline_joint_hard": "false",
                "stress_joint_sensitive": "true",
                "full_dev_join_status": "joined",
                "stress_join_status": "joined",
            },
        ] * 60
        metrics = audit_outcome_surface_consistency(rows)
        self.assertEqual(metrics["verdict"], "unresolved-blocker")
        self.assertIn("zero_overlap_persists_on_shared_join_mask", metrics["verdict_reasons"])

    def test_shared_join_mask_overlap_separate(self):
        rows = [
            {
                "baseline_joint_hard": "true",
                "stress_joint_sensitive": "true",
                "full_dev_join_status": "joined",
                "stress_join_status": "joined",
            },
            {
                "baseline_joint_hard": "true",
                "stress_joint_sensitive": "true",
                "full_dev_join_status": "missing",
                "stress_join_status": "joined",
            },
        ]
        self.assertEqual(compute_shared_join_mask_counts(rows)["shared_join_overlap_count"], 1)

    def test_heading_range_split_counts(self):
        rows = [
            {
                "baseline_heading_hard": "true",
                "baseline_range_hard": "false",
                "stress_heading_sensitive": "true",
                "stress_range_sensitive": "false",
            }
        ]
        metrics = compute_heading_range_consistency(rows)
        self.assertEqual(metrics["heading_hard_count"], 1)
        self.assertEqual(metrics["range_hard_count"], 0)

    def test_target_group_distribution_included(self):
        rows = [{"target_key": "t1", "group_id": "g1"}, {"target_key": "t1", "group_id": "g2"}]
        metrics = audit_outcome_surface_consistency(rows)
        self.assertEqual(metrics["target_distribution_if_available"]["target_key"]["t1"], 2)


if __name__ == "__main__":
    unittest.main()
