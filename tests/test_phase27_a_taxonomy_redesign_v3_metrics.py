import tempfile
import unittest
from pathlib import Path

from scripts import phase27_a_taxonomy_redesign_v3_metrics as metrics


class MetricsTest(unittest.TestCase):
    def test_multilabel_overlap_preserves_hard_ambiguous(self):
        rows = [{
            "evidence_sufficient_heading_hard": True,
            "semantic_geometric_conflict": True,
            "baseline_joint_hard": True,
            "stress_joint_sensitive": True,
        }]
        report = metrics.compute_multilabel_overlap(
            rows,
            ["evidence_sufficient_heading_hard", "semantic_geometric_conflict"],
        )
        self.assertEqual(report["hard_ambiguity_overlap_count"], 1)
        self.assertEqual(
            report["pairwise_overlap"]["evidence_sufficient_heading_hard__AND__semantic_geometric_conflict"],
            1,
        )

    def test_heading_range_hard_split_is_separate(self):
        rows = [
            {"evidence_sufficient_heading_hard": True, "evidence_sufficient_range_hard": False},
            {"evidence_sufficient_heading_hard": False, "evidence_sufficient_range_hard": True},
            {"evidence_sufficient_heading_hard": True, "evidence_sufficient_range_hard": True},
        ]
        report = metrics.compute_heading_range_hard_split(rows)
        self.assertEqual(report["heading_only_hard"], 1)
        self.assertEqual(report["range_only_hard"], 1)
        self.assertEqual(report["heading_and_range_hard_overlap"], 1)

    def test_baseline_stress_zero_overlap_verdict(self):
        rows = [
            {"baseline_joint_hard": True, "stress_joint_sensitive": False},
            {"baseline_joint_hard": False, "stress_joint_sensitive": True},
        ]
        report = metrics.compute_baseline_stress_consistency_audit(rows)
        self.assertEqual(report["overlap_count"], 0)
        self.assertIn("zero-overlap", report["verdict"])

    def test_join_coverage_report(self):
        rows = [
            {"full_dev_join_status": "joined", "stress_join_status": "unjoined", "validation_status": "validated_ready"},
            {"full_dev_join_status": "unjoined", "stress_join_status": "joined", "validation_status": "candidate_only_unvalidated"},
        ]
        report = metrics.compute_join_coverage_bias_report(rows)
        self.assertEqual(report["full_dev_join_status"]["joined"], 1)
        self.assertEqual(report["stress_join_status"]["joined"], 1)

    def test_leakage_audit_passes_and_no_go_boundary_is_true(self):
        audit = metrics.compute_leakage_deployability_audit()
        self.assertTrue(audit["passed"])
        boundary = metrics.compute_no_go_training_policy_boundary([{}])
        self.assertFalse(boundary["training_policy_allowed"])

    def test_write_metrics_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"baseline_joint_hard": True, "stress_joint_sensitive": False}]
            bundle = metrics.write_metrics_bundle(rows, Path(td))
            self.assertIn("no_go_training_policy_boundary", bundle)
            self.assertTrue((Path(td) / "metrics" / "no_go_training_policy_boundary.json").exists())


if __name__ == "__main__":
    unittest.main()
