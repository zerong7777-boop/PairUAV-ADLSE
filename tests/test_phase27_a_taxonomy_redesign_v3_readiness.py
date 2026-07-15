import unittest

from scripts import phase27_a_taxonomy_redesign_v3_readiness as readiness


class ReadinessRulesTest(unittest.TestCase):
    def test_conflict_heading_hard_gives_correspondence_diagnostic(self):
        row = {
            "semantic_geometric_conflict_candidate": True,
            "baseline_heading_hard": True,
            "low_observable_candidate": False,
            "evidence_sufficient_candidate": True,
            "validation_status": "validated_ready",
        }
        out = readiness.derive_readiness_verdicts(row)
        self.assertTrue(out["READY_CORRESPONDENCE_DIAGNOSTIC"])
        self.assertTrue(out["READY_HEADING_HARD_TRAINING"])
        self.assertTrue(out["ANALYSIS_ONLY"])

    def test_control_preservation(self):
        row = {
            "ordinary_candidate": True,
            "control_candidate": True,
            "baseline_heading_hard": False,
            "baseline_range_hard": False,
            "stress_joint_sensitive": False,
            "low_observable_candidate": False,
        }
        out = readiness.derive_readiness_verdicts(row)
        self.assertTrue(out["READY_CONTROL_PRESERVATION"])

    def test_low_observable_quarantine(self):
        out = readiness.derive_readiness_verdicts({"low_observable_candidate": True})
        self.assertTrue(out["QUARANTINE_LOW_OBSERVABLE"])
        self.assertTrue(out["NOT_READY"])

    def test_multi_modal_is_multi_hypothesis(self):
        out = readiness.derive_readiness_verdicts({"multi_modal_ambiguous": True})
        self.assertTrue(out["READY_MULTI_HYPOTHESIS"])
        self.assertTrue(out["ANALYSIS_ONLY"])

    def test_missing_validation_not_training_ready(self):
        out = readiness.derive_readiness_verdicts({
            "evidence_sufficient_candidate": True,
            "baseline_heading_hard": True,
            "validation_status": "unknown_due_to_missing_join",
        })
        self.assertFalse(out["READY_HEADING_HARD_TRAINING"])
        self.assertTrue(out["ANALYSIS_ONLY"])


if __name__ == "__main__":
    unittest.main()
