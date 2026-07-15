import unittest

from scripts.phase27_a_v3_b_diagnostic_slices import build_b_diagnostic_slices, write_training_policy_readiness_verdict


class BDiagnosticSlicesTests(unittest.TestCase):
    def test_required_slices_and_diagnostic_only(self):
        rows = [
            {
                "canonical_pair_id": "p1",
                "semantic_geometric_conflict": "true",
                "evidence_sufficient_heading_hard": "true",
                "multi_modal_ambiguous": "true",
                "stress_sensitive_ambiguous": "true",
                "control_candidate": "true",
                "stress_sensitivity_score": "0.1",
                "baseline_error_score": "0.1",
            }
        ]
        slices = build_b_diagnostic_slices(rows)
        names = {row["slice_name"] for row in slices}
        self.assertIn("semantic_geometric_conflict", names)
        self.assertIn("heading_hard_semantic_geometric_conflict", names)
        self.assertIn("heading_hard_multi_modal_ambiguous", names)
        self.assertIn("hard_ambiguity_overlap", names)
        self.assertIn("control_candidate_low_stress_low_error", names)
        self.assertTrue(all(row["diagnostic_only"] == "true" for row in slices))
        forbidden = {"gate_label", "train_label", "sampler_weight", "oversample", "loss_weight"}
        self.assertFalse(any(forbidden.intersection(row.keys()) for row in slices))

    def test_readiness_verdict_defaults_no_go_when_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            verdict = write_training_policy_readiness_verdict(tmpdir, {})
        self.assertEqual(verdict["verdict"], "training-policy-no-go")

    def test_readiness_verdict_blocks_outcome(self):
        import tempfile

        metrics = {
            "outcome": {"verdict": "unresolved-blocker"},
            "join_bias": {"verdict": "join-bias-acceptable-for-analysis"},
            "predictability": {"useful_pair_count": 1},
            "stable_control": {"verdict": "control-preservation-safe"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            verdict = write_training_policy_readiness_verdict(tmpdir, metrics)
        self.assertEqual(verdict["verdict"], "training-policy-blocked-by-outcome-surface")


if __name__ == "__main__":
    unittest.main()
