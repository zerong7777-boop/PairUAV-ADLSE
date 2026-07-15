import unittest

from scripts.phase27_a_validation_spine_suites import (
    control_stability_suite,
    identity_suite,
    leakage_suite,
    state_distribution_suite,
    state_error_association_suite,
    training_readiness_suite,
)


class Phase27AValidationSpineSuitesTest(unittest.TestCase):
    def test_identity_allows_unresolved_too_small_reference(self):
        result = identity_suite(
            evidence_to_baseline={
                "canonical_overlap": 100,
                "duplicates": 0,
                "collisions": 0,
            },
            evidence_to_reference={
                "canonical_overlap": 0,
                "failure_classification": "reference_too_small",
            },
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reference_status"], "reference_unresolved")

    def test_leakage_rejects_final_score_in_deployable_evidence(self):
        result = leakage_suite(
            deployable_evidence={
                "pair_id": "ordinary-001",
                "final_score": 0.97,
            }
        )

        self.assertFalse(result["passed"])
        self.assertIn("final_score", result["forbidden_deployable_fields"])

    def test_state_distribution_rejects_unknown_dominant_distribution(self):
        result = state_distribution_suite(
            {
                "ordinary": 0.02,
                "hard": 0.01,
                "unknown": 0.90,
            }
        )

        self.assertFalse(result["passed"])

    def test_state_error_association_accepts_harder_states_with_more_error(self):
        result = state_error_association_suite(
            {
                "ordinary": 0.20,
                "hard": 0.70,
                "ambiguous": 0.85,
            }
        )

        self.assertTrue(result["passed"])
        self.assertGreater(result["hard_minus_control_delta"], 0)

    def test_control_stability_accepts_separated_controls(self):
        result = control_stability_suite(
            {
                "ordinary": 0.20,
                "hard": 0.80,
            }
        )

        self.assertTrue(result["passed"])

    def test_training_readiness_allows_shadow_valid_unresolved_reference(self):
        identity = {
            "passed": True,
            "reference_status": "reference_unresolved",
        }
        passed_suite = {"passed": True}

        result = training_readiness_suite(
            identity=identity,
            lineage=passed_suite,
            leakage=passed_suite,
            distribution=passed_suite,
            state_error=passed_suite,
            control=passed_suite,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["verdict"],
            "A-validation-spine-reference-unresolved-but-shadow-valid",
        )


if __name__ == "__main__":
    unittest.main()
