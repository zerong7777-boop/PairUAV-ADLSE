import unittest

from scripts.phase27_a_v3_2c_outcome_consistency_audit import (
    build_shared_surface,
    choose_state,
    compute_variant_summary,
)


class OutcomeConsistencyAuditTests(unittest.TestCase):
    def test_choose_state_priority_and_fallback(self):
        self.assertEqual(choose_state({"reacquired_state": "evidence_base_regime:control", "a_state": "hard"}), "evidence_base_regime:control")
        self.assertEqual(choose_state({"a_state": "hard", "state": "easy"}), "hard")
        self.assertEqual(choose_state({"state": "stable"}), "stable")
        self.assertEqual(choose_state({"candidate_state": "ambiguous"}), "ambiguous")
        self.assertEqual(choose_state({}), "unknown_unlabeled")

    def test_build_shared_surface_requires_exact_identity(self):
        manifest = [
            {"canonical_pair_id": "g/01_02", "a_state": "hard", "target_key": "g"},
            {"canonical_pair_id": "g/01_03", "a_state": "stable", "target_key": "g"},
        ]
        baseline = [
            {
                "canonical_pair_id": "g/01_02",
                "variant_id": "baseline",
                "row_status": "ok",
                "joint_error": "10",
                "heading_abs_error": "6",
                "range_abs_error": "8",
                "prediction_heading": "1",
            },
            {
                "canonical_pair_id": "g/01_03",
                "variant_id": "baseline",
                "row_status": "ok",
                "joint_error": "2",
                "heading_abs_error": "1",
                "range_abs_error": "1",
                "prediction_heading": "1",
            },
        ]
        stress = {
            "stress_a": [
                {
                    "canonical_pair_id": "g/01_02",
                    "variant_id": "stress_a",
                    "row_status": "ok",
                    "joint_error": "16",
                    "heading_abs_error": "12",
                    "range_abs_error": "9",
                    "prediction_heading": "1",
                },
                {
                    "canonical_pair_id": "g/01_03",
                    "variant_id": "stress_a",
                    "row_status": "ok",
                    "joint_error": "2.5",
                    "heading_abs_error": "1.2",
                    "range_abs_error": "1.1",
                    "prediction_heading": "1",
                },
            ]
        }
        shared, issues = build_shared_surface(manifest, baseline, stress)
        self.assertEqual(len(shared), 2)
        self.assertEqual(issues["missing_baseline"], 0)
        self.assertEqual(issues["missing_stress_rows"], 0)
        self.assertEqual(shared[0]["state"], "hard")
        self.assertEqual(shared[0]["shared_outcome"], "1")
        self.assertAlmostEqual(float(shared[0]["stress_a_delta"]), 6.0)

    def test_compute_variant_summary_counts_prediction_success(self):
        rows = [
            {"variant_id": "baseline", "row_status": "ok", "prediction_heading": "1"},
            {"variant_id": "baseline", "row_status": "model_error", "prediction_heading": ""},
        ]
        summary = compute_variant_summary(rows)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["prediction_success_count"], 1)
        self.assertEqual(summary["status_counts"]["ok"], 1)


if __name__ == "__main__":
    unittest.main()
