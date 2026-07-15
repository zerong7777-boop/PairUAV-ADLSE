import unittest

from scripts.phase27_a_validation_spine_manifest import (
    classify_columns,
    join_rows_by_key,
    make_shadow_rows,
)
from scripts.phase27_a_validation_spine_slices import (
    compute_slice_metrics,
    row_in_slice,
)


class Phase27AValidationSpineManifestSliceTests(unittest.TestCase):
    def test_classify_columns_separates_deployable_evidence_from_analysis_only(self):
        result = classify_columns(
            [
                "canonical_pair_id",
                "evidence_state",
                "final_score",
                "angle_err",
                "observability_score",
            ]
        )

        self.assertIn("observability_score", result["deployable_evidence_columns"])
        self.assertIn("evidence_state", result["deployable_evidence_columns"])
        self.assertIn("final_score", result["analysis_only_columns"])
        self.assertIn("angle_err", result["analysis_only_columns"])
        self.assertFalse(result["leakage_passed"])

    def test_join_rows_by_key_reports_unmatched_left_and_right_counts(self):
        result = join_rows_by_key(
            [{"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}],
            [{"canonical_pair_id": "a", "baseline_final_score": "0.4"}],
        )

        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["canonical_pair_id"], "a")
        self.assertEqual(result["unmatched_left_count"], 1)
        self.assertEqual(result["unmatched_right_count"], 0)

    def test_row_in_slice_matches_ordinary_control_anchor_only(self):
        row = {
            "canonical_pair_id": "a",
            "evidence_state": "ordinary_control_anchor",
            "semantic_geometry_conflict_score": "0.1",
            "observability_score": "0.9",
        }

        self.assertTrue(row_in_slice(row, "ordinary_control_anchor"))
        self.assertFalse(row_in_slice(row, "semantic_geometry_conflict"))

    def test_compute_slice_metrics_counts_slices_and_baseline_score_stats(self):
        rows = [
            {
                "canonical_pair_id": "a",
                "evidence_state": "ordinary_control_anchor",
                "baseline_final_score": "0.2",
                "matcher_sufficiency_score": "0.8",
                "reference_overlap": "1",
                "baseline_overlap": "1",
            },
            {
                "canonical_pair_id": "b",
                "evidence_state": "hard_trainable",
                "baseline_final_score": "0.6",
                "matcher_sufficiency_score": "0.6",
                "baseline_overlap": "1",
            },
            {
                "canonical_pair_id": "c",
                "evidence_state": "hard_trainable",
                "semantic_geometry_conflict_score": "0.8",
                "baseline_final_score": "1.0",
                "matcher_sufficiency_score": "0.4",
            },
        ]

        metrics = compute_slice_metrics(rows)

        self.assertEqual(metrics["ordinary_control_anchor"]["row_count"], 1)
        self.assertEqual(metrics["hard_trainable"]["row_count"], 2)
        self.assertEqual(metrics["semantic_geometry_conflict"]["row_count"], 1)
        self.assertAlmostEqual(metrics["hard_trainable"]["baseline_final_score_mean"], 0.8)
        self.assertAlmostEqual(metrics["hard_trainable"]["baseline_final_score_variance"], 0.04)
        self.assertEqual(metrics["ordinary_control_anchor"]["reference_overlap_count"], 1)
        self.assertEqual(metrics["hard_trainable"]["baseline_overlap_count"], 1)

    def test_make_shadow_rows_joins_evidence_to_baseline_first(self):
        result = make_shadow_rows(
            evidence_rows=[
                {"canonical_pair_id": "a", "evidence_state": "ordinary_control_anchor"},
                {"canonical_pair_id": "b", "evidence_state": "hard_trainable"},
            ],
            baseline_rows=[
                {"canonical_pair_id": "a", "baseline_final_score": "0.25"},
            ],
        )

        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["canonical_pair_id"], "a")
        self.assertEqual(result["rows"][0]["baseline_final_score"], "0.25")
        self.assertEqual(result["unmatched_evidence_count"], 1)
        self.assertEqual(result["unmatched_baseline_count"], 0)


if __name__ == "__main__":
    unittest.main()
