import unittest

from scripts.phase27_a_v3_2b_outcome_reexport import reexport


class OutcomeReexportTests(unittest.TestCase):
    def test_preserves_duplicate_and_missing_status(self):
        fixed = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}]
        baseline = [{"canonical_pair_id": "a", "source_image_a": "s.jpg", "source_image_b": "t.jpg"}]
        stress = [
            ("v1", [
                {"canonical_pair_id": "a", "source_row_index": "1", "baseline_angle_abs_error": "2"},
                {"canonical_pair_id": "a", "source_row_index": "2", "baseline_angle_abs_error": "3"},
            ])
        ]
        b, s, dup, metrics = reexport(fixed, baseline, stress)
        self.assertEqual(b[1]["baseline_join_status"], "missing")
        self.assertEqual(len(dup), 1)
        self.assertEqual(metrics["stress_metrics"]["v1"]["duplicate_id_count"], 1)
        self.assertTrue(any(r["stress_missing_status"] == "missing" for r in s))


if __name__ == "__main__":
    unittest.main()

