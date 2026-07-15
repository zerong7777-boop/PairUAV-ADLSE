import unittest

from scripts.phase27_a_v3_2b_shared_wide_surface import build_shared_wide


class SharedWideTests(unittest.TestCase):
    def test_duplicate_stress_blocks_ready_status(self):
        fixed = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}]
        baseline = [
            {"canonical_pair_id": "a", "baseline_join_status": "joined", "baseline_angle_abs_error": "1"},
            {"canonical_pair_id": "b", "baseline_join_status": "joined", "baseline_angle_abs_error": "1"},
        ]
        stress = [
            {"canonical_pair_id": "a", "variant_id": "v", "stress_join_status": "duplicate", "stress_duplicate_status": "duplicate"},
            {"canonical_pair_id": "b", "variant_id": "v", "stress_join_status": "joined", "stress_duplicate_status": "none", "stress_source_target_composite_present": "true", "stress_angle_abs_error": "3"},
        ]
        rows, fields, metrics = build_shared_wide(fixed, baseline, stress)
        self.assertEqual(rows[0]["shared_pair_status"], "not_ready")
        self.assertEqual(rows[1]["shared_pair_status"], "ready")
        self.assertEqual(metrics["shared_ready_count"], 1)


if __name__ == "__main__":
    unittest.main()

