import unittest

from scripts.phase27_a_v3_1_shared_surface_builder import build_shared_surface


class SharedSurfaceBuilderTests(unittest.TestCase):
    def test_shared_missing_and_duplicate_cases(self):
        candidates = [
            {"canonical_pair_id": "p1", "ambiguity_candidate": "true"},
            {"canonical_pair_id": "p2"},
            {"canonical_pair_id": "p3"},
            {"canonical_pair_id": "p4"},
            {"canonical_pair_id": "dup"},
        ]
        baseline = [
            {"canonical_pair_id": "p1", "full_dev_join_status": "joined", "baseline_angle_abs_error": "1", "baseline_distance_rel_error": "2", "baseline_joint_error_score": "3", "baseline_joint_hard": "true"},
            {"canonical_pair_id": "p2", "full_dev_join_status": "joined", "baseline_angle_abs_error": "1", "baseline_distance_rel_error": "2", "baseline_joint_error_score": "3"},
            {"canonical_pair_id": "dup", "full_dev_join_status": "joined"},
            {"canonical_pair_id": "dup", "full_dev_join_status": "joined"},
        ]
        stress = {
            "main": [
                {"canonical_pair_id": "p1", "stress_join_status": "joined", "stress_baseline_angle_abs_error": "2", "stress_baseline_distance_rel_error": "5", "stress_baseline_final_score": "9", "stress_joint_sensitive": "true"},
                {"canonical_pair_id": "p3", "stress_join_status": "joined", "stress_baseline_angle_abs_error": "2"},
            ]
        }
        rows = {row["canonical_pair_id"]: row for row in build_shared_surface(candidates, baseline, stress)}
        self.assertEqual(rows["p1"]["shared_join_status"], "joined")
        self.assertEqual(rows["p1"]["stress_main_heading_delta"], 1.0)
        self.assertEqual(rows["p2"]["shared_join_status"], "missing_stress")
        self.assertEqual(rows["p3"]["shared_join_status"], "missing_baseline")
        self.assertEqual(rows["p4"]["shared_join_status"], "missing_both")
        self.assertEqual(rows["dup"]["shared_join_status"], "duplicate_blocked")
        self.assertEqual(rows["p1"]["ambiguity_candidate"], "true")


if __name__ == "__main__":
    unittest.main()
