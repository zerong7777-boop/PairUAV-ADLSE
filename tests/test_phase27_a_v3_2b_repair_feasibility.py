import unittest

from scripts.phase27_a_v3_2b_repair_feasibility import decide_verdict, repair_candidates


class RepairFeasibilityTests(unittest.TestCase):
    def test_duplicates_block_ready(self):
        verdict = decide_verdict(
            {"stress_duplicate_blocked_count": 1, "stress_source_target_composite_missing_count": 0},
            {"shared_coverage_ratio": 1.0},
            {"decision": "use_reexport_only_bounded"},
        )
        self.assertEqual(verdict["verdict"], "shared-surface-blocked-identity-duplicates")

    def test_clean_surface_ready(self):
        verdict = decide_verdict(
            {"stress_duplicate_blocked_count": 0, "stress_source_target_composite_missing_count": 0, "stress_missing_id_count": 0},
            {"shared_coverage_ratio": 0.5},
            {"decision": "attempt_fixed_manifest_reacquisition"},
        )
        self.assertEqual(verdict["verdict"], "shared-surface-ready-for-outcome-audit")


if __name__ == "__main__":
    unittest.main()

