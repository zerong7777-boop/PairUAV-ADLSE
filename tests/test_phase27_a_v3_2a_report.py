"""Tests for A-v3.2a route verdict logic."""
import unittest

from scripts.phase27_a_v3_2a_report import compute_route_verdict


class ReportVerdictTests(unittest.TestCase):
    def test_stress_duplicate_blocks_candidate_full_pass(self):
        pairwise_metrics = {
            "pairwise_rows": [
                {
                    "left_artifact": "candidate",
                    "right_artifact": "full_dev",
                    "key_role": "promotion_key",
                    "intersection_count": "10000",
                    "duplicate_blocked_count": "0",
                    "promotion_eligible": "true",
                },
                {
                    "left_artifact": "candidate",
                    "right_artifact": "stress64030429",
                    "key_role": "promotion_key_candidate",
                    "intersection_count": "8207",
                    "duplicate_blocked_count": "723",
                    "promotion_eligible": "false",
                },
            ]
        }
        verdict = compute_route_verdict({}, pairwise_metrics, {"repair_candidates": []})
        self.assertEqual(verdict["verdict"], "identity-contract-blocked-duplicates")

    def test_all_stress_clean_allows_pass(self):
        pairwise_metrics = {
            "pairwise_rows": [
                {
                    "left_artifact": "candidate",
                    "right_artifact": "full_dev",
                    "key_role": "promotion_key",
                    "intersection_count": "10000",
                    "duplicate_blocked_count": "0",
                    "promotion_eligible": "true",
                },
                {
                    "left_artifact": "candidate",
                    "right_artifact": "stress64030429",
                    "key_role": "promotion_key_candidate",
                    "intersection_count": "8207",
                    "duplicate_blocked_count": "0",
                    "promotion_eligible": "true",
                },
            ]
        }
        verdict = compute_route_verdict({}, pairwise_metrics, {"repair_candidates": []})
        self.assertEqual(verdict["verdict"], "identity-contract-pass")


if __name__ == "__main__":
    unittest.main()
