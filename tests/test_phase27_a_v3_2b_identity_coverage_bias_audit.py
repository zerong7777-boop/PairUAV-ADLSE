import unittest

from scripts.phase27_a_v3_2b_identity_coverage_bias_audit import audit


class AuditTests(unittest.TestCase):
    def test_duplicate_and_missing_metrics_block_coverage(self):
        fixed = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}]
        shared = [{"canonical_pair_id": "a", "shared_pair_status": "ready"}, {"canonical_pair_id": "b", "shared_pair_status": "not_ready"}]
        outcome = {"stress_metrics": {"v": {"duplicate_id_count": 1, "source_target_composite_missing_row_count": 2, "missing_id_count": 1}}}
        identity, coverage, bias = audit(fixed, shared, outcome)
        self.assertEqual(identity["stress_duplicate_blocked_count"], 1)
        self.assertEqual(coverage["shared_coverage_ratio"], 0.5)
        self.assertTrue(bias)


if __name__ == "__main__":
    unittest.main()

