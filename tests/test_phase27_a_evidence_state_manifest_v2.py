import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_evidence_state_schema_v2 import (
    BASE_FLAG_COLUMNS,
    BASE_REGIMES,
    RISK_TAGS,
    audit_construction_columns_v2,
    assign_evidence_state_v2,
    write_manifest_schema_v2,
)


class Phase27EvidenceStateManifestV2Test(unittest.TestCase):
    def test_assignment_has_exactly_one_base_regime(self):
        result = assign_evidence_state_v2(
            {
                "match_count": 30,
                "valid_ratio": 1.0,
                "confidence_sum": 12.0,
                "mean_confidence": 0.4,
                "occupied_cells": 8,
                "anchor_spread": 0.1,
                "scale_balance": 0.5,
                "fallback_used": 0,
            }
        )
        self.assertIn(result["base_regime"], BASE_REGIMES)
        self.assertEqual(sum(result[column] for column in BASE_FLAG_COLUMNS), 1)

    def test_risk_tags_are_binary(self):
        result = assign_evidence_state_v2(
            {
                "match_count": 30,
                "valid_ratio": 1.0,
                "confidence_sum": 12.0,
                "mean_confidence": 0.4,
                "occupied_cells": 8,
                "anchor_spread": 0.01,
                "scale_balance": 0.2,
                "fallback_used": 0,
            }
        )
        self.assertTrue(all(result[tag] in (0, 1) for tag in RISK_TAGS))

    def test_ordinary_anchor_can_coexist_with_weak_risk_tag(self):
        result = assign_evidence_state_v2(
            {
                "match_count": 30,
                "valid_ratio": 1.0,
                "confidence_sum": 12.0,
                "mean_confidence": 0.4,
                "occupied_cells": 8,
                "spatial_entropy": 0.1,
                "anchor_spread": 0.1,
                "scale_balance": 0.5,
                "fallback_used": 0,
            }
        )
        self.assertEqual(result["base_regime"], "ordinary_control_anchor")
        self.assertEqual(result["weak_spatial_support_tag"], 1)
        self.assertEqual(result["ordinary_with_risk_tag"], 1)

    def test_ordinary_anchor_is_positive_not_risk_residual(self):
        result = assign_evidence_state_v2(
            {
                "match_count": 30,
                "valid_ratio": 1.0,
                "confidence_sum": 12.0,
                "mean_confidence": 0.4,
                "occupied_cells": 8,
                "anchor_spread": 0.1,
                "scale_balance": 0.5,
                "fallback_used": 0,
            }
        )
        self.assertEqual(result["base_regime"], "ordinary_control_anchor")
        self.assertEqual(result["base_ordinary_control_anchor"], 1)

    def test_forbidden_construction_columns_are_rejected(self):
        audit = audit_construction_columns_v2(
            [
                "match_count",
                "final_score",
                "gt_angle",
                "gt_distance",
                "angle_err",
                "range_err",
                "combined_error",
                "phase14_slice",
                "leaderboard_rank",
            ]
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["forbidden_columns"],
            [
                "final_score",
                "gt_angle",
                "gt_distance",
                "angle_err",
                "range_err",
                "combined_error",
                "phase14_slice",
                "leaderboard_rank",
            ],
        )

    def test_missing_core_features_is_unknown(self):
        result = assign_evidence_state_v2({"match_count": 30, "fallback_used": 0})
        self.assertEqual(result["base_regime"], "unknown_insufficient_features")
        self.assertEqual(sum(result[column] for column in BASE_FLAG_COLUMNS), 1)

    def test_schema_writer_includes_v2_constants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest_schema_v2.json"
            write_manifest_schema_v2(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "phase27_a_manifest_v2")
        self.assertEqual(payload["base_regimes"], BASE_REGIMES)
        self.assertEqual(payload["risk_tags"], RISK_TAGS)
        self.assertIn("final_score", payload["forbidden_construction_patterns"])


if __name__ == "__main__":
    unittest.main()
