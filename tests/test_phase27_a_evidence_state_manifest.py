import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_evidence_state_schema import (
    REQUIRED_ID_COLUMNS,
    STATE_COLUMNS,
    audit_construction_columns,
    assign_evidence_states,
    write_manifest_schema,
)


class Phase27EvidenceStateSchemaTest(unittest.TestCase):
    def test_forbidden_fields_are_rejected_for_construction(self):
        audit = audit_construction_columns(
            ["match_count", "angle_err", "phase14_slice", "confidence_sum"]
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["forbidden_columns"], ["angle_err", "phase14_slice"])
        self.assertIn("match_count", audit["allowed_columns"])
        self.assertIn("confidence_sum", audit["allowed_columns"])

    def test_validation_only_fields_are_not_construction_safe(self):
        audit = audit_construction_columns(
            ["combined_error", "official_final_score", "leaderboard_rank"]
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["forbidden_columns"],
            ["combined_error", "official_final_score", "leaderboard_rank"],
        )

    def test_state_assignment_returns_all_binary_states(self):
        states = assign_evidence_states(
            {
                "match_count": 80,
                "valid_ratio": 0.7,
                "confidence_sum": 10.0,
                "occupied_cells": 12,
                "spatial_entropy": 0.8,
                "anchor_spread": 0.3,
                "scale_balance": 0.9,
            }
        )
        self.assertEqual(set(states), set(STATE_COLUMNS))
        self.assertTrue(all(value in (0, 1) for value in states.values()))
        self.assertEqual(states["high_evidence_anchor"], 1)

    def test_ordinary_control_excludes_low_observable(self):
        states = assign_evidence_states(
            {
                "match_count": 2,
                "valid_ratio": 0.1,
                "confidence_sum": 0.2,
                "occupied_cells": 1,
            }
        )
        self.assertEqual(states["low_observable"], 1)
        self.assertEqual(states["ordinary_control"], 0)

    def test_high_evidence_protected_when_observability_missing(self):
        states = assign_evidence_states({"confidence_sum": 100.0, "valid_ratio": 0.9})
        self.assertEqual(states["high_evidence_anchor"], 0)
        self.assertEqual(states["low_observable"], 1)

    def test_conflict_and_shift_states_are_binary(self):
        states = assign_evidence_states(
            {
                "match_count": 30,
                "occupied_cells": 8,
                "confidence_sum": 4.0,
                "semantic_proxy": 0.9,
                "geometry_proxy": 0.2,
                "target_shift_proxy": 0.8,
            }
        )
        self.assertEqual(states["semantic_geometry_conflict_candidate"], 1)
        self.assertEqual(states["target_regime_shift_candidate"], 1)
        self.assertTrue(all(value in (0, 1) for value in states.values()))

    def test_write_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest_schema.json"
            write_manifest_schema(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["required_id_columns"], REQUIRED_ID_COLUMNS)
        self.assertEqual(payload["state_columns"], STATE_COLUMNS)
        self.assertEqual(payload["schema_version"], "phase27_a_manifest_v1")
        self.assertIn("forbidden_construction_patterns", payload)


if __name__ == "__main__":
    unittest.main()
