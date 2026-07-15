import tempfile
import unittest
from pathlib import Path

from scripts import phase27_a_taxonomy_redesign_v3_io as io
from scripts import phase27_a_taxonomy_redesign_v3_schema as schema


class SchemaContractTest(unittest.TestCase):
    def test_required_layer_fields_exist(self):
        self.assertIn("evidence_sufficient_candidate", schema.LAYER1_CANDIDATE_FIELDS)
        self.assertIn("baseline_heading_hard", schema.LAYER2_OUTCOME_FIELDS)
        self.assertIn("READY_CORRESPONDENCE_DIAGNOSTIC", schema.LAYER3_READINESS_FIELDS)

    def test_all_required_fields_have_registry_entries(self):
        self.assertTrue(schema.validate_field_registry())
        for field in schema.REQUIRED_FIELDS:
            self.assertIn(field, schema.SOURCE_CATEGORY_BY_FIELD)
            self.assertIn(field, schema.DEPLOYABILITY_BY_FIELD)
            self.assertIn(schema.SOURCE_CATEGORY_BY_FIELD[field], schema.SOURCE_CATEGORIES)
            self.assertIn(schema.DEPLOYABILITY_BY_FIELD[field], schema.DEPLOYABILITY_LABELS)

    def test_forbidden_leakage_patterns_are_not_deployable(self):
        self.assertTrue(schema.assert_no_forbidden_deployable_fields())
        self.assertNotEqual(schema.DEPLOYABILITY_BY_FIELD["base_hard_trainable"], "deployable")
        self.assertNotEqual(schema.DEPLOYABILITY_BY_FIELD["final_state"], "deployable")
        self.assertNotEqual(schema.DEPLOYABILITY_BY_FIELD["derived_state"], "deployable")

    def test_validate_required_fields_reports_missing(self):
        with self.assertRaisesRegex(ValueError, "evidence_sufficient_candidate"):
            schema.validate_required_fields(["canonical_pair_id"])

    def test_write_schema_registry(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "schema.json"
            schema.write_schema_registry(path)
            self.assertTrue(path.exists())
            self.assertIn("phase27_a_taxonomy_redesign_v3_schema_v1", path.read_text())


class IoContractTest(unittest.TestCase):
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.csv"
            rows = [{"canonical_pair_id": "a::b", "value": "1"}]
            io.write_csv_dicts(path, rows, ["canonical_pair_id", "value"])
            self.assertEqual(io.read_csv_dicts(path), rows)

    def test_canonical_pair_id_selection_when_present(self):
        self.assertEqual(io.build_canonical_pair_id({"canonical_pair_id": "x"}), "x")

    def test_canonical_pair_id_fallback_from_source_target(self):
        self.assertEqual(
            io.build_canonical_pair_id({"source_image_key": "src", "target_image_key": "tgt"}),
            "src::tgt",
        )

    def test_left_join_preserves_all_rows_and_statuses(self):
        left = [
            {"canonical_pair_id": "a"},
            {"canonical_pair_id": "b"},
            {"source_image_key": "", "target_image_key": ""},
        ]
        right = [{"canonical_pair_id": "a", "score": "0.5"}]
        rows = io.left_join_by_pair_id(left, right, "full_dev")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["full_dev_join_status"], "joined")
        self.assertEqual(rows[0]["full_dev_score"], "0.5")
        self.assertEqual(rows[1]["full_dev_join_status"], "unjoined")
        self.assertEqual(rows[2]["full_dev_join_status"], "missing_key")
        self.assertEqual(
            io.summarize_join_status(rows, "full_dev_join_status"),
            {"joined": 1, "unjoined": 1, "missing_key": 1},
        )

    def test_numeric_and_bool_parsing(self):
        self.assertIsNone(io.to_float(""))
        self.assertIsNone(io.to_float(None))
        self.assertEqual(io.to_float("1.25"), 1.25)
        self.assertTrue(io.to_bool("true"))
        self.assertFalse(io.to_bool("0"))


if __name__ == "__main__":
    unittest.main()
