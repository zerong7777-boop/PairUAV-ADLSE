import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase27_a_taxonomy_redesign_v2_schema as schema


class Phase27ATaxonomySchemaTest(unittest.TestCase):
    def test_source_categories_are_exact_contract(self):
        self.assertEqual(
            schema.SOURCE_CATEGORIES,
            (
                "INPUT_SIDE_NON_LEAKING",
                "MATCHER_SIDE_NON_LEAKING",
                "BASELINE_OUTCOME_VALIDATION_ONLY",
                "OUT_OF_FOLD_TRAINING_ELIGIBLE",
                "ANALYSIS_ONLY",
            ),
        )

    def test_validate_schema_columns_accepts_complete_contract(self):
        self.assertEqual(
            schema.IDENTITY_FIELDS,
            (
                "canonical_pair_id",
                "source_image_key",
                "target_image_key",
                "target_key",
                "scene_key",
                "split_key",
                "key_schema_version",
            ),
        )
        self.assertEqual(
            schema.NON_LEAKING_AXIS_FIELDS,
            (
                "evidence_sufficiency_score",
                "heading_observability_score",
                "range_observability_score",
                "semantic_geometric_conflict_score",
                "match_sufficiency_score",
                "layout_scale_risk_score",
                "augmentation_consistency_score",
            ),
        )
        self.assertEqual(
            schema.ANALYSIS_ONLY_AXIS_FIELDS,
            (
                "baseline_error_score",
                "heading_error_score",
                "range_error_score",
                "stress_sensitivity_score",
                "checkpoint_disagreement_score",
                "tail_outlier_flag",
            ),
        )
        self.assertEqual(
            schema.DERIVED_FIELDS,
            (
                "ambiguity_tail_risk_score",
                "low_observable_flag",
                "control_stability_score",
                "derived_state",
                "training_readiness_verdict",
                "validation_status",
            ),
        )
        columns = (
            schema.IDENTITY_FIELDS
            + schema.NON_LEAKING_AXIS_FIELDS
            + schema.ANALYSIS_ONLY_AXIS_FIELDS
            + schema.DERIVED_FIELDS
        )
        result = schema.validate_schema_columns(columns)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["unknown"], [])
        self.assertEqual(result["duplicate"], [])

    def test_validate_schema_columns_rejects_missing_unknown_and_duplicate(self):
        columns = [
            "canonical_pair_id",
            "canonical_pair_id",
            "evidence_sufficiency_score",
            "unknown_metric",
        ]
        result = schema.validate_schema_columns(columns)
        self.assertIn("source_image_key", result["missing"])
        self.assertIn("unknown_metric", result["unknown"])
        self.assertIn("canonical_pair_id", result["duplicate"])

    def test_validate_source_categories_covers_every_field_once(self):
        result = schema.validate_source_categories()
        self.assertEqual(result["unknown_categories"], {})
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(result["extra_fields"], [])

    def test_deployable_fields_exclude_forbidden_patterns(self):
        self.assertEqual(schema.assert_no_forbidden_deployable_fields(), [])
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["baseline_error_score"], "BASELINE_OUTCOME_VALIDATION_ONLY")
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["heading_error_score"], "BASELINE_OUTCOME_VALIDATION_ONLY")
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["range_error_score"], "BASELINE_OUTCOME_VALIDATION_ONLY")
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["stress_sensitivity_score"], "BASELINE_OUTCOME_VALIDATION_ONLY")
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["tail_outlier_flag"], "BASELINE_OUTCOME_VALIDATION_ONLY")
        self.assertEqual(schema.FIELD_SOURCE_CATEGORY["ambiguity_tail_risk_score"], "BASELINE_OUTCOME_VALIDATION_ONLY")

        original = dict(schema.FIELD_SOURCE_CATEGORY)
        try:
            schema.FIELD_SOURCE_CATEGORY["baseline_error_score"] = "INPUT_SIDE_NON_LEAKING"
            with self.assertRaises(AssertionError):
                schema.assert_no_forbidden_deployable_fields()
        finally:
            schema.FIELD_SOURCE_CATEGORY.clear()
            schema.FIELD_SOURCE_CATEGORY.update(original)

    def test_write_source_category_registry_writes_stable_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "registry.csv"
            json_path = Path(tmpdir) / "registry.json"

            csv_result = schema.write_source_category_registry(csv_path)
            json_result = schema.write_source_category_registry(json_path)

            self.assertEqual(csv_result, csv_path)
            self.assertEqual(json_result, json_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], {"field": "canonical_pair_id", "source_category": "INPUT_SIDE_NON_LEAKING"})
            self.assertEqual(rows[-1], {"field": "validation_status", "source_category": "ANALYSIS_ONLY"})

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_categories"], list(schema.SOURCE_CATEGORIES))
            self.assertIn(
                {"field": "derived_state", "source_category": "OUT_OF_FOLD_TRAINING_ELIGIBLE"},
                payload["fields"],
            )


if __name__ == "__main__":
    unittest.main()
