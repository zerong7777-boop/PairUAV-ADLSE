import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_validation_spine_registry import (
    artifact_entry,
    compute_columns_hash,
    load_registry,
    validate_registry,
    write_registry,
)


class ValidationSpineRegistryTests(unittest.TestCase):
    def test_artifact_entry_has_required_fields(self):
        entry = artifact_entry(
            artifact_id="evidence_manifest_v3",
            artifact_kind="evidence_manifest",
            path="/tmp/evidence.csv",
            schema_version="evidence_manifest_v1",
            key_schema_version="pair_key_v1",
            row_count=3,
            columns=["canonical_pair_id", "evidence_state"],
            source_artifacts=["pair_manifest_v1"],
            generated_by="unit-test",
            read_only=True,
        )
        self.assertEqual(entry["columns_hash"], compute_columns_hash(["evidence_state", "canonical_pair_id"]))
        self.assertEqual(entry["source_artifacts"], ["pair_manifest_v1"])
        for field in (
            "artifact_id",
            "artifact_kind",
            "path",
            "storage_location",
            "schema_version",
            "key_schema_version",
            "row_count",
            "columns_hash",
            "source_artifacts",
            "generated_by",
            "generated_at",
            "read_only",
            "notes",
        ):
            self.assertIn(field, entry)

    def test_registry_roundtrip_and_lineage_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifacts.json"
            registry = {
                "pair_manifest_v1": artifact_entry("pair_manifest_v1", "pair_manifest", "/tmp/pair.csv", "pair_v1", "pair_key_v1", 2, ["canonical_pair_id"], [], "unit-test", True),
                "evidence_manifest_v3": artifact_entry("evidence_manifest_v3", "evidence_manifest", "/tmp/evidence.csv", "evidence_v1", "pair_key_v1", 2, ["canonical_pair_id"], ["pair_manifest_v1"], "unit-test", True),
            }
            write_registry(path, registry)
            loaded = load_registry(path)
            result = validate_registry(loaded)
            self.assertTrue(result["passed"])
            self.assertEqual(len(loaded), 2)

    def test_registry_fails_missing_source_artifact(self):
        registry = {
            "derived": artifact_entry("derived", "metrics", "/tmp/m.json", "metrics_v1", "pair_key_v1", 1, ["x"], ["missing"], "unit-test", False)
        }
        result = validate_registry(registry)
        self.assertFalse(result["passed"])
        self.assertIn("derived -> missing", result["missing_source_artifacts"])


if __name__ == "__main__":
    unittest.main()
