import csv
import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_2a_identity_common import (
    canonical_pair_id_key,
    direction_invariant_source_target_pair_key,
    identity_key_strategies,
    normalize_image_key,
    normalize_token,
    path_normalized_source_target_pair_key,
    read_csv_dicts,
    row_index_diagnostic_key,
    source_target_pair_composite_key,
    write_csv_dicts,
    write_json,
)


class IdentityCommonTests(unittest.TestCase):
    def test_csv_json_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            write_csv_dicts(path, [{"a": 1, "b": 2, "c": 3}], ["a", "b"])
            self.assertEqual(read_csv_dicts(path), [{"a": "1", "b": "2"}])
            write_json(Path(tmp) / "x.json", {"b": 2, "a": 1})
            self.assertIn('"a"', (Path(tmp) / "x.json").read_text())
        self.assertEqual(normalize_token(" A\\\\B//C "), "a/b/c")
        self.assertEqual(normalize_image_key("/Root/IMG-001.JPG"), "img_001")

    def test_key_strategies(self):
        row = {
            "canonical_pair_id": "C1",
            "source_image_key": "a.jpg",
            "target_image_key": "b.JPG",
            "pair_key": "p",
            "source_row_index": "12",
        }
        self.assertEqual(canonical_pair_id_key(row), "c1")
        self.assertEqual(source_target_pair_composite_key(row), "a.jpg||b.jpg||p")
        self.assertEqual(direction_invariant_source_target_pair_key(row), "a.jpg||b.jpg||p")
        self.assertEqual(path_normalized_source_target_pair_key(row), "a||b||p")
        self.assertEqual(row_index_diagnostic_key(row), "12")
        roles = {s["name"]: s["role"] for s in identity_key_strategies()}
        self.assertEqual(roles["row_index_diagnostic_only"], "forbidden_for_promotion")


if __name__ == "__main__":
    unittest.main()
