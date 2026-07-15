import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_2b_common import (
    canonical_pair_key,
    classify_duplicate_groups,
    normalize_image_key,
    normalize_token,
    read_csv_dicts,
    sha256_rows,
    source_target_composite_key,
    write_csv_dicts,
    write_json,
)


class CommonTests(unittest.TestCase):
    def test_csv_json_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            rows = [{"a": "1", "b": "2"}]
            write_csv_dicts(path, rows, ["a", "b"])
            self.assertEqual(read_csv_dicts(path), rows)
            write_json(Path(d) / "x.json", {"b": 2, "a": 1})
            self.assertIn('"a": 1', (Path(d) / "x.json").read_text())
            self.assertNotEqual(sha256_rows(rows, ["a"]), sha256_rows([{"a": "2"}], ["a"]))

    def test_keys_and_duplicates(self):
        self.assertEqual(normalize_token("A\\\\B//C "), "a/b/c")
        self.assertEqual(normalize_image_key("G/IMG-01.JPEG"), "g/img-01")
        row = {"canonical_pair_id": " 0839/01_01 ", "source_image_key": "A.JPG", "target_image_key": "B.png"}
        self.assertEqual(canonical_pair_key(row), "0839/01_01")
        self.assertEqual(source_target_composite_key(row), "a|b")
        groups = classify_duplicate_groups([{"id": "a"}, {"id": "a"}, {"id": ""}, {"id": "b"}], lambda r: r["id"])
        self.assertEqual(len(groups["duplicates"]), 1)
        self.assertEqual(len(groups["missing"]), 1)
        self.assertEqual(len(groups["unique"]), 1)


if __name__ == "__main__":
    unittest.main()

