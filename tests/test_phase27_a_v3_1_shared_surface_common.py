import csv
import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import (
    detect_duplicate_keys,
    quantiles,
    read_csv_dicts,
    safe_div,
    stable_pair_key,
    to_float,
    truthy,
    write_csv_dicts,
)


class SharedSurfaceCommonTests(unittest.TestCase):
    def test_truthy_and_to_float_and_safe_div(self):
        self.assertTrue(truthy("joined"))
        self.assertTrue(truthy("TRUE"))
        self.assertFalse(truthy("unjoined"))
        self.assertEqual(to_float("bad"), 0.0)
        self.assertEqual(to_float("2.5"), 2.5)
        self.assertEqual(safe_div(1, 0), 0.0)
        self.assertEqual(safe_div(3, 2), 1.5)

    def test_stable_pair_key_priority(self):
        self.assertEqual(stable_pair_key({"canonical_pair_id": "c1"}), ("canonical_pair_id", "c1"))
        row = {"source_image_key": "s", "target_image_key": "t", "pair_key": "p"}
        self.assertEqual(stable_pair_key(row), ("source_target_pair_composite", "s::t::p"))
        self.assertEqual(stable_pair_key({"pair_id": "p1"}), ("fallback_pair_id", "p1"))

    def test_detect_duplicates(self):
        rows = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}]
        self.assertEqual(set(detect_duplicate_keys(rows)), {"a"})

    def test_csv_roundtrip_selected_fields_and_quantiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv_dicts(path, [{"a": 1, "b": 2, "c": 3}], ["a", "b"])
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle)), {"a": "1", "b": "2"})
            self.assertEqual(read_csv_dicts(path), [{"a": "1", "b": "2"}])
        self.assertEqual(quantiles(range(1, 101))["p95"], 95.05)


if __name__ == "__main__":
    unittest.main()
