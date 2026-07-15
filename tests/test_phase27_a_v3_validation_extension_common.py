import csv
import json
import os
import tempfile
import unittest

from scripts.phase27_a_v3_validation_extension_common import (
    auc_pairwise,
    count_true,
    decile_table,
    ensure_dirs,
    group_count,
    precision_recall_at_k,
    quantiles,
    rank_rows,
    read_csv_dicts,
    safe_div,
    to_float,
    truthy,
    write_csv_dicts,
    write_json,
)


class Phase27AValidationExtensionCommonTests(unittest.TestCase):
    def test_to_float_handles_blank_and_invalid_values(self):
        self.assertEqual(to_float(""), 0.0)
        self.assertEqual(to_float("   "), 0.0)
        self.assertEqual(to_float(None), 0.0)
        self.assertEqual(to_float("not-a-number"), 0.0)
        self.assertEqual(to_float("3.25"), 3.25)
        self.assertEqual(to_float(4), 4.0)

    def test_truthy_handles_bool_string_and_int_values(self):
        for value in (True, 1, "1", "true", "TRUE", "yes", "Y", "on"):
            self.assertTrue(truthy(value))
        for value in (False, 0, "0", "false", "FALSE", "no", "N", "", None):
            self.assertFalse(truthy(value))

    def test_quantiles_returns_requested_percentiles(self):
        values = list(range(1, 101))
        self.assertEqual(
            quantiles(values),
            {"p50": 50.5, "p90": 90.1, "p95": 95.05, "p99": 99.01},
        )

    def test_safe_div_returns_zero_for_zero_denominator(self):
        self.assertEqual(safe_div(5, 0), 0.0)
        self.assertEqual(safe_div(5, "bad"), 0.0)
        self.assertEqual(safe_div(6, 3), 2.0)

    def test_read_write_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "rows.csv")
            rows = [
                {"id": "a", "score": "1.5", "extra": "ignored"},
                {"id": "b", "score": "2.0"},
            ]
            write_csv_dicts(path, rows, ["id", "score"])

            with open(path, newline="", encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
            self.assertEqual(raw_rows, [{"id": "a", "score": "1.5"}, {"id": "b", "score": "2.0"}])
            self.assertEqual(read_csv_dicts(path), raw_rows)

    def test_write_json_and_ensure_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "a", "b")
            ensure_dirs(out_dir)
            path = os.path.join(out_dir, "data.json")
            write_json(path, {"b": 2, "a": 1})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"a": 1, "b": 2})

    def test_rank_rows_sorts_descending_with_missing_scores_last(self):
        rows = [
            {"id": "missing"},
            {"id": "low", "score": "1"},
            {"id": "bad", "score": "bad"},
            {"id": "high", "score": "3"},
        ]
        ranked = rank_rows(rows, "score")
        self.assertEqual([row["id"] for row in ranked], ["high", "low", "missing", "bad"])

    def test_precision_recall_at_k_computes_precision_and_recall(self):
        rows = [
            {"id": "a", "score": "0.9", "outcome": "1"},
            {"id": "b", "score": "0.8", "outcome": "0"},
            {"id": "c", "score": "0.7", "outcome": "1"},
            {"id": "d", "score": "0.1", "outcome": "1"},
        ]
        metrics = precision_recall_at_k(rows, "score", "outcome", 2)
        self.assertEqual(metrics["k"], 2)
        self.assertEqual(metrics["tp_at_k"], 1)
        self.assertEqual(metrics["positives"], 3)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0 / 3.0)

    def test_auc_pairwise_perfect_and_tied_scores(self):
        perfect = [
            {"score": "0.9", "outcome": "1"},
            {"score": "0.8", "outcome": "1"},
            {"score": "0.2", "outcome": "0"},
            {"score": "0.1", "outcome": "0"},
        ]
        tied = [
            {"score": "0.5", "outcome": "1"},
            {"score": "0.5", "outcome": "0"},
        ]
        self.assertEqual(auc_pairwise(perfect, "score", "outcome"), 1.0)
        self.assertEqual(auc_pairwise(tied, "score", "outcome"), 0.5)

    def test_decile_table_creates_10_bins_when_enough_rows_exist(self):
        rows = [{"score": str(i), "outcome": "1" if i % 2 else "0"} for i in range(100)]
        table = decile_table(rows, "score", "outcome")
        self.assertEqual(len(table), 10)
        self.assertEqual(sum(row["n"] for row in table), 100)
        self.assertEqual(table[0]["decile"], 1)
        self.assertEqual(table[-1]["decile"], 10)

    def test_count_true_and_group_count(self):
        rows = [
            {"flag": "1", "kind": "a"},
            {"flag": "false", "kind": "b"},
            {"flag": True, "kind": "a"},
            {"kind": ""},
        ]
        self.assertEqual(count_true(rows, "flag"), 2)
        self.assertEqual(group_count(rows, "kind"), {"a": 2, "b": 1, "": 1})


if __name__ == "__main__":
    unittest.main()
