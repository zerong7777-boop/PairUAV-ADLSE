import unittest

from scripts.phase27_a_evidence_state_v3_overlap_identity import (
    FORBIDDEN_IDENTITY_PATTERNS,
    IDENTITY_COLUMNS,
    attach_canonical_keys,
    audit_forbidden_columns,
    canonical_image_token,
    canonical_pair_keys,
    classify_order,
    count_key_duplicates,
    key_coverage,
)


class Phase27OverlapIdentityTests(unittest.TestCase):
    def test_canonical_image_token_normalizes_extensions_and_padding(self):
        self.assertEqual("02", canonical_image_token("image-02.jpeg"))
        self.assertEqual("02", canonical_image_token("image-2"))
        self.assertEqual("07", canonical_image_token("Image_0007.JPG"))
        self.assertEqual("08", canonical_image_token("0008.png"))

    def test_pair_keys_preserve_order_and_orderless_key(self):
        row = {
            "group_id": "0881",
            "image_a_name": "image-33.jpeg",
            "image_b_name": "image-02.jpeg",
        }
        keys = canonical_pair_keys(row)
        self.assertEqual("0881/33_02", keys["canonical_pair_id"])
        self.assertEqual("0881/02_33", keys["canonical_pair_id_flipped"])
        self.assertEqual("0881/02_33", keys["canonical_pair_id_orderless"])
        self.assertEqual("0881", keys["canonical_group_id"])
        self.assertEqual("33", keys["canonical_query_id"])
        self.assertEqual("02", keys["canonical_reference_id"])

    def test_pair_id_field_is_respected_and_normalized(self):
        row = {"group_id": "0881", "pair_id": "0881/image-33_image-2"}
        keys = canonical_pair_keys(row)
        self.assertEqual("0881/33_02", keys["canonical_pair_id"])
        self.assertEqual("pair_id", keys["identity_source"])

    def test_order_classification(self):
        self.assertEqual("same_order", classify_order("0881/33_02", "0881/33_02"))
        self.assertEqual("flipped_order", classify_order("0881/33_02", "0881/02_33"))
        self.assertEqual("different_pair", classify_order("0881/33_02", "0881/11_22"))

    def test_duplicate_counter_reports_duplicate_keys(self):
        rows = [
            {"canonical_pair_id": "g/01_02"},
            {"canonical_pair_id": "g/01_02"},
            {"canonical_pair_id": "g/01_03"},
        ]
        result = count_key_duplicates(rows, "canonical_pair_id")
        self.assertEqual(1, result["duplicate_key_count"])
        self.assertEqual(2, result["duplicate_row_count"])
        self.assertEqual({"g/01_02": 2}, result["duplicate_keys"])

    def test_key_coverage_counts_nonempty_rows(self):
        rows = [
            {"canonical_pair_id": "g/01_02"},
            {"canonical_pair_id": ""},
            {},
        ]
        result = key_coverage(rows, "canonical_pair_id")
        self.assertEqual(3, result["row_count"])
        self.assertEqual(1, result["nonempty_row_count"])
        self.assertEqual(2, result["missing_row_count"])
        self.assertAlmostEqual(1 / 3, result["nonempty_fraction"])

    def test_attach_canonical_keys_does_not_mutate_input(self):
        row = {
            "group_id": "0881",
            "image_a_name": "image-33.jpeg",
            "image_b_name": "image-02.jpeg",
        }
        out = attach_canonical_keys(row)
        self.assertEqual("0881/33_02", out["canonical_pair_id"])
        self.assertNotIn("canonical_pair_id", row)

    def test_forbidden_columns_are_rejected(self):
        audit = audit_forbidden_columns(["group_id", "image_a", "final_score", "angle_err"])
        self.assertFalse(audit["passed"])
        self.assertIn("final_score", audit["forbidden_columns"])
        self.assertIn("angle_err", audit["forbidden_columns"])

    def test_constants_match_identity_contract(self):
        self.assertIn("final_score", FORBIDDEN_IDENTITY_PATTERNS)
        self.assertIn("angle_err", FORBIDDEN_IDENTITY_PATTERNS)
        self.assertIn("group_id", IDENTITY_COLUMNS)
        self.assertIn("image_a_name", IDENTITY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
