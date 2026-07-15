import unittest

from scripts.phase27_a_validation_spine_keys import (
    attach_canonical_keys,
    audit_forbidden_columns,
    audit_overlap,
    canonical_image_key,
    canonical_pair_keys,
    classify_pair_overlap,
    count_duplicate_keys,
)


class ValidationSpineKeyTests(unittest.TestCase):
    def test_image_key_normalizes_case_extension_and_padding(self):
        self.assertEqual(canonical_image_key("Image-02.JPG"), "02")
        self.assertEqual(canonical_image_key("image_2.jpeg"), "02")
        self.assertEqual(canonical_image_key("/root/group/image-0007.png"), "07")

    def test_pair_keys_keep_order_and_orderless_key(self):
        row = {"namespace": "train", "group_id": "0881", "image_a_name": "image-33.jpeg", "image_b_name": "image-02.jpeg"}
        keys = canonical_pair_keys(row)
        self.assertEqual(keys["canonical_pair_id"], "train/0881/33_02")
        self.assertEqual(keys["flipped_pair_id"], "train/0881/02_33")
        self.assertEqual(keys["orderless_pair_id"], "train/0881/02_33")
        self.assertEqual(keys["key_schema_version"], "pair_key_v1")

    def test_pair_id_fallback(self):
        row = {"namespace": "train", "pair_id": "0881/image-33_image-2"}
        keys = canonical_pair_keys(row)
        self.assertEqual(keys["canonical_pair_id"], "train/0881/33_02")

    def test_duplicate_keys_are_counted(self):
        rows = [{"canonical_pair_id": "a"}, {"canonical_pair_id": "a"}, {"canonical_pair_id": "b"}]
        result = count_duplicate_keys(rows, "canonical_pair_id")
        self.assertEqual(result["duplicate_key_count"], 1)
        self.assertEqual(result["duplicate_row_count"], 2)

    def test_forbidden_columns_fail(self):
        result = audit_forbidden_columns(["group_id", "image_a", "final_score", "angle_err"])
        self.assertFalse(result["passed"])
        self.assertIn("final_score", result["forbidden_columns"])
        self.assertIn("angle_err", result["forbidden_columns"])

    def test_overlap_audit_reports_flipped_and_unmatched(self):
        left = [
            attach_canonical_keys({"namespace": "train", "group_id": "g", "image_a_name": "image-01", "image_b_name": "image-02"}),
            attach_canonical_keys({"namespace": "train", "group_id": "g", "image_a_name": "image-05", "image_b_name": "image-06"}),
        ]
        right = [
            attach_canonical_keys({"namespace": "train", "group_id": "g", "image_a_name": "image-02", "image_b_name": "image-01"}),
        ]
        result = audit_overlap("left", left, "right", right)
        self.assertEqual(result["canonical_overlap"], 0)
        self.assertEqual(result["order_flipped_overlap"], 1)
        self.assertEqual(result["orderless_overlap"], 1)
        self.assertEqual(result["unmatched_left_count"], 1)
        self.assertEqual(classify_pair_overlap(left[0], right[0]), "flipped_order")


if __name__ == "__main__":
    unittest.main()
