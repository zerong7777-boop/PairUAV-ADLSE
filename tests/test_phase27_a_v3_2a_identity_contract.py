import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_2a_identity_contract import contract_field_table, write_identity_contract


class IdentityContractTests(unittest.TestCase):
    def test_field_roles_and_contract_file(self):
        table = {row["field"]: row for row in contract_field_table()}
        self.assertEqual(table["canonical_pair_id"]["role"], "promotion_key")
        self.assertEqual(table["source_row_index"]["role"], "forbidden_key")
        self.assertEqual(table["target_key"]["role"], "metadata_only")
        with tempfile.TemporaryDirectory() as tmp:
            result = write_identity_contract(tmp)
            path = Path(result["path"])
            text = path.read_text(encoding="utf-8")
            self.assertIn("No training", text)
            self.assertIn("canonical_pair_id", text)


if __name__ == "__main__":
    unittest.main()
