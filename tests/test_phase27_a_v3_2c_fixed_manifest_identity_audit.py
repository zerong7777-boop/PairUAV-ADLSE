import unittest

from scripts.phase27_a_v3_2c_fixed_manifest_identity_audit import audit_outputs


class FixedManifestIdentityAuditTests(unittest.TestCase):
    def test_passes_when_rows_match_and_identity_present(self):
        manifest = [{"canonical_pair_id": "a", "manifest_hash": "h"}]
        outcome = [
            (
                "baseline",
                [
                    {
                        "canonical_pair_id": "a",
                        "variant_id": "baseline",
                        "source_image_key": "s",
                        "target_image_key": "t",
                        "manifest_hash": "h",
                        "row_status": "ok",
                    }
                ],
            )
        ]
        rows, metrics = audit_outputs(manifest, outcome)
        self.assertEqual(metrics["verdict"], "fixed-manifest-runner-smoke-pass")
        self.assertEqual(rows[0]["pass"], "true")

    def test_blocks_metadata_loss(self):
        manifest = [{"canonical_pair_id": "a", "manifest_hash": "h"}]
        outcome = [("baseline", [{"canonical_pair_id": "a", "variant_id": "baseline", "manifest_hash": "h", "row_status": "ok"}])]
        _, metrics = audit_outputs(manifest, outcome)
        self.assertEqual(metrics["verdict"], "fixed-manifest-runner-blocked-metadata-loss")


if __name__ == "__main__":
    unittest.main()

