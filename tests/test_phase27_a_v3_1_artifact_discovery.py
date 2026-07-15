import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_1_artifact_discovery import (
    build_input_manifest,
    classify_surface_file,
    discover_csv_files,
    preview_csv_header,
)


class ArtifactDiscoveryTests(unittest.TestCase):
    def test_discover_preview_classify_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "experiments" / "phase27_a_taxonomy_redesign_v3" / "manifests"
            base = root / "experiments" / "phase27_a_validation_spine" / "baseline_surfaces"
            ext = root / "experiments" / "phase27_a_v3_validation_extension_outcome_consistency_audit"
            exp.mkdir(parents=True)
            base.mkdir(parents=True)
            ext.mkdir(parents=True)
            (exp / "training_readiness_verdict_manifest.csv").write_text("canonical_pair_id,evidence_sufficient_candidate\np1,true\n", encoding="utf-8")
            (base / "baseline_surface_main.csv").write_text("canonical_pair_id,baseline_pred_angle\np1,1\n", encoding="utf-8")
            (ext / "stress_surface_aug.csv").write_text("canonical_pair_id,stress_pred_angle\np1,2\n", encoding="utf-8")
            (ext / "random.txt").write_text("x", encoding="utf-8")
            files = discover_csv_files([root])
            self.assertEqual(len(files), 3)
            preview = preview_csv_header(exp / "training_readiness_verdict_manifest.csv")
            self.assertEqual(preview["fieldnames"][0], "canonical_pair_id")
            self.assertEqual(classify_surface_file(base / "baseline_surface_main.csv", ["baseline_pred_angle"]), "baseline_surface")
            manifest = build_input_manifest(root)
            self.assertFalse(manifest["missing_expected_families"])


if __name__ == "__main__":
    unittest.main()
