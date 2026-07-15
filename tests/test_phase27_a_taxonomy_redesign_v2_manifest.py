import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = STAGE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        self._old_rules = sys.modules.get("phase27_a_taxonomy_redesign_v2_rules")
        fake_rules = types.ModuleType("phase27_a_taxonomy_redesign_v2_rules")

        def assign_rows(rows):
            out = []
            for row in rows:
                for required in [
                    "source_image_key",
                    "target_image_key",
                    "target_key",
                    "scene_key",
                    "split_key",
                    "key_schema_version",
                    "ambiguity_tail_risk_score",
                    "low_observable_flag",
                    "control_stability_score",
                    "validation_status",
                ]:
                    if required not in row:
                        raise AssertionError(f"missing rules input {required}")
                next_row = dict(row)
                stress = float(next_row.get("stress_sensitivity_score") or 0.0)
                if next_row["low_observable_flag"] == "True":
                    next_row["derived_state"] = "low_observable_review"
                elif stress > 0:
                    next_row["derived_state"] = "evidence_sufficient_hard"
                else:
                    next_row["derived_state"] = "stable_control_anchor"
                next_row["training_readiness_verdict"] = "review" if stress > 0 else "holdout"
                out.append(next_row)
            return out

        fake_rules.assign_rows = assign_rows
        sys.modules["phase27_a_taxonomy_redesign_v2_rules"] = fake_rules

    def tearDown(self):
        if self._old_rules is None:
            sys.modules.pop("phase27_a_taxonomy_redesign_v2_rules", None)
        else:
            sys.modules["phase27_a_taxonomy_redesign_v2_rules"] = self._old_rules

    def test_manifest_maps_real_columns_and_preserves_evidence_rows(self):
        module = importlib.import_module("phase27_a_taxonomy_redesign_v2_manifest")
        self.assertNotIn("pandas", sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.csv"
            full_path = tmp_path / "full.csv"
            stress_a_path = tmp_path / "stress_a.csv"
            stress_b_path = tmp_path / "stress_b.csv"
            out_manifest = tmp_path / "manifest.csv"
            out_registry = tmp_path / "source_registry.json"
            out_audit = tmp_path / "leakage_audit.json"

            write_csv(
                evidence_path,
                [
                    {
                        "canonical_pair_id": "p1",
                        "source_split": "dev",
                        "group_id": "g1",
                        "pair_id": "1",
                        "image_a": "a1",
                        "image_b": "b1",
                        "image_a_name": "a1.png",
                        "image_b_name": "b1.png",
                        "feature_complete": "1",
                        "observable_adequate": "1",
                        "image_quality_adequate": "1",
                        "pair_identity_valid": "1",
                        "adequacy_passed": "1",
                        "low_observable_reason": "",
                        "observability_axis": "0.8",
                        "pair_similarity_axis": "0.6",
                        "scale_risk_axis": "0.1",
                        "layout_risk_axis": "0.2",
                        "conflict_risk_axis": "0.0",
                        "control_centrality_score": "0.9",
                        "base_regime": "easy",
                        "base_easy": "1",
                    },
                    {
                        "canonical_pair_id": "p2",
                        "source_split": "dev",
                        "group_id": "g1",
                        "pair_id": "2",
                        "observability_axis": "0.3",
                        "pair_similarity_axis": "0.4",
                        "scale_risk_axis": "0.7",
                        "layout_risk_axis": "0.8",
                        "conflict_risk_axis": "0.9",
                        "control_centrality_score": "0.2",
                        "base_regime": "hard",
                        "base_hard": "1",
                    },
                    {
                        "canonical_pair_id": "p3",
                        "source_split": "dev",
                        "group_id": "g2",
                        "pair_id": "3",
                        "observable_adequate": "0",
                        "observability_axis": "0.5",
                        "pair_similarity_axis": "",
                        "scale_risk_axis": "0.3",
                        "layout_risk_axis": "0.4",
                        "conflict_risk_axis": "0.1",
                        "control_centrality_score": "0.5",
                        "base_regime": "easy",
                    },
                ],
            )
            write_csv(
                full_path,
                [
                    {
                        "canonical_pair_id": "p1",
                        "baseline_final_score": "0.1",
                        "baseline_angle_rel_error": "0.2",
                        "baseline_distance_rel_error": "0.1",
                        "baseline_angle_abs_error": "2",
                        "baseline_distance_abs_error": "5",
                        "baseline_surface_source": "full_dev",
                    },
                    {
                        "canonical_pair_id": "p2",
                        "baseline_final_score": "0.9",
                        "baseline_angle_rel_error": "0.8",
                        "baseline_distance_rel_error": "0.7",
                        "baseline_angle_abs_error": "8",
                        "baseline_distance_abs_error": "20",
                        "baseline_surface_source": "full_dev",
                    },
                ],
            )
            write_csv(
                stress_a_path,
                [
                    {"canonical_pair_id": "p1", "baseline_final_score": "0.2", "baseline_surface_source": "stress_a"},
                    {"canonical_pair_id": "p2", "baseline_final_score": "1.2", "baseline_surface_source": "stress_a"},
                ],
            )
            write_csv(
                stress_b_path,
                [
                    {"canonical_pair_id": "p1", "baseline_final_score": "0.3", "baseline_surface_source": "stress_b"},
                    {"canonical_pair_id": "p2", "baseline_final_score": "1.5", "baseline_surface_source": "stress_b"},
                ],
            )

            module.main(
                [
                    "--evidence-manifest",
                    str(evidence_path),
                    "--full-dev-surface",
                    str(full_path),
                    "--stress-surface",
                    str(stress_a_path),
                    "--stress-surface",
                    str(stress_b_path),
                    "--out-manifest",
                    str(out_manifest),
                    "--out-source-registry",
                    str(out_registry),
                    "--out-leakage-audit",
                    str(out_audit),
                ]
            )

            manifest = read_csv(out_manifest)
            self.assertEqual(["p1", "p2", "p3"], [row["canonical_pair_id"] for row in manifest])
            self.assertEqual(["easy", "hard", "easy"], [row["old_base_regime"] for row in manifest])
            required = {
                "canonical_pair_id",
                "source_image_key",
                "target_image_key",
                "target_key",
                "scene_key",
                "split_key",
                "key_schema_version",
                "evidence_sufficiency_score",
                "heading_observability_score",
                "range_observability_score",
                "semantic_geometric_conflict_score",
                "layout_scale_risk_score",
                "match_sufficiency_score",
                "match_sufficiency_source",
                "augmentation_consistency_score",
                "augmentation_consistency_note",
                "baseline_error_score",
                "heading_error_score",
                "range_error_score",
                "stress_sensitivity_score",
                "checkpoint_disagreement_score",
                "tail_outlier_flag",
                "ambiguity_tail_risk_score",
                "low_observable_flag",
                "control_stability_score",
                "validation_status",
                "derived_state",
                "training_readiness_verdict",
            }
            self.assertTrue(required.issubset(set(manifest[0].keys())))
            self.assertEqual("a1", manifest[0]["source_image_key"])
            self.assertEqual("b1", manifest[0]["target_image_key"])
            self.assertEqual("g1", manifest[0]["target_key"])
            self.assertEqual("g1", manifest[0]["scene_key"])
            self.assertEqual("dev", manifest[0]["split_key"])
            self.assertEqual("phase27_a_taxonomy_redesign_v2", manifest[0]["key_schema_version"])
            self.assertEqual("MATCHER_SIDE_NON_LEAKING", manifest[0]["match_sufficiency_source"])
            self.assertIn("proxy", manifest[0]["augmentation_consistency_note"])
            self.assertAlmostEqual(0.7, float(manifest[0]["evidence_sufficiency_score"]))
            self.assertAlmostEqual(0.5, float(manifest[2]["evidence_sufficiency_score"]))
            self.assertGreater(float(manifest[1]["stress_sensitivity_score"]), float(manifest[0]["stress_sensitivity_score"]))
            self.assertEqual("0.000000", manifest[2]["stress_sensitivity_score"])
            self.assertEqual(["joined_full_and_stress", "joined_full_and_stress", "unvalidated"], [row["validation_status"] for row in manifest])
            self.assertEqual("False", manifest[0]["low_observable_flag"])
            self.assertEqual("True", manifest[2]["low_observable_flag"])
            self.assertGreaterEqual(float(manifest[1]["ambiguity_tail_risk_score"]), 1.0)
            self.assertGreater(float(manifest[0]["control_stability_score"]), float(manifest[1]["control_stability_score"]))
            self.assertEqual(
                ["stable_control_anchor", "evidence_sufficient_hard", "low_observable_review"],
                [row["derived_state"] for row in manifest],
            )

            registry = json.loads(out_registry.read_text(encoding="utf-8"))
            self.assertEqual(str(evidence_path), registry["evidence_manifest"]["path"])
            self.assertEqual(2, len(registry["stress_surfaces"]))

            audit = json.loads(out_audit.read_text(encoding="utf-8"))
            self.assertEqual(3, audit["evidence_rows"])
            self.assertEqual(1, audit["unmatched_evidence_rows"])
            self.assertEqual(["canonical_pair_id"], audit["join_keys"])


if __name__ == "__main__":
    unittest.main()
