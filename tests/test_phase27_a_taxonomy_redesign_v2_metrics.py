import csv
import importlib
import json
import sys
import tempfile
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


class MetricsModuleTests(unittest.TestCase):
    def test_metrics_cli_writes_expected_outputs_without_training_verdict(self):
        module = importlib.import_module("phase27_a_taxonomy_redesign_v2_metrics")
        self.assertNotIn("pandas", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.csv"
            out_dir = tmp_path / "metrics"
            write_csv(
                manifest_path,
                [
                    {
                        "canonical_pair_id": "p1",
                        "derived_state": "stable_control_anchor",
                        "old_base_regime": "ordinary_control_anchor",
                        "target_key": "g1",
                        "scene_key": "scene_a",
                        "baseline_error_score": "0.1",
                        "heading_error_score": "0.2",
                        "range_error_score": "0.1",
                        "stress_sensitivity_score": "0.0",
                        "checkpoint_disagreement_score": "0.0",
                        "tail_outlier_flag": "False",
                        "scene": "urban",
                        "target_label": "car",
                        "full_dev_joined": "True",
                        "stress_joined": "True",
                    },
                    {
                        "canonical_pair_id": "p2",
                        "derived_state": "evidence_sufficient_hard",
                        "old_base_regime": "hard_trainable",
                        "target_key": "g2",
                        "scene_key": "scene_a",
                        "baseline_error_score": "0.7",
                        "heading_error_score": "0.8",
                        "range_error_score": "0.6",
                        "stress_sensitivity_score": "0.5",
                        "checkpoint_disagreement_score": "0.2",
                        "tail_outlier_flag": "False",
                        "scene": "urban",
                        "target_label": "truck",
                        "full_dev_joined": "True",
                        "stress_joined": "True",
                    },
                    {
                        "canonical_pair_id": "p3",
                        "derived_state": "evidence_sufficient_hard",
                        "old_base_regime": "hard_trainable",
                        "target_key": "g1",
                        "scene_key": "scene_b",
                        "baseline_error_score": "0.5",
                        "heading_error_score": "0.4",
                        "range_error_score": "0.3",
                        "stress_sensitivity_score": "0.3",
                        "checkpoint_disagreement_score": "0.1",
                        "tail_outlier_flag": "False",
                        "scene": "forest",
                        "target_label": "car",
                        "full_dev_joined": "False",
                        "stress_joined": "True",
                    },
                    {
                        "canonical_pair_id": "p4",
                        "derived_state": "tail_risk",
                        "old_base_regime": "ordinary_control_anchor",
                        "target_key": "g1",
                        "scene_key": "scene_b",
                        "baseline_error_score": "0.9",
                        "heading_error_score": "1.0",
                        "range_error_score": "0.7",
                        "stress_sensitivity_score": "0.8",
                        "checkpoint_disagreement_score": "0.4",
                        "tail_outlier_flag": "True",
                        "scene": "forest",
                        "target_label": "car",
                        "full_dev_joined": "True",
                        "stress_joined": "False",
                    },
                ],
            )

            module.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--bootstrap-iters",
                    "50",
                    "--seed",
                    "123",
                ]
            )

            expected = [
                "state_counts.csv",
                "state_counts.json",
                "old_vs_new_taxonomy_comparison.csv",
                "old_vs_new_taxonomy_comparison.json",
                "per_state_surface_metrics.csv",
                "per_state_surface_metrics.json",
                "bootstrap_ci.json",
                "target_scene_bias.csv",
                "target_scene_bias.json",
                "join_coverage.json",
                "tail_risk_metrics.json",
                "final_verdict.json",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), name)

            counts = read_csv(out_dir / "state_counts.csv")
            stress = next(row for row in counts if row["derived_state"] == "evidence_sufficient_hard")
            self.assertEqual("2", stress["count"])
            self.assertAlmostEqual(0.5, float(stress["fraction"]))

            per_state = read_csv(out_dir / "per_state_surface_metrics.csv")
            self.assertIn("mean_composite_score", per_state[0])
            self.assertIn("median_composite_score", per_state[0])
            self.assertNotIn("hard_fraction", per_state[0])

            comparison = json.loads((out_dir / "old_vs_new_taxonomy_comparison.json").read_text(encoding="utf-8"))
            self.assertIn("old_hard_trainable_count", comparison)
            self.assertIn("old_ordinary_control_anchor_count", comparison)
            self.assertIn("new_state_hard_control_delta", comparison)
            self.assertGreater(comparison["new_state_hard_control_delta"], 0)

            bias = read_csv(out_dir / "target_scene_bias.csv")
            self.assertIn("target_key", bias[0])
            self.assertIn("scene_key", bias[0])

            coverage = json.loads((out_dir / "join_coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(4, coverage["total_rows"])
            self.assertAlmostEqual(0.75, coverage["full_dev_joined_fraction"])
            self.assertAlmostEqual(0.75, coverage["stress_joined_fraction"])

            tail = json.loads((out_dir / "tail_risk_metrics.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(0.25, tail["tail_outlier_rate"])

            verdict = json.loads((out_dir / "final_verdict.json").read_text(encoding="utf-8"))
            allowed = {
                "taxonomy-redesign-v2-ready-for-knowledge-review",
                "taxonomy-redesign-v2-needs-redesign",
                "taxonomy-redesign-v2-blocked-by-coverage",
                "taxonomy-redesign-v2-smoke-only",
            }
            self.assertIn(verdict["label"], allowed)
            self.assertNotEqual("ready-for-training-policy-spec", verdict["label"])

            ci = json.loads((out_dir / "bootstrap_ci.json").read_text(encoding="utf-8"))
            self.assertEqual(50, ci["bootstrap_iters"])
            self.assertEqual(123, ci["seed"])
            self.assertIn("new_state_hard_minus_control_delta", ci)
            self.assertGreater(ci["new_state_hard_minus_control_delta"], 0)

    def test_low_but_nonzero_overlap_is_not_blocked_by_fixed_half_threshold(self):
        module = importlib.import_module("phase27_a_taxonomy_redesign_v2_metrics")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.csv"
            out_dir = tmp_path / "metrics"
            rows = []
            for idx in range(20):
                rows.append(
                    {
                        "canonical_pair_id": f"p{idx}",
                        "derived_state": "evidence_sufficient_hard" if idx < 3 else "stable_control_anchor",
                        "old_base_regime": "hard_trainable" if idx < 3 else "ordinary_control_anchor",
                        "target_key": f"g{idx % 2}",
                        "scene_key": f"s{idx % 3}",
                        "baseline_error_score": "0.8" if idx < 3 else "0.1",
                        "heading_error_score": "0.7" if idx < 3 else "0.1",
                        "range_error_score": "0.7" if idx < 3 else "0.1",
                        "stress_sensitivity_score": "0.6" if idx < 3 else "0.0",
                        "checkpoint_disagreement_score": "0.3" if idx < 3 else "0.0",
                        "tail_outlier_flag": "False",
                        "full_dev_joined": "True" if idx < 4 else "False",
                        "stress_joined": "True" if idx < 4 else "False",
                    }
                )
            write_csv(manifest_path, rows)
            module.main(["--manifest", str(manifest_path), "--out-dir", str(out_dir), "--bootstrap-iters", "10", "--seed", "7"])
            coverage = json.loads((out_dir / "join_coverage.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(0.2, coverage["full_dev_joined_fraction"])
            verdict = json.loads((out_dir / "final_verdict.json").read_text(encoding="utf-8"))
            self.assertNotEqual("taxonomy-redesign-v2-blocked-by-coverage", verdict["label"])


if __name__ == "__main__":
    unittest.main()
