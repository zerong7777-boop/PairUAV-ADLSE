import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.phase27_a_target_regime_conditioned_surface_audit import run_audit


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TargetRegimeConditionedSurfaceAuditTests(unittest.TestCase):
    def test_target_regime_assignment_and_residual_surface(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for target, base in [("easy_t", 10.0), ("mid_t", 30.0), ("hard_t", 80.0)]:
                for idx, state in enumerate(["easy_state", "hard_state"] * 6):
                    delta = -5.0 if state == "easy_state" else 5.0
                    rows.append(
                        {
                            "canonical_pair_id": f"{target}_{idx}",
                            "target_key": target,
                            "state": state,
                            "joint_error": str(base + delta),
                            "heading_abs_error": str(base + delta),
                            "range_abs_error": "1.0",
                            "join_ok": "1",
                        }
                    )
            shared = tmp_path / "shared.csv"
            out = tmp_path / "out"
            write_csv(shared, rows)

            metrics = run_audit(shared, out, min_joined_fraction=0.95, min_cell_count=3)

            self.assertEqual(metrics["verdict"], "joint-surface-supported")
            self.assertEqual(metrics["target_regime_count"], 3)
            self.assertEqual(metrics["evidence_state_count"], 2)
            self.assertEqual(metrics["cell_count"], 6)
            surface_path = out / "tables" / "target_regime_evidence_state_surface.csv"
            with surface_path.open(encoding="utf-8") as f:
                surface = list(csv.DictReader(f))
            self.assertEqual(len(surface), 6)
            deploy_path = out / "metrics" / "leakage_deployability_audit.json"
            deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
            self.assertEqual(deploy["target_regime"]["deployability"], "validation_only")

    def test_blocked_when_cell_coverage_is_too_small(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                {
                    "canonical_pair_id": "a",
                    "target_key": "one_target",
                    "state": "only_state",
                    "joint_error": "1.0",
                    "heading_abs_error": "1.0",
                    "range_abs_error": "0.1",
                    "join_ok": "1",
                }
            ]
            shared = tmp_path / "shared.csv"
            out = tmp_path / "out"
            write_csv(shared, rows)

            metrics = run_audit(shared, out, min_joined_fraction=0.95, min_cell_count=3)

            self.assertEqual(metrics["verdict"], "blocked-coverage")
            self.assertEqual(
                metrics["reason"], "insufficient_target_regime_or_evidence_state_coverage"
            )


if __name__ == "__main__":
    unittest.main()
