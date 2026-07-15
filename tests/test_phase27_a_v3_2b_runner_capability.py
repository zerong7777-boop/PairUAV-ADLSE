import tempfile
import unittest
from pathlib import Path

from scripts.phase27_a_v3_2b_runner_capability import audit_runner_capability


class RunnerCapabilityTests(unittest.TestCase):
    def test_detects_manifest_runner(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval.py"
            p.write_text("parser.add_argument('--manifest-csv')\n", encoding="utf-8")
            data = audit_runner_capability(d)
            self.assertEqual(data["decision"], "attempt_fixed_manifest_reacquisition")

    def test_ignores_tests_and_detects_reexport_only(self):
        with tempfile.TemporaryDirectory() as d:
            tests = Path(d) / "tests"
            tests.mkdir()
            (tests / "test_fake.py").write_text("parser.add_argument('--manifest-csv')\n", encoding="utf-8")
            (Path(d) / "infer_pairuav_with_progress.py").write_text("parser.add_argument('--json-root')\n", encoding="utf-8")
            data = audit_runner_capability(d)
            self.assertEqual(data["capability_status"], "reexport_only")
            self.assertEqual(data["decision"], "blocked_runner_interface")


if __name__ == "__main__":
    unittest.main()
