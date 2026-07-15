import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module(module_name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tokens_module = _load_module(
    "pairuav_correspondence_tokens_under_test",
    Path("reloc3r") / "datasets" / "pairuav_correspondence_tokens.py",
)

TOKEN_FEATURE_NAMES = tokens_module.TOKEN_FEATURE_NAMES
HYPOTHESIS_NAMES = tokens_module.HYPOTHESIS_NAMES
HYPOTHESIS_FEATURE_NAMES = tokens_module.HYPOTHESIS_FEATURE_NAMES
GLOBAL_FEATURE_NAMES = tokens_module.GLOBAL_FEATURE_NAMES
build_correspondence_token_packet = tokens_module.build_correspondence_token_packet
build_correspondence_token_manifest = tokens_module.build_correspondence_token_manifest
load_correspondence_token_manifest = tokens_module.load_correspondence_token_manifest


def _write_npz(path):
    keypoints0 = np.array(
        [
            [2.0, 2.0],
            [10.0, 2.0],
            [2.0, 10.0],
            [10.0, 10.0],
            [6.0, 6.0],
            [15.0, 4.0],
        ],
        dtype=np.float32,
    )
    shift = np.array([3.0, 1.0], dtype=np.float32)
    keypoints1 = keypoints0 + shift
    matches = np.array([0, 1, 2, 3, 4, -1], dtype=np.int64)
    confidence = np.array([0.5, 0.9, 0.8, 0.7, 0.6, 0.0], dtype=np.float32)
    np.savez(
        path,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        match_confidence=confidence,
    )


class CorrespondenceTokenPacketTest(unittest.TestCase):
    def test_packet_has_fixed_shapes_and_geometry_hypotheses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_matches.npz"
            _write_npz(path)

            packet = build_correspondence_token_packet(path, image_size=(20, 20), topk=4, grid_size=4)

            self.assertEqual(packet["tokens"].shape, (4, len(TOKEN_FEATURE_NAMES)))
            self.assertEqual(packet["token_mask"].shape, (4,))
            self.assertEqual(packet["hypothesis_features"].shape, (len(HYPOTHESIS_NAMES), len(HYPOTHESIS_FEATURE_NAMES)))
            self.assertEqual(packet["global_stats"].shape, (len(GLOBAL_FEATURE_NAMES),))
            self.assertEqual(float(packet["fallback_used"]), 0.0)
            self.assertEqual(float(packet["token_mask"].sum()), 4.0)
            self.assertTrue(np.isfinite(packet["tokens"]).all())
            self.assertLess(packet["hypothesis_features"][0, 6], 1e-5)
            self.assertLess(packet["hypothesis_features"][1, 6], 1e-5)
            self.assertLess(packet["hypothesis_features"][2, 6], 1e-5)

    def test_missing_packet_returns_fallback_shapes(self):
        packet = build_correspondence_token_packet("/missing/file.npz", topk=8)

        self.assertEqual(packet["tokens"].shape, (8, len(TOKEN_FEATURE_NAMES)))
        self.assertEqual(float(packet["fallback_used"]), 1.0)
        self.assertEqual(float(packet["token_mask"].sum()), 0.0)

    def test_manifest_builder_writes_loadable_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            match_dir = root / "cache" / "0001"
            match_dir.mkdir(parents=True)
            _write_npz(match_dir / "image-01_image-02_matches.npz")
            records = root / "records.csv"
            records.write_text("sample_id,split\n0001/01_02,train\n", encoding="utf-8")
            output = root / "tokens.jsonl"
            summary_path = root / "summary.json"

            summary = build_correspondence_token_manifest(
                records_csv=records,
                cache_root=root / "cache",
                output_jsonl=output,
                summary_json=summary_path,
                split="train",
                image_size=(20, 20),
                topk=4,
                grid_size=4,
            )
            loaded = load_correspondence_token_manifest(output)
            summary_json = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["covered"], 1)
            self.assertEqual(summary_json["topk"], 4)
            self.assertIn("0001/01_02", loaded)
            self.assertEqual(len(loaded["0001/01_02"]["tokens"]), 4)
            self.assertFalse(loaded["0001/01_02"]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
