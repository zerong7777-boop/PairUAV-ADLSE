import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_pairuav_module():
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "datasets" / "pairuav.py"

    base_module = types.ModuleType("reloc3r.datasets.base.base_stereo_view_dataset")

    class BaseStereoViewDataset:
        def __init__(self, *args, **kwargs):
            pass

    base_module.BaseStereoViewDataset = BaseStereoViewDataset
    transforms_module = types.ModuleType("reloc3r.datasets.utils.transforms")
    transforms_module.ImgNorm = object
    image_module = types.ModuleType("reloc3r.utils.image")
    image_module.imread_cv2 = lambda path: None

    sys.modules["reloc3r.datasets.base.base_stereo_view_dataset"] = base_module
    sys.modules["reloc3r.datasets.utils.transforms"] = transforms_module
    sys.modules["reloc3r.utils.image"] = image_module

    spec = importlib.util.spec_from_file_location("pairuav_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pairuav = _load_pairuav_module()
_normalize_record = pairuav._normalize_record
_stable_target_index = pairuav._stable_target_index


class PairUAVTargetMetadataTest(unittest.TestCase):
    def test_stable_target_index_uses_digits_when_available(self):
        self.assertEqual(_stable_target_index("group_17", 4096), 17)
        self.assertEqual(_stable_target_index("17", 8), 1)

    def test_stable_target_index_hashes_non_numeric_values(self):
        first = _stable_target_index("alpha_scene", 4096)
        second = _stable_target_index("alpha_scene", 4096)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 4096)

    def test_normalize_record_adds_target_group_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "group_23"
            group_dir.mkdir()
            json_path = group_dir / "0001.json"
            json_path.write_text(
                json.dumps(
                    {
                        "group_id": "group_23",
                        "json_id": "0001",
                        "image_a": "a.jpg",
                        "image_b": "b.jpg",
                        "heading_deg": 12.0,
                        "range_value": 34.0,
                    }
                ),
                encoding="utf-8",
            )

            record = _normalize_record(json_path, require_labels=True, num_target_groups=4096)

        self.assertEqual(record["group_id"], "group_23")
        self.assertEqual(record["target_group_index"], 23)
        self.assertIsInstance(record["target_group_index"], np.integer)


if __name__ == "__main__":
    unittest.main()
