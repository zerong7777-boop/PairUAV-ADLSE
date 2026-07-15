import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_pose_head_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "pose_head.py"
    spec = importlib.util.spec_from_file_location("pose_head_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pose_head_under_test"] = module
    spec.loader.exec_module(module)
    return module


TargetConditionedPairUAVHead = _load_pose_head_module().TargetConditionedPairUAVHead


class _PatchEmbed:
    patch_size = (16, 16)


class _Net:
    patch_embed = _PatchEmbed()
    dec_embed_dim = 8


class TargetConditionedPairUAVHeadTest(unittest.TestCase):
    def test_forward_accepts_target_group_index(self):
        torch.manual_seed(7)
        head = TargetConditionedPairUAVHead(
            _Net(),
            num_resconv_block=1,
            num_target_groups=32,
            target_embed_dim=4,
        )
        decout = [torch.randn(2, 4, 8)]
        target_group_index = torch.tensor([1, 2], dtype=torch.long)
        out = head(decout, (32, 32), target_group_index=target_group_index)
        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))

    def test_missing_target_group_index_falls_back_to_zero_group(self):
        torch.manual_seed(7)
        head = TargetConditionedPairUAVHead(
            _Net(),
            num_resconv_block=1,
            num_target_groups=32,
            target_embed_dim=4,
        )
        decout = [torch.randn(2, 4, 8)]
        out = head(decout, (32, 32))
        self.assertEqual(out["heading_vec"].shape, (2, 2))
        self.assertEqual(out["range_value"].shape, (2, 1))

    def test_adapter_can_change_output_after_scale_is_enabled(self):
        torch.manual_seed(7)
        head = TargetConditionedPairUAVHead(
            _Net(),
            num_resconv_block=1,
            num_target_groups=32,
            target_embed_dim=4,
        )
        head.adapter_scale.data.fill_(1.0)
        decout = [torch.randn(2, 4, 8)]
        group_a = torch.tensor([1, 1], dtype=torch.long)
        group_b = torch.tensor([2, 2], dtype=torch.long)
        out_a = head(decout, (32, 32), target_group_index=group_a)
        out_b = head(decout, (32, 32), target_group_index=group_b)
        self.assertGreater(torch.max(torch.abs(out_a["range_value"] - out_b["range_value"])).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
