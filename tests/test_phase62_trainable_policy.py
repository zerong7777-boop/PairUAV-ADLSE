from __future__ import annotations

import torch

from reloc3r.trainable_policy import apply_trainable_policy, should_train_parameter


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dec_depth = 3
        self.pose_head = torch.nn.Linear(2, 2)
        self.dec_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
        self.dec_norm = torch.nn.LayerNorm(2)
        self.encoder = torch.nn.Linear(2, 2)


def trainable_names(module: torch.nn.Module) -> set[str]:
    return {name for name, param in module.named_parameters() if param.requires_grad}


def test_should_train_parameter_phase62_policies() -> None:
    assert should_train_parameter("pose_head.proj.weight", "pose_head", dec_depth=12)
    assert not should_train_parameter("dec_blocks.11.attn.weight", "pose_head", dec_depth=12)
    assert should_train_parameter("dec_blocks.11.attn.weight", "pose_head_last_decoder1", dec_depth=12)
    assert not should_train_parameter("dec_blocks.10.attn.weight", "pose_head_last_decoder1", dec_depth=12)
    assert should_train_parameter("dec_blocks.10.attn.weight", "pose_head_last_decoder2", dec_depth=12)
    assert should_train_parameter("dec_norm.weight", "pose_head_last_decoder1", dec_depth=12)


def test_apply_trainable_policy_pose_head_only() -> None:
    module = TinyModule()
    summary = apply_trainable_policy(module, "pose_head")
    names = trainable_names(module)
    assert summary["trainable_params"] > 0
    assert names == {"pose_head.weight", "pose_head.bias"}


def test_apply_trainable_policy_pose_head_last_decoder1() -> None:
    module = TinyModule()
    summary = apply_trainable_policy(module, "pose_head_last_decoder1")
    names = trainable_names(module)
    assert summary["trainable_params"] > 0
    assert "pose_head.weight" in names
    assert "dec_blocks.2.weight" in names
    assert "dec_blocks.1.weight" not in names
    assert "dec_norm.weight" in names
    assert "encoder.weight" not in names
