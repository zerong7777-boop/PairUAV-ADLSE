"""Selective parameter-freeze policies for bounded PairUAV probes."""

from __future__ import annotations

from typing import Iterable, Optional


def _last_decoder_count(policy: str) -> int:
    prefix = "pose_head_last_decoder"
    if not policy.startswith(prefix):
        return 0
    value = policy[len(prefix) :]
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid decoder-count policy: {policy}") from exc
    if count <= 0:
        raise ValueError(f"Decoder-count policy must be positive: {policy}")
    return count


def should_train_parameter(name: str, policy: str, dec_depth: Optional[int] = None) -> bool:
    policy = (policy or "").strip()
    if policy in {"", "all"}:
        return True
    if policy == "none":
        return False
    if policy == "angle_specialist":
        return "angle_specialist" in name
    if policy == "heading_residual":
        return (
            name.startswith("pose_head.heading_residual")
            or name.startswith("pose_head.heading_proj")
            or name.startswith("pose_head.heading_res_conv")
            or name.startswith("pose_head.heading_more_mlps")
            or name.startswith("pose_head.heading_layer_projs")
            or name.startswith("pose_head.heading_layer_res_convs")
            or name.startswith("pose_head.heading_layer_more_mlps")
            or name.startswith("pose_head.heading_fusion_mlp")
        )
    if policy == "pose_head":
        return name.startswith("pose_head.") or name.startswith("head.")
    if policy == "paaer_heading_only":
        return (
            name.startswith("pose_head.task_tokens")
            or name.startswith("pose_head.layer_embed")
            or name.startswith("pose_head.task_cross_attn")
            or name.startswith("pose_head.task_norm")
            or name.startswith("pose_head.task_ffn")
            or name.startswith("pose_head.heading_token_mlp")
            or name.startswith("pose_head.heading_layer_logits")
            or name.startswith("pose_head.fc_heading")
            or name.startswith("pose_head.phase104e")
        )
    if policy == "paaer_range_fc_only":
        return name.startswith("pose_head.fc_range")
    if policy == "paaer_range_path_only":
        return (
            name.startswith("pose_head.proj")
            or name.startswith("pose_head.res_conv")
            or name.startswith("pose_head.more_mlps")
            or name.startswith("pose_head.fc_range")
        )

    decoder_count = _last_decoder_count(policy)
    if decoder_count:
        if should_train_parameter(name, "pose_head", dec_depth=dec_depth):
            return True
        if name.startswith("dec_norm."):
            return True
        if dec_depth is None:
            return False
        first_trainable = max(0, int(dec_depth) - decoder_count)
        for idx in range(first_trainable, int(dec_depth)):
            if name.startswith(f"dec_blocks.{idx}."):
                return True
        return False

    raise ValueError(f"Unsupported trainable policy: {policy}")


def apply_trainable_policy(module, policy: str) -> dict:
    dec_depth = getattr(module, "dec_depth", None)
    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in module.named_parameters():
        total_params += int(param.numel())
        trainable = should_train_parameter(name, policy, dec_depth=dec_depth)
        param.requires_grad = bool(trainable)
        if trainable:
            trainable_params += int(param.numel())
            trainable_names.append(name)
    if trainable_params == 0:
        raise ValueError(f"Trainable policy {policy} selected zero parameters")
    return {
        "policy": policy,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_names": trainable_names,
    }
