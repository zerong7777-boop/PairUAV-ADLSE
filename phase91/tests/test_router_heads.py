from __future__ import annotations

import types

import torch

from reloc3r.pose_head import RouterLPairUAVHead, RouterQPairUAVHead


def _fake_net(dec_embed_dim: int = 32):
    return types.SimpleNamespace(
        patch_embed=types.SimpleNamespace(patch_size=(16,)),
        dec_embed_dim=dec_embed_dim,
    )


def _fake_decout(batch_size: int = 2, grid_size: int = 4, dec_embed_dim: int = 32):
    tokens = grid_size * grid_size
    # Include one non-decoder tensor to ensure the heads select decoder-dim layers.
    return [
        torch.randn(batch_size, tokens, dec_embed_dim * 2),
        torch.randn(batch_size, tokens, dec_embed_dim),
        torch.randn(batch_size, tokens, dec_embed_dim),
        torch.randn(batch_size, tokens, dec_embed_dim),
    ]


def test_router_l_outputs_axis_predictions_and_layer_weights():
    head = RouterLPairUAVHead(_fake_net())
    out = head(_fake_decout(), img_shape=(64, 64))

    assert out["heading_vec"].shape == (2, 2)
    assert out["range_value"].shape == (2, 1)
    assert out["router_l_heading_alpha"].shape == (2, 3)
    assert out["router_l_range_alpha"].shape == (2, 3)
    assert torch.allclose(out["router_l_heading_alpha"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(out["router_l_range_alpha"].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_router_q_outputs_axis_predictions_and_layer_attention_summary():
    head = RouterQPairUAVHead(_fake_net(), task_token_num_heads=4)
    out = head(_fake_decout(), img_shape=(64, 64))

    assert out["heading_vec"].shape == (2, 2)
    assert out["range_value"].shape == (2, 1)
    assert out["router_q_heading_layer_attention"].shape == (2, 3)
    assert out["router_q_range_layer_attention"].shape == (2, 3)
    assert torch.allclose(out["router_q_heading_layer_attention"].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out["router_q_range_layer_attention"].sum(dim=-1), torch.ones(2), atol=1e-5)
