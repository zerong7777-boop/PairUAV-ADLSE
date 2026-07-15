from __future__ import annotations

import types

import torch

from phase92.stage1_modes import output_modes
from reloc3r.pose_head import (
    MSRStaticEvidenceSplitPairUAVHead,
    MSRTwoBottleneckAllEvidencePairUAVHead,
    MSRTwoBottleneckStaticSplitPairUAVHead,
)


def _fake_net(dec_embed_dim: int = 32):
    return types.SimpleNamespace(
        patch_embed=types.SimpleNamespace(patch_size=(16,)),
        dec_embed_dim=dec_embed_dim,
    )


def _fake_decout(batch_size: int = 2, grid_size: int = 4, dec_embed_dim: int = 32):
    tokens = grid_size * grid_size
    return [
        torch.randn(batch_size, tokens, dec_embed_dim * 2),
        torch.randn(batch_size, tokens, dec_embed_dim),
        torch.randn(batch_size, tokens, dec_embed_dim),
        torch.randn(batch_size, tokens, dec_embed_dim),
    ]


def _assert_pairuav_contract(out):
    assert out["heading_vec"].shape == (2, 2)
    assert out["range_value"].shape == (2, 1)
    assert torch.isfinite(out["heading_vec"]).all()
    assert torch.isfinite(out["range_value"]).all()
    assert torch.allclose(out["heading_vec"].norm(dim=-1), torch.ones(2), atol=1e-5)


def test_stage1_mode_registry_lists_only_cde_modes():
    assert output_modes() == [
        "pairuav_msr_c_two_bottleneck_heading_range",
        "pairuav_msr_d_static_split_heading_range",
        "pairuav_msr_e_bottleneck_static_split_heading_range",
    ]


def test_msr_c_two_bottlenecks_all_evidence_forward_contract():
    head = MSRTwoBottleneckAllEvidencePairUAVHead(_fake_net(), msr_bottleneck_dim=8)
    out = head(_fake_decout(), img_shape=(64, 64))
    _assert_pairuav_contract(out)
    assert out["msr_heading_z_norm"].shape == (2,)
    assert out["msr_range_z_norm"].shape == (2,)


def test_msr_d_static_evidence_split_forward_contract():
    head = MSRStaticEvidenceSplitPairUAVHead(_fake_net())
    out = head(_fake_decout(), img_shape=(64, 64))
    _assert_pairuav_contract(out)
    assert out["msr_static_heading_feat_norm"].shape == (2,)
    assert out["msr_static_range_feat_norm"].shape == (2,)


def test_msr_e_bottleneck_static_split_forward_contract():
    head = MSRTwoBottleneckStaticSplitPairUAVHead(_fake_net(), msr_bottleneck_dim=8)
    out = head(_fake_decout(), img_shape=(64, 64))
    _assert_pairuav_contract(out)
    assert out["msr_heading_z_norm"].shape == (2,)
    assert out["msr_range_z_norm"].shape == (2,)
    assert out["msr_static_heading_feat_norm"].shape == (2,)
    assert out["msr_static_range_feat_norm"].shape == (2,)
