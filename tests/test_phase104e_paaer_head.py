import torch

from reloc3r.reloc3r_relpose import Reloc3rRelpose


def _small_model(output_mode="pairuav_phase104e_paaer_hard_heading_range"):
    return Reloc3rRelpose(
        img_size=64,
        patch_size=16,
        enc_embed_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        dec_embed_dim=64,
        dec_depth=3,
        dec_num_heads=4,
        output_mode=output_mode,
    )


def _synthetic_decout(model, batch_size=2, height=64, width=64):
    patch_size = model.pose_head.patch_size
    token_count = (height // patch_size) * (width // patch_size)
    return [
        torch.randn(batch_size, token_count, model.dec_embed_dim),
        torch.randn(batch_size, token_count, model.dec_embed_dim),
        torch.randn(batch_size, token_count, model.dec_embed_dim),
    ]


@torch.no_grad()
def test_phase104e_hard_outputs_real_base_and_protected_range():
    model = _small_model("pairuav_phase104e_paaer_hard_heading_range")
    model.eval()
    decout = _synthetic_decout(model)
    pred = model.pose_head(decout, img_shape=(64, 64))

    assert pred["heading_vec"].shape == (2, 2)
    assert pred["phase104e_heading_expert_vec"].shape == (2, 2)
    assert pred["phase104e_base_heading_vec"].shape == (2, 2)
    assert pred["range_value"].shape == (2, 1)
    assert pred["phase104e_protected_range_value"].shape == (2, 1)
    assert pred["phase104e_heading_expert_minus_base_delta_deg"].shape == (2,)
    assert torch.allclose(
        pred["range_value"],
        pred["phase104e_protected_range_value"],
        atol=0.0,
        rtol=0.0,
    )
    assert pred["phase104e_range_final_minus_protected_abs"].max().item() == 0.0


@torch.no_grad()
def test_phase104e_range_contract_uses_final_range_value():
    model = _small_model("pairuav_phase104e_paaer_hard_heading_range")
    decout = _synthetic_decout(model)
    pred = model.pose_head(decout, img_shape=(64, 64))

    recomputed = (
        pred["range_value"] - pred["phase104e_protected_range_value"].detach()
    ).abs()
    assert torch.allclose(pred["phase104e_range_final_minus_protected_abs"], recomputed)


def test_phase104e_heading_loss_does_not_backprop_to_fc_range():
    model = _small_model("pairuav_phase104e_paaer_hard_heading_range")
    model.train()
    decout = _synthetic_decout(model)
    pred = model.pose_head(decout, img_shape=(64, 64))

    target = torch.nn.functional.normalize(torch.randn_like(pred["heading_vec"]), dim=-1)
    loss = 1.0 - (pred["heading_vec"] * target).sum(dim=-1).mean()
    loss.backward()

    heading_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if (
            "pose_head.phase104e_heading" in name
            or "pose_head.phase104e_task_tokens" in name
            or "pose_head.phase104e_layer_embed" in name
        ):
            heading_grad_norm += float(param.grad.detach().abs().sum().item())

    assert heading_grad_norm > 0.0
    assert model.pose_head.fc_range.weight.grad is None
    assert model.pose_head.fc_range.bias.grad is None


@torch.no_grad()
def test_phase104e_blend_outputs_alpha_stats_inputs():
    model = _small_model("pairuav_phase104e_paaer_blend_heading_range")
    decout = _synthetic_decout(model)
    pred = model.pose_head(decout, img_shape=(64, 64))

    alpha = pred["phase104e_heading_blend_alpha"]
    assert alpha.shape == (2, 1)
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0
    assert pred["phase104e_heading_blend_alpha_std"].ndim == 0
    assert pred["phase104e_range_final_minus_protected_abs"].max().item() == 0.0


def test_paaer_heading_only_policy_freezes_range_and_backbone():
    model = _small_model("pairuav_phase104e_paaer_hard_heading_range")
    summary = model.freeze_for_trainable_policy("paaer_heading_only")
    trainable_names = set(summary["trainable_names"])

    assert any("pose_head.phase104e" in name for name in trainable_names)
    assert not any(name.startswith("pose_head.fc_range") for name in trainable_names)
    assert not any(name.startswith("enc_blocks.") for name in trainable_names)
    assert not any(name.startswith("dec_blocks.") for name in trainable_names)
