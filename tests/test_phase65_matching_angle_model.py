import torch

from reloc3r.phase65_matching_angle_model import (
    Phase65MatchingAngleBranch,
    angle_abs_error_deg,
    phase65_angle_loss,
)


def _batch(batch_size=3, topk=8):
    return {
        "tokens": torch.randn(batch_size, topk, 18),
        "token_mask": torch.ones(batch_size, topk),
        "hypothesis_features": torch.randn(batch_size, 3, 9),
        "global_stats": torch.randn(batch_size, 18),
        "rank1_heading": torch.tensor([10.0, -179.0, 170.0])[:batch_size],
        "rank1_distance": torch.tensor([1.0, -2.0, 3.0])[:batch_size],
        "target_heading": torch.tensor([10.5, 179.0, -170.0])[:batch_size],
    }


def test_phase65_forward_shapes_and_finiteness():
    batch = _batch()
    model = Phase65MatchingAngleBranch(hidden_dim=32, num_layers=1, num_heads=4, num_residual_candidates=3)
    out = model(
        batch["tokens"],
        batch["token_mask"],
        batch["hypothesis_features"],
        batch["global_stats"],
        batch["rank1_heading"],
        batch["rank1_distance"],
    )
    assert out["corrected_heading"].shape == (3,)
    assert out["candidate_headings"].shape == (3, 5)
    assert out["candidate_weights"].shape == (3, 5)
    assert out["residual_candidates"].shape == (3, 3)
    for key in ["corrected_heading", "candidate_headings", "candidate_weights", "residual_candidates"]:
        assert torch.isfinite(out[key]).all()


def test_phase65_initialization_is_rank1_parity():
    batch = _batch()
    model = Phase65MatchingAngleBranch(hidden_dim=32, num_layers=1, num_heads=4, num_residual_candidates=3)
    model.eval()
    with torch.no_grad():
        out = model(
            batch["tokens"],
            batch["token_mask"],
            batch["hypothesis_features"],
            batch["global_stats"],
            batch["rank1_heading"],
            batch["rank1_distance"],
        )
    assert torch.max(angle_abs_error_deg(out["corrected_heading"], batch["rank1_heading"])) < 1e-4
    assert torch.max(torch.abs(out["residual_candidates"])) < 1e-6
    assert torch.max(out["candidate_weights"][:, 1]) < 1e-8


def test_phase65_loss_is_finite_and_backward_works():
    batch = _batch()
    model = Phase65MatchingAngleBranch(hidden_dim=32, num_layers=1, num_heads=4, num_residual_candidates=3)
    out = model(
        batch["tokens"],
        batch["token_mask"],
        batch["hypothesis_features"],
        batch["global_stats"],
        batch["rank1_heading"],
        batch["rank1_distance"],
    )
    loss = phase65_angle_loss(out, batch["target_heading"])
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum())
    assert grad_norm > 0.0
