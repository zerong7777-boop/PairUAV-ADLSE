import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_deg_tensor(values):
    return torch.remainder(values + 180.0, 360.0) - 180.0


class Phase64TokenAngleSpecialist(nn.Module):
    """Small token-set angle residual model for Phase64 smoke training."""

    def __init__(
        self,
        token_dim=18,
        hypothesis_dim=9,
        global_dim=18,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_residual_deg=0.30,
    ):
        super().__init__()
        self.max_residual_deg = float(max_residual_deg)
        self.token_proj = nn.Sequential(
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.hypothesis_proj = nn.Sequential(
            nn.Linear(int(hypothesis_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.pool_query = nn.Linear(int(hidden_dim), 1)
        context_dim = int(hidden_dim) + int(global_dim) + 3
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.residual_head = nn.Linear(int(hidden_dim), 1)
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        self._init_rank1_parity()

    def _init_rank1_parity(self):
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, tokens, token_mask, hypothesis_features, global_stats, rank1_heading, rank1_distance):
        token_embed = self.token_proj(tokens)
        hypothesis_embed = self.hypothesis_proj(hypothesis_features)
        sequence = torch.cat([hypothesis_embed, token_embed], dim=1)
        hypothesis_mask = torch.ones(
            token_mask.shape[0],
            hypothesis_embed.shape[1],
            device=token_mask.device,
            dtype=token_mask.dtype,
        )
        full_mask = torch.cat([hypothesis_mask, token_mask], dim=1)
        key_padding_mask = full_mask <= 0.0
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        attn_logits = self.pool_query(encoded).squeeze(-1)
        attn_logits = attn_logits.masked_fill(key_padding_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(encoded * attn.unsqueeze(-1), dim=1)

        rank1_rad = torch.deg2rad(rank1_heading.view(-1, 1))
        rank_context = torch.cat(
            [
                torch.sin(rank1_rad),
                torch.cos(rank1_rad),
                rank1_distance.view(-1, 1),
            ],
            dim=-1,
        )
        context = self.context_mlp(torch.cat([pooled, global_stats, rank_context], dim=-1))
        raw_residual = self.max_residual_deg * torch.tanh(self.residual_head(context)).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(context)).squeeze(-1)
        residual = gate * raw_residual
        corrected_heading = wrap_deg_tensor(rank1_heading.view(-1) + residual)
        return {
            "corrected_heading": corrected_heading,
            "residual": residual,
            "raw_residual": raw_residual,
            "gate": gate,
            "attention": attn,
        }


def phase64_angle_loss(outputs, residual_target, fallback_used=None):
    residual_error = outputs["residual"] - residual_target.view(-1)
    loss = F.smooth_l1_loss(residual_error, torch.zeros_like(residual_error), reduction="none")
    if fallback_used is not None:
        weights = 1.0 - torch.clamp(fallback_used.view(-1), 0.0, 1.0)
        loss = loss * weights
        denom = torch.clamp(weights.sum(), min=1.0)
        return loss.sum() / denom
    return loss.mean()
