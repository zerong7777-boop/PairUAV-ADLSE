import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_deg_tensor(values):
    return torch.remainder(values + 180.0, 360.0) - 180.0


def angle_abs_error_deg(pred, target):
    return torch.abs(torch.remainder(pred - target + 180.0, 360.0) - 180.0)


def circular_weighted_mean_deg(candidate_headings, candidate_weights):
    radians = torch.deg2rad(candidate_headings)
    x = torch.sum(candidate_weights * torch.cos(radians), dim=-1)
    y = torch.sum(candidate_weights * torch.sin(radians), dim=-1)
    return wrap_deg_tensor(torch.rad2deg(torch.atan2(y, x)))


class Phase65MatchingAngleBranch(nn.Module):
    """Matching-aware angle branch with direct and residual candidates.

    Candidate 0 is the frozen rank1 heading. Candidate 1 is a direct heading
    source. Remaining candidates are learned residuals around rank1.
    """

    def __init__(
        self,
        token_dim=18,
        hypothesis_dim=9,
        global_dim=18,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_residual_candidates=4,
        dropout=0.1,
        max_residual_deg=5.0,
    ):
        super().__init__()
        self.max_residual_deg = float(max_residual_deg)
        self.num_residual_candidates = int(num_residual_candidates)
        self.num_candidates = self.num_residual_candidates + 2
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
        self.type_embedding = nn.Parameter(torch.zeros(2, int(hidden_dim)))
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
        self.candidate_queries = nn.Parameter(torch.randn(self.num_residual_candidates, int(hidden_dim)) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        candidate_context_dim = int(hidden_dim) + int(global_dim) + 3
        self.candidate_context = nn.Sequential(
            nn.Linear(candidate_context_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        pooled_context_dim = int(hidden_dim) + int(global_dim) + 3
        self.pooled_context = nn.Sequential(
            nn.Linear(pooled_context_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.residual_head = nn.Linear(int(hidden_dim), 1)
        self.direct_vec_head = nn.Linear(int(hidden_dim), 2)
        self.candidate_logit_head = nn.Linear(int(hidden_dim), self.num_candidates)
        self._init_rank1_parity()

    def _init_rank1_parity(self):
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.direct_vec_head.weight)
        nn.init.zeros_(self.direct_vec_head.bias)
        nn.init.zeros_(self.candidate_logit_head.weight)
        with torch.no_grad():
            self.candidate_logit_head.bias.zero_()
            self.candidate_logit_head.bias[0] = 4.0
            self.candidate_logit_head.bias[1] = -20.0
            self.candidate_logit_head.bias[2:] = 4.0

    def _rank_context(self, rank1_heading, rank1_distance):
        rank1_rad = torch.deg2rad(rank1_heading.view(-1, 1))
        return torch.cat(
            [
                torch.sin(rank1_rad),
                torch.cos(rank1_rad),
                rank1_distance.view(-1, 1),
            ],
            dim=-1,
        )

    def forward(self, tokens, token_mask, hypothesis_features, global_stats, rank1_heading, rank1_distance):
        batch_size = tokens.shape[0]
        token_embed = self.token_proj(tokens) + self.type_embedding[1].view(1, 1, -1)
        hypothesis_embed = self.hypothesis_proj(hypothesis_features) + self.type_embedding[0].view(1, 1, -1)
        sequence = torch.cat([hypothesis_embed, token_embed], dim=1)
        hypothesis_mask = torch.ones(
            batch_size,
            hypothesis_embed.shape[1],
            device=token_mask.device,
            dtype=token_mask.dtype,
        )
        full_mask = torch.cat([hypothesis_mask, token_mask], dim=1)
        key_padding_mask = full_mask <= 0.0
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)

        queries = self.candidate_queries.unsqueeze(0).expand(batch_size, -1, -1)
        candidate_contexts, candidate_attention = self.query_attention(
            queries,
            encoded,
            encoded,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        rank_context = self._rank_context(rank1_heading, rank1_distance)
        expanded_global = global_stats.unsqueeze(1).expand(-1, self.num_residual_candidates, -1)
        expanded_rank = rank_context.unsqueeze(1).expand(-1, self.num_residual_candidates, -1)
        candidate_contexts = self.candidate_context(
            torch.cat([candidate_contexts, expanded_global, expanded_rank], dim=-1)
        )
        residual_candidates = self.max_residual_deg * torch.tanh(self.residual_head(candidate_contexts).squeeze(-1))

        pooled = candidate_contexts.mean(dim=1)
        pooled = self.pooled_context(torch.cat([pooled, global_stats, rank_context], dim=-1))
        direct_vec = self.direct_vec_head(pooled)
        # Avoid the undefined gradient of atan2(0, 0) at rank1-parity init.
        direct_heading = wrap_deg_tensor(torch.rad2deg(torch.atan2(direct_vec[:, 1], direct_vec[:, 0] + 1e-6)))
        residual_headings = wrap_deg_tensor(rank1_heading.view(-1, 1) + residual_candidates)
        candidate_headings = torch.cat(
            [
                rank1_heading.view(-1, 1),
                direct_heading.view(-1, 1),
                residual_headings,
            ],
            dim=-1,
        )
        candidate_logits = self.candidate_logit_head(pooled)
        candidate_weights = torch.softmax(candidate_logits, dim=-1)
        corrected_heading = circular_weighted_mean_deg(candidate_headings, candidate_weights)
        residual = wrap_deg_tensor(corrected_heading - rank1_heading.view(-1))
        return {
            "corrected_heading": corrected_heading,
            "residual": residual,
            "direct_heading": direct_heading,
            "direct_vec": direct_vec,
            "residual_candidates": residual_candidates,
            "candidate_headings": candidate_headings,
            "candidate_logits": candidate_logits,
            "candidate_weights": candidate_weights,
            "candidate_attention": candidate_attention,
        }


def phase65_angle_loss(
    outputs,
    target_heading,
    *,
    main_weight=1.0,
    candidate_min_weight=0.25,
    entropy_weight=0.0,
):
    corrected_error = angle_abs_error_deg(outputs["corrected_heading"], target_heading.view(-1))
    main_loss = F.smooth_l1_loss(corrected_error, torch.zeros_like(corrected_error), reduction="mean")
    candidate_errors = angle_abs_error_deg(outputs["candidate_headings"], target_heading.view(-1, 1))
    min_candidate_loss = F.smooth_l1_loss(
        torch.min(candidate_errors, dim=-1).values,
        torch.zeros_like(corrected_error),
        reduction="mean",
    )
    weights = torch.clamp(outputs["candidate_weights"], min=1e-8)
    entropy = -torch.sum(weights * torch.log(weights), dim=-1).mean()
    return main_weight * main_loss + candidate_min_weight * min_candidate_loss - entropy_weight * entropy
