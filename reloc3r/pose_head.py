import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pdb import set_trace as bb


# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


# parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
class PoseHead(nn.Module):
    """ 
    pose regression head
    """
    def __init__(self, 
                 net, 
                 num_resconv_block=2,
                 rot_representation='9D'):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.num_resconv_block = num_resconv_block
        self.rot_representation = rot_representation  

        output_dim = 4*self.patch_size**2

        self.proj = nn.Linear(net.dec_embed_dim, output_dim)
        self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(output_dim, output_dim)) 
            for _ in range(self.num_resconv_block)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(output_dim, 9)
        else:
            self.fc_rot = nn.Linear(output_dim, 6)
        
    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    
    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        else:
            out_r = self.rotation_6d_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,9]
        pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)
        res = {"pose": pose}

        return res


def compute_local_alignment_summary(decout, paired_decout=None):
    """
    Build a compact per-sample summary from cross-token cosine similarities.
    Missing paired tokens intentionally return zeros so old call sites remain valid.
    """
    tokens = decout[-1]
    B = tokens.shape[0]
    if paired_decout is None:
        return tokens.new_zeros(B, 8)

    paired_tokens = paired_decout[-1]
    if paired_tokens.shape[0] != B:
        raise ValueError(
            f"paired_decout batch size {paired_tokens.shape[0]} does not match decout batch size {B}"
        )

    q = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    k = F.normalize(paired_tokens.float(), dim=-1, eps=1e-6)
    sim = torch.matmul(q, k.transpose(-1, -2))

    row_max = sim.max(dim=-1).values
    col_max = sim.max(dim=-2).values
    summary = torch.stack(
        [
            sim.mean(dim=(-1, -2)),
            sim.std(dim=(-1, -2), unbiased=False),
            sim.amax(dim=(-1, -2)),
            sim.amin(dim=(-1, -2)),
            row_max.mean(dim=-1),
            row_max.std(dim=-1, unbiased=False),
            col_max.mean(dim=-1),
            col_max.std(dim=-1, unbiased=False),
        ],
        dim=-1,
    )
    return summary.to(dtype=tokens.dtype)


class PairUAVHead(nn.Module):
    """
    Minimal PairUAV heading/range regression head.
    Reuses the same token -> conv -> pooled feature path as PoseHead.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.num_resconv_block = num_resconv_block

        output_dim = 4 * self.patch_size ** 2
        self.proj = nn.Linear(net.dec_embed_dim, output_dim)
        self.res_conv = nn.ModuleList(
            [copy.deepcopy(ResConvBlock(output_dim, output_dim)) for _ in range(self.num_resconv_block)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_heading = nn.Linear(output_dim, 2)
        self.fc_range = nn.Linear(output_dim, 1)

    def _extract_features(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return self.more_mlps(feat)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        feat = self._extract_features(decout, img_shape)

        heading_vec = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class AxisDecoupledPairUAVHead(PairUAVHead):
    """
    Late-fork PairUAV head for an isolated axis-coupling causality probe.

    The base extractor and final fc layer names are inherited from PairUAVHead
    so existing PairUAV checkpoints can be loaded with only the new branch
    parameters missing. Zero branch scales make initial outputs exactly match
    the base shared head after partial checkpoint loading.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        axis_branch_hidden_dim=256,
        axis_branch_dropout=0.0,
        axis_branch_init_scale=0.0,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        hidden_dim = int(axis_branch_hidden_dim)
        dropout = float(axis_branch_dropout)
        self.heading_axis_adapter = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.range_axis_adapter = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.heading_axis_scale = nn.Parameter(torch.tensor(float(axis_branch_init_scale)))
        self.range_axis_scale = nn.Parameter(torch.tensor(float(axis_branch_init_scale)))

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        feat = self._extract_features(decout, img_shape)
        heading_delta = self.heading_axis_adapter(feat)
        range_delta = self.range_axis_adapter(feat)
        heading_feat = feat + torch.tanh(self.heading_axis_scale) * heading_delta
        range_feat = feat + torch.tanh(self.range_axis_scale) * range_delta

        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "axis_decoupled_heading_feat_norm": heading_feat.detach().norm(dim=-1),
            "axis_decoupled_range_feat_norm": range_feat.detach().norm(dim=-1),
            "axis_decoupled_heading_scale": torch.tanh(self.heading_axis_scale).view(1),
            "axis_decoupled_range_scale": torch.tanh(self.range_axis_scale).view(1),
        }

    def freeze_range_axis(self):
        """Keep range prediction on the base path while allowing heading branch training."""
        for param in self.range_axis_adapter.parameters():
            param.requires_grad = False
        self.range_axis_scale.requires_grad = False

    def freeze_heading_axis(self):
        """Symmetric helper for diagnostics."""
        for param in self.heading_axis_adapter.parameters():
            param.requires_grad = False
        self.heading_axis_scale.requires_grad = False


class MidSplitPairUAVHead(PairUAVHead):
    """
    Mid-split PairUAV head.

    Shares token projection and residual convolution with PairUAVHead, then
    forks the pooled feature into separate heading/range MLP trunks.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.heading_more_mlps = copy.deepcopy(self.more_mlps)
        self.range_more_mlps = copy.deepcopy(self.more_mlps)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        heading_feat = self.heading_more_mlps(feat)
        range_feat = self.range_more_mlps(feat)

        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class EarlySplitPairUAVHead(PairUAVHead):
    """
    Early-split PairUAV head.

    Heading and range use separate token projection, residual convolution, and
    MLP trunks from decoder-token features onward.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.heading_proj = copy.deepcopy(self.proj)
        self.range_proj = copy.deepcopy(self.proj)
        self.heading_res_conv = copy.deepcopy(self.res_conv)
        self.range_res_conv = copy.deepcopy(self.res_conv)
        self.heading_more_mlps = copy.deepcopy(self.more_mlps)
        self.range_more_mlps = copy.deepcopy(self.more_mlps)

    def _axis_features(self, decout, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._axis_features(
            decout, img_shape, self.heading_proj, self.heading_res_conv, self.heading_more_mlps
        )
        range_feat = self._axis_features(
            decout, img_shape, self.range_proj, self.range_res_conv, self.range_more_mlps
        )

        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class RangeH0HeadingH2PairUAVHead(PairUAVHead):
    """
    Axis-asymmetric head: keep range on the H0 shared trunk, but give heading
    its own H2-style MLP branch after shared projection/residual features.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.heading_more_mlps = copy.deepcopy(self.more_mlps)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        heading_feat = self.heading_more_mlps(feat)
        range_feat = self.more_mlps(feat)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class RangeH0HeadingH3PairUAVHead(PairUAVHead):
    """
    Axis-asymmetric head: keep range on the full H0 path, but give heading an
    H3-style independent projection/residual/MLP path from decoder tokens.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.heading_proj = copy.deepcopy(self.proj)
        self.heading_res_conv = copy.deepcopy(self.res_conv)
        self.heading_more_mlps = copy.deepcopy(self.more_mlps)

    def _heading_features(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        feat = self.heading_proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.heading_res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return self.heading_more_mlps(feat)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class RangeH0HeadingEarlyMidLatePairUAVHead(PairUAVHead):
    """
    H8: keep range on the H0 last-layer path, while heading reads early/mid/late
    decoder layers with independent token-to-grid extractors and a fusion MLP.
    """

    def __init__(self, net, num_resconv_block=2):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.heading_layer_projs = nn.ModuleList([copy.deepcopy(self.proj) for _ in range(3)])
        self.heading_layer_res_convs = nn.ModuleList([copy.deepcopy(self.res_conv) for _ in range(3)])
        self.heading_layer_more_mlps = nn.ModuleList([copy.deepcopy(self.more_mlps) for _ in range(3)])
        self.heading_fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def _select_early_mid_late(self, decout):
        decoder_dim = decout[-1].shape[-1]
        decoder_layers = [tokens for tokens in decout if tokens.shape[-1] == decoder_dim]
        n_layers = len(decoder_layers)
        if n_layers == 0:
            raise ValueError("decout must contain at least one decoder-dim layer")
        return [decoder_layers[0], decoder_layers[n_layers // 2], decoder_layers[-1]]

    def _layer_feature(self, tokens, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def _heading_features(self, decout, img_shape):
        layer_tokens = self._select_early_mid_late(decout)
        layer_feats = [
            self._layer_feature(tokens, img_shape, proj, res_conv, more_mlps)
            for tokens, proj, res_conv, more_mlps in zip(
                layer_tokens,
                self.heading_layer_projs,
                self.heading_layer_res_convs,
                self.heading_layer_more_mlps,
            )
        ]
        return self.heading_fusion_mlp(torch.cat(layer_feats, dim=-1))

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }



class RangeH0HeadingSelectableReadoutPairUAVHead(PairUAVHead):
    """
    Phase88 H8 ablation head.

    Range keeps the H0 final-layer path. Heading reads a selected subset of
    decoder-dim layers using independent token-to-grid extractors and fuses
    their pooled features. This isolates whether H8 needs early/mid/late
    evidence or only extra heading capacity.
    """

    _VALID_LAYER_NAMES = ("early", "mid", "late")

    def __init__(self, net, heading_readout_layers=("late",), num_resconv_block=2, heading_fusion_hidden_dim=None):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        selected = tuple(heading_readout_layers)
        if len(selected) == 0:
            raise ValueError("heading_readout_layers must contain at least one layer name")
        invalid = [name for name in selected if name not in self._VALID_LAYER_NAMES]
        if invalid:
            raise ValueError(f"Unsupported heading_readout_layers={selected}")

        output_dim = 4 * self.patch_size ** 2
        self.heading_readout_layers = selected
        self.heading_layer_projs = nn.ModuleList([copy.deepcopy(self.proj) for _ in selected])
        self.heading_layer_res_convs = nn.ModuleList([copy.deepcopy(self.res_conv) for _ in selected])
        self.heading_layer_more_mlps = nn.ModuleList([copy.deepcopy(self.more_mlps) for _ in selected])
        hidden_dim = int(heading_fusion_hidden_dim or output_dim)
        self.heading_fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * len(selected), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def _select_named_layers(self, decout):
        decoder_dim = decout[-1].shape[-1]
        decoder_layers = [tokens for tokens in decout if tokens.shape[-1] == decoder_dim]
        n_layers = len(decoder_layers)
        if n_layers == 0:
            raise ValueError("decout must contain at least one decoder-dim layer")
        candidates = {
            "early": decoder_layers[0],
            "mid": decoder_layers[n_layers // 2],
            "late": decoder_layers[-1],
        }
        return [candidates[name] for name in self.heading_readout_layers]

    def _layer_feature(self, tokens, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def _heading_features(self, decout, img_shape):
        layer_tokens = self._select_named_layers(decout)
        layer_feats = [
            self._layer_feature(tokens, img_shape, proj, res_conv, more_mlps)
            for tokens, proj, res_conv, more_mlps in zip(
                layer_tokens,
                self.heading_layer_projs,
                self.heading_layer_res_convs,
                self.heading_layer_more_mlps,
            )
        ]
        return self.heading_fusion_mlp(torch.cat(layer_feats, dim=-1))

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }



def _phase104_select_named_decoder_layers(decout):
    decoder_dim = decout[-1].shape[-1]
    decoder_layers = [tokens for tokens in decout if tokens.shape[-1] == decoder_dim]
    n_layers = len(decoder_layers)
    if n_layers == 0:
        raise ValueError("decout must contain at least one decoder-dim layer")
    return {
        "early": decoder_layers[0],
        "mid": decoder_layers[n_layers // 2],
        "late": decoder_layers[-1],
    }


def _phase104_entropy(prob, dim=-1):
    prob = prob.clamp_min(1e-8)
    return -(prob * prob.log()).sum(dim=dim)


class AxisAsyncQueryBridgePairUAVHead(PairUAVHead):
    """
    Phase104 query-bridge head.

    Heading and range use factor-specific task queries to read an
    early/mid/late decoder token bank. Fixed masks test the H8 evidence-depth
    hypothesis; optional learnable layer weights and gated factor bridges test
    whether routing and controlled exchange improve over fixed H8 readout.
    """

    _LAYER_NAMES = ("early", "mid", "late")

    def __init__(
        self,
        net,
        num_resconv_block=2,
        heading_layers=("mid", "late"),
        range_layers=("late",),
        learnable_layer_weights=False,
        use_gated_bridge=False,
        task_token_num_heads=8,
        bridge_hidden_dim=128,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.heading_layers = tuple(heading_layers)
        self.range_layers = tuple(range_layers)
        self.learnable_layer_weights = bool(learnable_layer_weights)
        self.use_gated_bridge = bool(use_gated_bridge)
        invalid = [
            name
            for name in self.heading_layers + self.range_layers
            if name not in self._LAYER_NAMES
        ]
        if invalid:
            raise ValueError(f"Unsupported Phase104 layer names: {invalid}")

        output_dim = 4 * self.patch_size ** 2
        token_dim = net.dec_embed_dim
        num_heads = int(task_token_num_heads)
        if token_dim % num_heads != 0:
            num_heads = 1

        self.task_tokens = nn.Parameter(torch.zeros(2, token_dim))
        nn.init.normal_(self.task_tokens, std=0.02)
        self.layer_embed = nn.Parameter(torch.zeros(len(self._LAYER_NAMES), token_dim))
        nn.init.normal_(self.layer_embed, std=0.02)
        self.task_cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.task_norm = nn.LayerNorm(token_dim)
        self.task_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.heading_token_mlp = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.range_token_mlp = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.heading_layer_logits = nn.Parameter(torch.zeros(len(self._LAYER_NAMES)))
        self.range_layer_logits = nn.Parameter(torch.zeros(len(self._LAYER_NAMES)))

        hidden_dim = int(bridge_hidden_dim)
        self.hr_gate = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rh_gate = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hr_bridge = nn.Linear(output_dim, output_dim)
        self.rh_bridge = nn.Linear(output_dim, output_dim)

    def _mask_vector(self, names, device, dtype):
        mask = torch.zeros(len(self._LAYER_NAMES), device=device, dtype=dtype)
        for name in names:
            mask[self._LAYER_NAMES.index(name)] = 1.0
        return mask

    def _layer_weight(self, logits, names, device, dtype):
        if self.learnable_layer_weights:
            return torch.softmax(logits.to(device=device, dtype=dtype), dim=0)
        mask = self._mask_vector(names, device=device, dtype=dtype)
        return mask / mask.sum().clamp_min(1.0)

    def _token_bank(self, decout, factor_names, layer_weight):
        candidates = _phase104_select_named_decoder_layers(decout)
        bank_parts = []
        layer_sizes = []
        layer_indices = []
        for idx, name in enumerate(self._LAYER_NAMES):
            include = self.learnable_layer_weights or (name in factor_names)
            if not include:
                continue
            tokens = candidates[name] + self.layer_embed[idx].view(1, 1, -1)
            tokens = tokens * layer_weight[idx].view(1, 1, 1)
            bank_parts.append(tokens)
            layer_sizes.append(tokens.shape[1])
            layer_indices.append(idx)
        return torch.cat(bank_parts, dim=1), layer_sizes, layer_indices

    def _query_one_factor(self, decout, factor_index, factor_names, layer_logits):
        device = decout[-1].device
        dtype = decout[-1].dtype
        layer_weight = self._layer_weight(layer_logits, factor_names, device=device, dtype=dtype)
        token_bank, layer_sizes, layer_indices = self._token_bank(decout, factor_names, layer_weight)
        B = token_bank.shape[0]
        task_token = self.task_tokens[factor_index : factor_index + 1].unsqueeze(0).expand(B, -1, -1)
        attended, attn = self.task_cross_attn(
            task_token,
            token_bank,
            token_bank,
            need_weights=True,
            average_attn_weights=True,
        )
        task_token = self.task_norm(task_token + attended)
        task_token = self.task_norm(task_token + self.task_ffn(task_token))

        token_prob = attn[:, 0]
        layer_attn_full = token_bank.new_zeros(B, len(self._LAYER_NAMES))
        start = 0
        for size, layer_idx in zip(layer_sizes, layer_indices):
            layer_attn_full[:, layer_idx] = token_prob[:, start : start + size].sum(dim=-1)
            start += size
        layer_attn_full = layer_attn_full / layer_attn_full.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = _phase104_entropy(token_prob, dim=-1)
        return task_token[:, 0], layer_attn_full.detach(), entropy.detach()

    def _task_features(self, decout):
        heading_task, heading_attn, heading_entropy = self._query_one_factor(
            decout, 0, self.heading_layers, self.heading_layer_logits
        )
        range_task, range_attn, range_entropy = self._query_one_factor(
            decout, 1, self.range_layers, self.range_layer_logits
        )
        heading_feat = self.heading_token_mlp(heading_task)
        range_feat = self.range_token_mlp(range_task)
        return heading_feat, range_feat, heading_attn, range_attn, heading_entropy, range_entropy

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        del img_shape
        heading_feat, range_feat, heading_attn, range_attn, heading_entropy, range_entropy = self._task_features(decout)
        out = {
            "phase104_heading_layer_attention": heading_attn,
            "phase104_range_layer_attention": range_attn,
            "phase104_heading_token_entropy": heading_entropy,
            "phase104_range_token_entropy": range_entropy,
        }
        if self.use_gated_bridge:
            source_heading_feat = heading_feat
            source_range_feat = range_feat
            hr_gate = torch.sigmoid(self.hr_gate(torch.cat([source_heading_feat, source_range_feat], dim=-1)))
            rh_gate = torch.sigmoid(self.rh_gate(torch.cat([source_range_feat, source_heading_feat], dim=-1)))
            heading_feat = source_heading_feat + hr_gate * self.hr_bridge(source_range_feat)
            range_feat = source_range_feat + rh_gate * self.rh_bridge(source_heading_feat)
            out["phase104_bridge_hr_gate"] = hr_gate.detach()
            out["phase104_bridge_rh_gate"] = rh_gate.detach()
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        out.update(
            {
                "heading_vec": heading_vec,
                "range_value": range_value,
            }
        )
        return out


class RangeAnchoredHeadingResidualPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    """
    Phase104-C2: H8-mid-late base heading with detached range evidence as a
    stable metric anchor for a bounded circular heading residual.
    """

    def __init__(self, net, num_resconv_block=2, heading_residual_max_delta_deg=1.0):
        super().__init__(
            net=net,
            heading_readout_layers=("mid", "late"),
            num_resconv_block=num_resconv_block,
        )
        output_dim = 4 * self.patch_size ** 2
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))
        self.range_anchor_residual_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )
        self.reset_heading_residual_parameters()

    def reset_heading_residual_parameters(self):
        last = self.range_anchor_residual_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def reset_after_model_init(self):
        self.reset_heading_residual_parameters()

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)

        residual_input = torch.cat([heading_feat, range_feat.detach()], dim=-1)
        delta_rad = self.heading_residual_max_delta_rad * torch.tanh(
            self.range_anchor_residual_mlp(residual_input)
        )
        heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "phase104_heading_base_vec": base_heading_vec.detach(),
            "phase104_heading_residual_delta_rad": delta_rad,
            "phase104_heading_residual_delta_deg": delta_rad * (180.0 / math.pi),
            "phase104_range_anchor_norm": range_feat.detach().norm(dim=-1),
        }


class Phase104bFDERPairUAVHead(PairUAVHead):
    """
    Factor-Dependent Evidence Regime Router.

    Range stays on the protected H0/late path. Heading starts from an H8-style
    mid-late base and applies a zero-initialized, range-anchored circular
    residual whose evidence is selected by a per-sample factor router.
    """

    _LAYER_NAMES = ("early", "mid", "late")

    def __init__(
        self,
        net,
        num_resconv_block=2,
        use_sample_router=True,
        heading_residual_max_delta_deg=1.0,
        router_hidden_dim=256,
        task_token_num_heads=8,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.use_sample_router = bool(use_sample_router)
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))

        output_dim = 4 * self.patch_size ** 2
        token_dim = net.dec_embed_dim
        num_heads = int(task_token_num_heads)
        if token_dim % num_heads != 0:
            num_heads = 1

        self.heading_layer_projs = nn.ModuleList([copy.deepcopy(self.proj) for _ in self._LAYER_NAMES])
        self.heading_layer_res_convs = nn.ModuleList([copy.deepcopy(self.res_conv) for _ in self._LAYER_NAMES])
        self.heading_layer_more_mlps = nn.ModuleList([copy.deepcopy(self.more_mlps) for _ in self._LAYER_NAMES])
        self.heading_base_fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

        self.layer_summary_norm = nn.LayerNorm(token_dim)
        self.heading_factor_embed = nn.Parameter(torch.zeros(token_dim))
        self.range_factor_embed = nn.Parameter(torch.zeros(token_dim))
        nn.init.normal_(self.heading_factor_embed, std=0.02)
        nn.init.normal_(self.range_factor_embed, std=0.02)

        router_input_dim = token_dim * 4 + output_dim * 2
        hidden_dim = int(router_hidden_dim)
        self.heading_router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self._LAYER_NAMES)),
        )
        self.range_diag_router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self._LAYER_NAMES)),
        )

        self.register_buffer("fixed_heading_layer_logits", torch.tensor([-8.0, 0.0, 0.0]))
        self.register_buffer("fixed_range_layer_logits", torch.tensor([-8.0, -8.0, 0.0]))
        self._reset_router_biases()

        self.layer_embed = nn.Parameter(torch.zeros(len(self._LAYER_NAMES), token_dim))
        nn.init.normal_(self.layer_embed, std=0.02)
        self.heading_query = nn.Parameter(torch.zeros(1, token_dim))
        nn.init.normal_(self.heading_query, std=0.02)
        self.heading_cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.heading_task_norm = nn.LayerNorm(token_dim)
        self.heading_task_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.heading_evidence_mlp = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.heading_gate = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.heading_residual_mlp = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )
        self.reset_heading_residual_parameters()

    def _reset_router_biases(self):
        nn.init.zeros_(self.heading_router[-1].weight)
        self.heading_router[-1].bias.data.copy_(torch.tensor([-2.0, 1.0, 1.0]))
        nn.init.zeros_(self.range_diag_router[-1].weight)
        self.range_diag_router[-1].bias.data.copy_(torch.tensor([-3.0, -1.0, 2.0]))

    def reset_heading_residual_parameters(self):
        last = self.heading_residual_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def reset_after_model_init(self):
        self.reset_heading_residual_parameters()

    def _select_named_layers(self, decout):
        candidates = _phase104_select_named_decoder_layers(decout)
        return [candidates[name] for name in self._LAYER_NAMES]

    def _layer_feature(self, tokens, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def _base_heading_features(self, layer_features):
        return self.heading_base_fusion_mlp(torch.cat([layer_features[1], layer_features[2]], dim=-1))

    def _layer_summaries(self, layer_tokens):
        return torch.cat([self.layer_summary_norm(tokens.mean(dim=1)) for tokens in layer_tokens], dim=-1)

    def _router_input(self, layer_tokens, base_heading_feat, range_feat, factor_embed):
        B = base_heading_feat.shape[0]
        factor = factor_embed.view(1, -1).expand(B, -1)
        return torch.cat(
            [
                self._layer_summaries(layer_tokens),
                base_heading_feat,
                range_feat.detach(),
                factor,
            ],
            dim=-1,
        )

    def _layer_weights(self, layer_tokens, base_heading_feat, range_feat):
        B = base_heading_feat.shape[0]
        dtype = base_heading_feat.dtype
        device = base_heading_feat.device
        if self.use_sample_router:
            heading_logits = self.heading_router(
                self._router_input(layer_tokens, base_heading_feat, range_feat, self.heading_factor_embed)
            )
            range_logits = self.range_diag_router(
                self._router_input(layer_tokens, base_heading_feat, range_feat, self.range_factor_embed)
            )
        else:
            heading_logits = self.fixed_heading_layer_logits.to(device=device, dtype=dtype).view(1, -1).expand(B, -1)
            range_logits = self.fixed_range_layer_logits.to(device=device, dtype=dtype).view(1, -1).expand(B, -1)
        return torch.softmax(heading_logits, dim=-1), torch.softmax(range_logits, dim=-1)

    def _query_layer_evidence(self, layer_tokens):
        evidence = []
        attn_entropy = []
        for idx, tokens in enumerate(layer_tokens):
            B = tokens.shape[0]
            token_bank = tokens + self.layer_embed[idx].view(1, 1, -1)
            query = self.heading_query.view(1, 1, -1).expand(B, -1, -1)
            attended, attn = self.heading_cross_attn(
                query,
                token_bank,
                token_bank,
                need_weights=True,
                average_attn_weights=True,
            )
            task = self.heading_task_norm(query + attended)
            task = self.heading_task_norm(task + self.heading_task_ffn(task))
            evidence.append(self.heading_evidence_mlp(task[:, 0]))
            attn_entropy.append(_phase104_entropy(attn[:, 0], dim=-1))
        return torch.stack(evidence, dim=1), torch.stack(attn_entropy, dim=1)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        layer_tokens = self._select_named_layers(decout)
        layer_features = [
            self._layer_feature(tokens, img_shape, proj, res_conv, more_mlps)
            for tokens, proj, res_conv, more_mlps in zip(
                layer_tokens,
                self.heading_layer_projs,
                self.heading_layer_res_convs,
                self.heading_layer_more_mlps,
            )
        ]
        base_heading_feat = self._base_heading_features(layer_features)
        range_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_heading_feat), dim=-1, eps=1e-6)
        protected_range_value = self.fc_range(range_feat)

        heading_weight, range_diag_weight = self._layer_weights(layer_tokens, base_heading_feat, range_feat)
        layer_evidence, token_attention_entropy = self._query_layer_evidence(layer_tokens)
        heading_evidence = (heading_weight.unsqueeze(-1) * layer_evidence).sum(dim=1)
        residual_input = torch.cat([heading_evidence, base_heading_feat, range_feat.detach()], dim=-1)
        heading_gate = torch.sigmoid(self.heading_gate(residual_input))
        delta_rad = self.heading_residual_max_delta_rad * heading_gate * torch.tanh(
            self.heading_residual_mlp(residual_input)
        )
        heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)

        per_layer_norm = layer_evidence.detach().norm(dim=-1)
        per_layer_contribution = heading_weight.detach() * per_layer_norm
        return {
            "heading_vec": heading_vec,
            "range_value": protected_range_value,
            "phase104b_heading_base_vec": base_heading_vec.detach(),
            "phase104b_protected_range_value": protected_range_value.detach(),
            "phase104b_heading_layer_weights": heading_weight.detach(),
            "phase104b_range_diag_layer_weights": range_diag_weight.detach(),
            "phase104b_heading_router_entropy": _phase104_entropy(heading_weight, dim=-1).detach(),
            "phase104b_range_router_entropy": _phase104_entropy(range_diag_weight, dim=-1).detach(),
            "phase104b_heading_gate": heading_gate.detach(),
            "phase104b_heading_residual_delta_rad": delta_rad,
            "phase104b_heading_residual_delta_deg": delta_rad * (180.0 / math.pi),
            "phase104b_per_layer_heading_evidence_norm": per_layer_norm,
            "phase104b_per_layer_contribution": per_layer_contribution,
            "phase104b_token_attention_entropy": token_attention_entropy.detach(),
        }


class Phase104cObservabilityFactorRouterHead(PairUAVHead):
    """
    Phase104c / FDER-v2 OFFER-lite head.

    Range stays on the inherited H0/late protected path. Heading starts from an
    H8-style mid-late base and applies a bounded circular residual selected
    from typed evidence probes. Dynamic routing is conditioned on factor
    observability; the fixed mode is a typed-evidence control.
    """

    _LAYER_NAMES = ("early", "mid", "late")
    _EVIDENCE_NAMES = ("relational", "metric", "shared", "ambiguity")

    def __init__(
        self,
        net,
        num_resconv_block=2,
        use_observability_router=True,
        heading_residual_max_delta_deg=5.0,
        router_hidden_dim=256,
        task_token_num_heads=8,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.use_observability_router = bool(use_observability_router)
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))

        output_dim = 4 * self.patch_size ** 2
        token_dim = net.dec_embed_dim
        hidden_dim = int(router_hidden_dim)
        num_heads = int(task_token_num_heads)
        if token_dim % num_heads != 0:
            num_heads = 1

        self.heading_layer_projs = nn.ModuleList([copy.deepcopy(self.proj) for _ in self._LAYER_NAMES])
        self.heading_layer_res_convs = nn.ModuleList([copy.deepcopy(self.res_conv) for _ in self._LAYER_NAMES])
        self.heading_layer_more_mlps = nn.ModuleList([copy.deepcopy(self.more_mlps) for _ in self._LAYER_NAMES])
        self.heading_base_fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

        self.layer_embed = nn.Parameter(torch.zeros(len(self._LAYER_NAMES), token_dim))
        self.typed_queries = nn.Parameter(torch.zeros(len(self._EVIDENCE_NAMES), token_dim))
        nn.init.normal_(self.layer_embed, std=0.02)
        nn.init.normal_(self.typed_queries, std=0.02)
        self.evidence_cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.evidence_norm = nn.LayerNorm(token_dim)
        self.evidence_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.typed_evidence_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                )
                for _ in self._EVIDENCE_NAMES
            ]
        )
        self.ambiguity_feature_mlp = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

        observability_input_dim = output_dim * 5
        self.observability_head = nn.Sequential(
            nn.Linear(observability_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        router_input_dim = output_dim * (2 + len(self._EVIDENCE_NAMES)) + 2
        self.heading_evidence_router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self._EVIDENCE_NAMES)),
        )
        self.range_diag_evidence_router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self._EVIDENCE_NAMES)),
        )
        self.register_buffer("fixed_heading_evidence_logits", torch.tensor([1.5, -1.0, 1.0, -0.5]))
        self.register_buffer("fixed_range_diag_evidence_logits", torch.tensor([-1.0, 1.5, 0.5, -0.5]))

        residual_input_dim = output_dim * 3 + 1
        self.heading_gate = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.heading_residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )
        self.reset_phase104c_parameters()

    def reset_phase104c_parameters(self):
        nn.init.zeros_(self.heading_evidence_router[-1].weight)
        self.heading_evidence_router[-1].bias.data.copy_(torch.tensor([1.2, -0.8, 0.8, -0.2]))
        nn.init.zeros_(self.range_diag_evidence_router[-1].weight)
        self.range_diag_evidence_router[-1].bias.data.copy_(torch.tensor([-0.8, 1.2, 0.4, -0.2]))

        gate_last = self.heading_gate[-1]
        nn.init.normal_(gate_last.weight, mean=0.0, std=1e-4)
        gate_last.bias.data.fill_(math.log(0.15 / 0.85))

        residual_last = self.heading_residual_mlp[-1]
        nn.init.normal_(residual_last.weight, mean=0.0, std=1e-4)
        nn.init.normal_(residual_last.bias, mean=0.0, std=1e-4)

    def reset_after_model_init(self):
        self.reset_phase104c_parameters()

    def _select_named_layers(self, decout):
        candidates = _phase104_select_named_decoder_layers(decout)
        return [candidates[name] for name in self._LAYER_NAMES]

    def _layer_feature(self, tokens, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def _base_heading_features(self, layer_features):
        return self.heading_base_fusion_mlp(torch.cat([layer_features[1], layer_features[2]], dim=-1))

    def _build_token_bank(self, layer_tokens):
        parts = []
        layer_sizes = []
        for idx, tokens in enumerate(layer_tokens):
            parts.append(tokens + self.layer_embed[idx].view(1, 1, -1))
            layer_sizes.append(tokens.shape[1])
        return torch.cat(parts, dim=1), layer_sizes

    def _query_typed_tokens(self, layer_tokens):
        token_bank, layer_sizes = self._build_token_bank(layer_tokens)
        B = token_bank.shape[0]
        query = self.typed_queries.unsqueeze(0).expand(B, -1, -1)
        attended, attn = self.evidence_cross_attn(
            query,
            token_bank,
            token_bank,
            need_weights=True,
            average_attn_weights=True,
        )
        typed = self.evidence_norm(query + attended)
        typed = self.evidence_norm(typed + self.evidence_ffn(typed))
        layer_attention = token_bank.new_zeros(B, len(self._EVIDENCE_NAMES), len(self._LAYER_NAMES))
        start = 0
        for layer_idx, size in enumerate(layer_sizes):
            layer_attention[:, :, layer_idx] = attn[:, :, start : start + size].sum(dim=-1)
            start += size
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence_attention_entropy = _phase104_entropy(attn, dim=-1)
        return typed, layer_attention, evidence_attention_entropy

    def _typed_evidence(self, layer_tokens, layer_features):
        typed_tokens, layer_attention, evidence_attention_entropy = self._query_typed_tokens(layer_tokens)
        evidence_parts = [
            mlp(typed_tokens[:, idx])
            for idx, mlp in enumerate(self.typed_evidence_mlps)
        ]
        layer_stack = torch.stack(layer_features, dim=1)
        layer_std = layer_stack.std(dim=1, unbiased=False)
        mid_late_gap = (layer_features[1] - layer_features[2]).abs()
        early_late_gap = (layer_features[0] - layer_features[2]).abs()
        ambiguity_from_layers = self.ambiguity_feature_mlp(
            torch.cat([layer_std, mid_late_gap, early_late_gap], dim=-1)
        )
        evidence_parts[3] = evidence_parts[3] + ambiguity_from_layers
        return torch.stack(evidence_parts, dim=1), layer_attention, evidence_attention_entropy

    def _observability(self, base_heading_feat, range_feat, typed_evidence):
        rel = typed_evidence[:, 0]
        metric = typed_evidence[:, 1]
        ambiguity = typed_evidence[:, 3]
        obs_input = torch.cat(
            [
                base_heading_feat,
                range_feat.detach(),
                rel,
                metric,
                ambiguity,
            ],
            dim=-1,
        )
        obs = torch.sigmoid(self.observability_head(obs_input))
        return obs[:, 0:1], obs[:, 1:2]

    def _router_weights(self, base_heading_feat, range_feat, typed_evidence, heading_observability, range_observability):
        B = base_heading_feat.shape[0]
        dtype = base_heading_feat.dtype
        device = base_heading_feat.device
        if self.use_observability_router:
            router_input = torch.cat(
                [
                    base_heading_feat,
                    range_feat.detach(),
                    typed_evidence.flatten(start_dim=1),
                    heading_observability,
                    range_observability,
                ],
                dim=-1,
            )
            heading_logits = self.heading_evidence_router(router_input)
            range_logits = self.range_diag_evidence_router(router_input)
        else:
            heading_logits = self.fixed_heading_evidence_logits.to(device=device, dtype=dtype).view(1, -1).expand(B, -1)
            range_logits = self.fixed_range_diag_evidence_logits.to(device=device, dtype=dtype).view(1, -1).expand(B, -1)
        return torch.softmax(heading_logits, dim=-1), torch.softmax(range_logits, dim=-1)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        layer_tokens = self._select_named_layers(decout)
        layer_features = [
            self._layer_feature(tokens, img_shape, proj, res_conv, more_mlps)
            for tokens, proj, res_conv, more_mlps in zip(
                layer_tokens,
                self.heading_layer_projs,
                self.heading_layer_res_convs,
                self.heading_layer_more_mlps,
            )
        ]
        base_heading_feat = self._base_heading_features(layer_features)
        range_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_heading_feat), dim=-1, eps=1e-6)
        protected_range_value = self.fc_range(range_feat)

        typed_evidence, evidence_layer_attention, evidence_attention_entropy = self._typed_evidence(
            layer_tokens, layer_features
        )
        heading_observability, range_observability = self._observability(
            base_heading_feat, range_feat, typed_evidence
        )
        heading_weight, range_diag_weight = self._router_weights(
            base_heading_feat,
            range_feat,
            typed_evidence,
            heading_observability,
            range_observability,
        )
        heading_evidence = (heading_weight.unsqueeze(-1) * typed_evidence).sum(dim=1)
        residual_input = torch.cat(
            [
                heading_evidence,
                base_heading_feat,
                range_feat.detach(),
                heading_observability,
            ],
            dim=-1,
        )
        heading_gate = torch.sigmoid(self.heading_gate(residual_input))
        delta_rad = self.heading_residual_max_delta_rad * heading_gate * torch.tanh(
            self.heading_residual_mlp(residual_input)
        )
        heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)

        typed_evidence_norm = typed_evidence.detach().norm(dim=-1)
        return {
            "heading_vec": heading_vec,
            "range_value": protected_range_value,
            "phase104c_heading_base_vec": base_heading_vec.detach(),
            "phase104c_protected_range_value": protected_range_value.detach(),
            "phase104c_heading_gate": heading_gate.detach(),
            "phase104c_heading_residual_delta_rad": delta_rad,
            "phase104c_heading_residual_delta_deg": delta_rad * (180.0 / math.pi),
            "phase104c_heading_observability": heading_observability,
            "phase104c_range_observability": range_observability,
            "phase104c_heading_evidence_weights": heading_weight.detach(),
            "phase104c_range_diag_evidence_weights": range_diag_weight.detach(),
            "phase104c_heading_router_entropy": _phase104_entropy(heading_weight, dim=-1).detach(),
            "phase104c_range_router_entropy": _phase104_entropy(range_diag_weight, dim=-1).detach(),
            "phase104c_typed_evidence_norm": typed_evidence_norm,
            "phase104c_typed_evidence_contribution": heading_weight.detach() * typed_evidence_norm,
            "phase104c_evidence_layer_attention": evidence_layer_attention.detach().flatten(start_dim=1),
            "phase104c_evidence_attention_entropy": evidence_attention_entropy.detach(),
            "phase104c_protected_range_feature_norm": range_feat.detach().norm(dim=-1),
        }


class Phase104eProtectedAxisAsymmetricExpertHead(AxisAsyncQueryBridgePairUAVHead):
    """
    Phase104e / PAAER head.

    The heading expert reuses the A1 AxisAsyncQueryBridge heading path:
    fixed mid+late layer bank, one heading task token, task cross-attention,
    task norm/FFN, and the heading token MLP. Range stays on the protected
    H0/C0 metric path from PairUAVHead.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        use_blend=False,
        task_token_num_heads=8,
        bridge_hidden_dim=128,
        blend_hidden_dim=256,
    ):
        super().__init__(
            net=net,
            num_resconv_block=num_resconv_block,
            heading_layers=("mid", "late"),
            range_layers=("late",),
            learnable_layer_weights=False,
            use_gated_bridge=False,
            task_token_num_heads=task_token_num_heads,
            bridge_hidden_dim=bridge_hidden_dim,
        )
        self.use_blend = bool(use_blend)
        output_dim = 4 * self.patch_size ** 2
        self.phase104e_heading_fc = copy.deepcopy(self.fc_heading)
        self.phase104e_base_heading_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.phase104e_blend_gate = nn.Sequential(
            nn.Linear(output_dim * 3 + 4, int(blend_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(blend_hidden_dim), 1),
        )
        self.reset_phase104e_parameters()

    def reset_phase104e_parameters(self):
        gate_last = self.phase104e_blend_gate[-1]
        nn.init.zeros_(gate_last.weight)
        nn.init.zeros_(gate_last.bias)

    def reset_after_model_init(self):
        self.reset_phase104e_parameters()

    def _phase104e_layer_feature_from_tokens(self, tokens, img_shape):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return self.more_mlps(feat)

    def _phase104e_base_heading_feature(self, decout, img_shape):
        candidates = _phase104_select_named_decoder_layers(decout)
        mid_feat = self._phase104e_layer_feature_from_tokens(candidates["mid"], img_shape)
        late_feat = self._phase104e_layer_feature_from_tokens(candidates["late"], img_shape)
        return self.phase104e_base_heading_fusion(torch.cat([mid_feat, late_feat], dim=-1))

    def _phase104e_heading_expert_feature(self, decout):
        heading_task, heading_attn, heading_entropy = self._query_one_factor(
            decout, 0, self.heading_layers, self.heading_layer_logits
        )
        heading_feat = self.heading_token_mlp(heading_task)
        return heading_feat, heading_attn, heading_entropy

    @staticmethod
    def _phase104e_heading_delta_deg(base_heading_vec, expert_heading_vec):
        dot = (base_heading_vec * expert_heading_vec).sum(dim=-1).clamp(min=-1.0, max=1.0)
        cross = (
            base_heading_vec[:, 0] * expert_heading_vec[:, 1]
            - base_heading_vec[:, 1] * expert_heading_vec[:, 0]
        )
        return torch.rad2deg(torch.atan2(cross, dot)).abs()

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_expert_feat, heading_attn, heading_entropy = self._phase104e_heading_expert_feature(decout)
        expert_heading_vec = F.normalize(self.phase104e_heading_fc(heading_expert_feat), dim=-1, eps=1e-6)

        base_heading_feat = self._phase104e_base_heading_feature(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_heading_feat), dim=-1, eps=1e-6)

        range_feat = self._extract_features(decout, img_shape)
        protected_range_value = self.fc_range(range_feat)
        range_value = protected_range_value

        blend_alpha = protected_range_value.new_ones(protected_range_value.shape[0], 1)
        if self.use_blend:
            blend_input = torch.cat(
                [
                    base_heading_feat,
                    heading_expert_feat,
                    range_feat.detach(),
                    base_heading_vec.detach(),
                    expert_heading_vec.detach(),
                ],
                dim=-1,
            )
            blend_alpha = torch.sigmoid(self.phase104e_blend_gate(blend_input))
            heading_vec = F.normalize(
                (1.0 - blend_alpha) * base_heading_vec + blend_alpha * expert_heading_vec,
                dim=-1,
                eps=1e-6,
            )
        else:
            heading_vec = expert_heading_vec

        alpha_flat = blend_alpha.detach().view(-1)
        range_contract = (range_value - protected_range_value.detach()).abs()
        heading_delta_deg = self._phase104e_heading_delta_deg(base_heading_vec, expert_heading_vec)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "phase104e_heading_expert_vec": expert_heading_vec.detach(),
            "phase104e_base_heading_vec": base_heading_vec.detach(),
            "phase104e_protected_range_value": protected_range_value.detach(),
            "phase104e_range_final_minus_protected_abs": range_contract,
            "phase104e_heading_expert_minus_base_delta_deg": heading_delta_deg.detach(),
            "phase104e_heading_layer_attention": heading_attn.detach(),
            "phase104e_heading_token_entropy": heading_entropy.detach(),
            "phase104e_heading_blend_alpha": blend_alpha.detach(),
            "phase104e_heading_blend_alpha_mean": alpha_flat.mean(),
            "phase104e_heading_blend_alpha_std": alpha_flat.std(unbiased=False),
            "phase104e_heading_blend_alpha_min": alpha_flat.min(),
            "phase104e_heading_blend_alpha_max": alpha_flat.max(),
            "phase104e_range_anchor_norm": range_feat.detach().norm(dim=-1),
        }


class Phase104dPolarRelationMemoryHead(PairUAVHead):
    """
    Phase104d / PRM head.

    PRM differs from generic relation bottlenecks by giving every typed slot a
    different structural source before the slot query is applied. Range remains
    on the inherited H0/late protected path; relation memory only refines the
    H8-style mid/late heading base through a bounded circular residual.
    """

    _LAYER_NAMES = ("early", "mid", "late")
    _SLOT_NAMES = ("bearing", "layout", "scale", "overlap", "ambiguity")
    _SOURCE_NAMES = ("early", "mid", "late", "agreement", "disagreement")
    _SRC_EARLY = 0
    _SRC_MID = 1
    _SRC_LATE = 2
    _SRC_AGREEMENT = 3
    _SRC_DISAGREEMENT = 4

    def __init__(
        self,
        net,
        num_resconv_block=2,
        use_auxiliary=False,
        heading_readout_mode="residual",
        heading_bin_count=12,
        range_bin_count=8,
        heading_residual_max_delta_deg=5.0,
        slot_hidden_dim=256,
        task_token_num_heads=8,
        source_pool_grid=(12, 16),
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.use_auxiliary = bool(use_auxiliary)
        self.heading_readout_mode = str(heading_readout_mode)
        if self.heading_readout_mode not in ("residual", "direct_memory", "bounded_memory_delta"):
            raise ValueError(f"Unsupported Phase104d heading_readout_mode={heading_readout_mode}")
        self.heading_bin_count = int(heading_bin_count)
        self.range_bin_count = int(range_bin_count)
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))
        self.source_pool_grid = tuple(int(x) for x in source_pool_grid)

        output_dim = 4 * self.patch_size ** 2
        token_dim = net.dec_embed_dim
        hidden_dim = int(slot_hidden_dim)

        self.heading_layer_projs = nn.ModuleList([copy.deepcopy(self.proj) for _ in self._LAYER_NAMES])
        self.heading_layer_res_convs = nn.ModuleList([copy.deepcopy(self.res_conv) for _ in self._LAYER_NAMES])
        self.heading_layer_more_mlps = nn.ModuleList([copy.deepcopy(self.more_mlps) for _ in self._LAYER_NAMES])
        self.heading_base_fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

        self.slot_queries = nn.Parameter(torch.zeros(len(self._SLOT_NAMES), token_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.slot_norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in self._SLOT_NAMES])
        self.slot_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, token_dim * 2),
                    nn.ReLU(),
                    nn.Linear(token_dim * 2, token_dim),
                )
                for _ in self._SLOT_NAMES
            ]
        )
        self.slot_value_projs = nn.ModuleList([nn.Linear(token_dim, token_dim) for _ in self._SLOT_NAMES])
        self.slot_score_mlps = nn.ModuleList([nn.Linear(token_dim, 1) for _ in self._SLOT_NAMES])
        self.slot_feature_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                )
                for _ in self._SLOT_NAMES
            ]
        )

        self.bearing_coord_mlp = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, token_dim),
        )
        self.scale_coord_mlp = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, token_dim),
        )
        self.layout_gap_proj = nn.Linear(token_dim, token_dim)
        self.overlap_agreement_proj = nn.Linear(token_dim * 3, token_dim)
        self.overlap_disagreement_proj = nn.Linear(token_dim, token_dim)
        self.ambiguity_gap_proj = nn.Linear(token_dim, token_dim)
        self.ambiguity_std_proj = nn.Linear(token_dim, token_dim)

        residual_input_dim = output_dim * 6
        gate_input_dim = output_dim * 5
        self.heading_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.heading_residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        direct_input_dim = output_dim * 7
        self.heading_memory_fusion_mlp = nn.Sequential(
            nn.Linear(direct_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.heading_memory_gate = nn.Sequential(
            nn.Linear(direct_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bearing_bin_head = nn.Linear(output_dim, self.heading_bin_count)
        self.scale_bin_head = nn.Linear(output_dim, self.range_bin_count)
        self.ambiguity_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_phase104d_parameters()

    def reset_phase104d_parameters(self):
        gate_last = self.heading_gate[-1]
        nn.init.normal_(gate_last.weight, mean=0.0, std=1e-4)
        gate_last.bias.data.fill_(math.log(0.15 / 0.85))

        residual_last = self.heading_residual_mlp[-1]
        nn.init.normal_(residual_last.weight, mean=0.0, std=1e-4)
        nn.init.normal_(residual_last.bias, mean=0.0, std=1e-4)

        memory_gate_last = self.heading_memory_gate[-1]
        nn.init.normal_(memory_gate_last.weight, mean=0.0, std=1e-4)
        memory_gate_last.bias.data.fill_(math.log(0.35 / 0.65))

    def reset_after_model_init(self):
        self.reset_phase104d_parameters()

    def _select_named_layers(self, decout):
        candidates = _phase104_select_named_decoder_layers(decout)
        return [candidates[name] for name in self._LAYER_NAMES]

    def _layer_feature(self, tokens, img_shape, proj, res_conv, more_mlps):
        H, W = img_shape
        B, S, D = tokens.shape
        feat = proj(tokens)
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        for i in range(self.num_resconv_block):
            feat = res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return more_mlps(feat)

    def _base_heading_features(self, layer_features):
        return self.heading_base_fusion_mlp(torch.cat([layer_features[1], layer_features[2]], dim=-1))

    def _coordinate_features(self, tokens, img_shape):
        H, W = img_shape
        gh = H // self.patch_size
        gw = W // self.patch_size
        if gh * gw != tokens.shape[1]:
            coord = tokens.new_zeros(1, tokens.shape[1], 3)
            scale_coord = tokens.new_zeros(1, tokens.shape[1], 2)
            return coord, scale_coord
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, gh, device=tokens.device, dtype=tokens.dtype),
            torch.linspace(-1.0, 1.0, gw, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        r = torch.sqrt((x ** 2 + y ** 2).clamp_min(0.0))
        coord = torch.stack([x, y, r], dim=-1).view(1, gh * gw, 3)
        scale_coord = torch.stack([r, r ** 2], dim=-1).view(1, gh * gw, 2)
        return coord, scale_coord

    def _pooled_grid_shape(self, tokens, img_shape):
        H, W = img_shape
        gh = H // self.patch_size
        gw = W // self.patch_size
        if gh * gw != tokens.shape[1]:
            return None
        target_h = min(gh, max(1, int(self.source_pool_grid[0])))
        target_w = min(gw, max(1, int(self.source_pool_grid[1])))
        return target_h, target_w

    def _coordinate_features_for_grid(self, tokens, grid_shape):
        if grid_shape is None:
            coord = tokens.new_zeros(1, tokens.shape[1], 3)
            scale_coord = tokens.new_zeros(1, tokens.shape[1], 2)
            return coord, scale_coord
        gh, gw = grid_shape
        if gh * gw != tokens.shape[1]:
            coord = tokens.new_zeros(1, tokens.shape[1], 3)
            scale_coord = tokens.new_zeros(1, tokens.shape[1], 2)
            return coord, scale_coord
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, gh, device=tokens.device, dtype=tokens.dtype),
            torch.linspace(-1.0, 1.0, gw, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        r = torch.sqrt((x ** 2 + y ** 2).clamp_min(0.0))
        coord = torch.stack([x, y, r], dim=-1).view(1, gh * gw, 3)
        scale_coord = torch.stack([r, r ** 2], dim=-1).view(1, gh * gw, 2)
        return coord, scale_coord

    def _pool_token_component(self, tokens, img_shape):
        grid_shape = self._pooled_grid_shape(tokens, img_shape)
        if grid_shape is None:
            return tokens
        H, W = img_shape
        gh = H // self.patch_size
        gw = W // self.patch_size
        target_h, target_w = grid_shape
        if target_h == gh and target_w == gw:
            return tokens
        B, S, D = tokens.shape
        feat = tokens.transpose(1, 2).view(B, D, gh, gw)
        feat = F.adaptive_avg_pool2d(feat, (target_h, target_w))
        return feat.flatten(2).transpose(1, 2)

    def _label_tensor(self, component_tensors, components):
        labels = [
            torch.full(
                (component_tensor.shape[1],),
                int(component),
                device=component_tensor.device,
                dtype=torch.long,
            )
            for component_tensor, component in zip(component_tensors, components)
        ]
        return torch.cat(labels, dim=0)

    def _slot_source_banks(self, layer_tokens, img_shape):
        pooled_grid_shape = self._pooled_grid_shape(layer_tokens[-1], img_shape)
        early, mid, late = [
            self._pool_token_component(tokens, img_shape)
            for tokens in layer_tokens
        ]
        coord, scale_coord = self._coordinate_features_for_grid(late, pooled_grid_shape)
        bearing_coord = self.bearing_coord_mlp(coord).to(dtype=late.dtype)
        scale_coord = self.scale_coord_mlp(scale_coord).to(dtype=late.dtype)

        mid_late_gap = (mid - late).abs()
        early_mid_gap = (early - mid).abs()
        layer_std = torch.stack([early, mid, late], dim=2).std(dim=2, unbiased=False)

        bearing_parts = [
            mid + bearing_coord,
            late + bearing_coord,
        ]
        bearing_bank = torch.cat(bearing_parts, dim=1)
        bearing_labels = self._label_tensor(bearing_parts, [self._SRC_MID, self._SRC_LATE])

        layout_parts = [
            mid + bearing_coord,
            late + bearing_coord,
            self.layout_gap_proj(mid_late_gap),
        ]
        layout_bank = torch.cat(layout_parts, dim=1)
        layout_labels = self._label_tensor(
            layout_parts,
            [self._SRC_MID, self._SRC_LATE, self._SRC_DISAGREEMENT],
        )

        scale_parts = [late + scale_coord]
        scale_bank = torch.cat(scale_parts, dim=1)
        scale_labels = self._label_tensor(scale_parts, [self._SRC_LATE])

        agreement = self.overlap_agreement_proj(torch.cat([mid, late, mid * late], dim=-1))
        disagreement = self.overlap_disagreement_proj(mid_late_gap)
        overlap_parts = [
            agreement,
            disagreement,
        ]
        overlap_bank = torch.cat(overlap_parts, dim=1)
        overlap_labels = self._label_tensor(overlap_parts, [self._SRC_AGREEMENT, self._SRC_DISAGREEMENT])

        ambiguity_parts = [
            self.ambiguity_gap_proj(early_mid_gap),
            self.ambiguity_gap_proj(mid_late_gap),
            self.ambiguity_std_proj(layer_std),
        ]
        ambiguity_bank = torch.cat(ambiguity_parts, dim=1)
        ambiguity_labels = self._label_tensor(
            ambiguity_parts,
            [self._SRC_DISAGREEMENT, self._SRC_DISAGREEMENT, self._SRC_DISAGREEMENT],
        )

        return [
            (bearing_bank, bearing_labels),
            (layout_bank, layout_labels),
            (scale_bank, scale_labels),
            (overlap_bank, overlap_labels),
            (ambiguity_bank, ambiguity_labels),
        ]

    def _query_slot(self, slot_idx, bank, labels):
        B = bank.shape[0]
        query = self.slot_queries[slot_idx].view(1, 1, -1).expand(B, 1, -1)
        query_vec = query[:, 0]
        value = self.slot_value_projs[slot_idx](bank)
        query_score = (bank * query_vec.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(bank.shape[-1]))
        content_score = self.slot_score_mlps[slot_idx](bank).squeeze(-1)
        token_prob = torch.softmax(query_score + content_score, dim=-1)
        attended = (token_prob.unsqueeze(-1) * value).sum(dim=1, keepdim=True)
        slot = self.slot_norms[slot_idx](query + attended)
        slot = self.slot_norms[slot_idx](slot + self.slot_ffns[slot_idx](slot))
        source_mass = bank.new_zeros(B, len(self._SOURCE_NAMES))
        for source_idx in range(len(self._SOURCE_NAMES)):
            mask = labels == source_idx
            if bool(mask.any()):
                source_mass[:, source_idx] = token_prob[:, mask].sum(dim=-1)
        entropy = _phase104_entropy(token_prob, dim=-1)
        return slot[:, 0], source_mass, entropy

    def _relation_slots(self, layer_tokens, img_shape):
        slots = []
        source_masses = []
        entropies = []
        source_token_counts = []
        for idx, (bank, labels) in enumerate(self._slot_source_banks(layer_tokens, img_shape)):
            slot, source_mass, entropy = self._query_slot(idx, bank, labels)
            slots.append(slot)
            source_masses.append(source_mass)
            entropies.append(entropy)
            source_token_counts.append(bank.shape[1])
        slot_tokens = torch.stack(slots, dim=1)
        slot_features = torch.stack(
            [
                mlp(slot_tokens[:, idx])
                for idx, mlp in enumerate(self.slot_feature_mlps)
            ],
            dim=1,
        )
        token_count_tensor = torch.as_tensor(source_token_counts, device=slot_features.device, dtype=torch.long)
        token_count_tensor = token_count_tensor.view(1, -1).expand(slot_features.shape[0], -1)
        return (
            slot_tokens,
            slot_features,
            torch.stack(source_masses, dim=1),
            torch.stack(entropies, dim=1),
            token_count_tensor,
        )

    def _make_slot_mask(self, slot_features, phase104d_mask_slot):
        B = slot_features.shape[0]
        mask = slot_features.new_ones(B, len(self._SLOT_NAMES))
        if phase104d_mask_slot is None:
            return mask
        if torch.is_tensor(phase104d_mask_slot):
            slot_index = phase104d_mask_slot.to(device=slot_features.device).long().view(-1)
            if slot_index.numel() == 1:
                slot_index = slot_index.expand(B)
        else:
            slot_index = torch.full(
                (B,),
                int(phase104d_mask_slot),
                device=slot_features.device,
                dtype=torch.long,
            )
        slot_index = slot_index.clamp(min=0, max=len(self._SLOT_NAMES) - 1)
        mask[torch.arange(B, device=slot_features.device), slot_index] = 0.0
        return mask

    def forward(self, decout, img_shape, phase104d_mask_slot=None, **_unused_head_kwargs):
        layer_tokens = self._select_named_layers(decout)
        layer_features = [
            self._layer_feature(tokens, img_shape, proj, res_conv, more_mlps)
            for tokens, proj, res_conv, more_mlps in zip(
                layer_tokens,
                self.heading_layer_projs,
                self.heading_layer_res_convs,
                self.heading_layer_more_mlps,
            )
        ]
        base_heading_feat = self._base_heading_features(layer_features)
        range_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_heading_feat), dim=-1, eps=1e-6)
        protected_range_value = self.fc_range(range_feat)

        _slot_tokens, slot_features, slot_source_mass, slot_entropy, slot_source_token_count = self._relation_slots(
            layer_tokens,
            img_shape,
        )
        slot_mask = self._make_slot_mask(slot_features, phase104d_mask_slot)
        slot_features = slot_features * slot_mask.unsqueeze(-1)
        bearing_feat = slot_features[:, 0]
        layout_feat = slot_features[:, 1]
        scale_feat = slot_features[:, 2]
        overlap_feat = slot_features[:, 3]
        ambiguity_feat = slot_features[:, 4]

        residual_input = torch.cat(
            [
                bearing_feat,
                layout_feat,
                overlap_feat,
                ambiguity_feat,
                base_heading_feat,
                range_feat.detach(),
            ],
            dim=-1,
        )
        gate_input = torch.cat(
            [
                bearing_feat,
                layout_feat,
                ambiguity_feat,
                base_heading_feat,
                range_feat.detach(),
            ],
            dim=-1,
        )
        heading_gate = torch.sigmoid(self.heading_gate(gate_input))
        delta_rad = self.heading_residual_max_delta_rad * heading_gate * torch.tanh(
            self.heading_residual_mlp(residual_input)
        )
        residual_heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)

        direct_input = torch.cat(
            [
                bearing_feat,
                layout_feat,
                scale_feat,
                overlap_feat,
                ambiguity_feat,
                base_heading_feat,
                range_feat.detach(),
            ],
            dim=-1,
        )
        memory_feat = self.heading_memory_fusion_mlp(direct_input)
        memory_heading_vec = F.normalize(self.fc_heading(memory_feat), dim=-1, eps=1e-6)
        memory_gate = torch.sigmoid(self.heading_memory_gate(direct_input))
        fused_heading_vec = F.normalize(
            (1.0 - memory_gate) * base_heading_vec + memory_gate * memory_heading_vec,
            dim=-1,
            eps=1e-6,
        )
        base_memory_cos = (base_heading_vec * memory_heading_vec).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        base_memory_cross = (
            base_heading_vec[:, 0:1] * memory_heading_vec[:, 1:2]
            - base_heading_vec[:, 1:2] * memory_heading_vec[:, 0:1]
        )
        base_to_memory_delta_rad = torch.atan2(base_memory_cross, base_memory_cos)
        bounded_memory_delta_rad = self.heading_residual_max_delta_rad * torch.tanh(
            base_to_memory_delta_rad / max(self.heading_residual_max_delta_rad, 1e-6)
        )
        active_delta_rad = delta_rad
        if self.heading_readout_mode == "bounded_memory_delta":
            active_delta_rad = memory_gate * bounded_memory_delta_rad
            heading_vec = _rotate_heading_vec(base_heading_vec, active_delta_rad)
        elif self.heading_readout_mode == "direct_memory":
            active_delta_rad = memory_gate * bounded_memory_delta_rad
            heading_vec = fused_heading_vec
        else:
            heading_vec = residual_heading_vec
            memory_gate = memory_gate.detach() * 0.0
        base_memory_delta_deg = torch.rad2deg(torch.acos(base_memory_cos))

        out = {
            "heading_vec": heading_vec,
            "range_value": protected_range_value,
            "phase104d_heading_base_vec": base_heading_vec.detach(),
            "phase104d_heading_residual_vec": residual_heading_vec.detach(),
            "phase104d_heading_memory_vec": memory_heading_vec.detach(),
            "phase104d_protected_range_value": protected_range_value.detach(),
            "phase104d_heading_gate": heading_gate.detach(),
            "phase104d_heading_memory_gate": memory_gate.detach(),
            "phase104d_heading_residual_delta_rad": active_delta_rad,
            "phase104d_heading_residual_delta_deg": active_delta_rad * (180.0 / math.pi),
            "phase104d_heading_scalar_residual_delta_rad": delta_rad.detach(),
            "phase104d_heading_scalar_residual_delta_deg": (delta_rad * (180.0 / math.pi)).detach(),
            "phase104d_heading_memory_delta_rad": base_to_memory_delta_rad.detach(),
            "phase104d_heading_memory_delta_deg": (base_to_memory_delta_rad * (180.0 / math.pi)).detach(),
            "phase104d_heading_bounded_memory_delta_rad": bounded_memory_delta_rad.detach(),
            "phase104d_heading_bounded_memory_delta_deg": (bounded_memory_delta_rad * (180.0 / math.pi)).detach(),
            "phase104d_base_memory_delta_deg": base_memory_delta_deg.detach(),
            "phase104d_slot_norm": slot_features.detach().norm(dim=-1),
            "phase104d_slot_mask": slot_mask.detach(),
            "phase104d_slot_source_mass": slot_source_mass.detach(),
            "phase104d_slot_source_token_count": slot_source_token_count.detach(),
            "phase104d_slot_attention_entropy": slot_entropy.detach(),
            "phase104d_range_final_minus_protected_abs": (protected_range_value - protected_range_value.detach()).abs(),
            "phase104d_scale_slot_norm": scale_feat.detach().norm(dim=-1),
        }
        if self.use_auxiliary:
            out.update(
                {
                    "phase104d_bearing_bin_logits": self.bearing_bin_head(bearing_feat),
                    "phase104d_scale_bin_logits": self.scale_bin_head(scale_feat),
                    "phase104d_ambiguity_pred": torch.sigmoid(self.ambiguity_head(ambiguity_feat)),
                }
            )
        return out


class MSRCompactBottleneck(nn.Module):
    """Compact relation variable used by MSR/MNR smoke heads."""

    def __init__(self, input_dim, bottleneck_dim=128, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(input_dim, int(bottleneck_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(int(bottleneck_dim), input_dim)

    def forward(self, feat):
        z = F.relu(self.down(feat))
        z = self.dropout(z)
        expanded = F.relu(self.up(z))
        return expanded, z


class MNRAnchoredResidualBottleneckPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    """
    Phase94 MNR-A0: H8-anchored residual bottleneck complement head.

    The base H8 mid/late heading path and H0 range path keep the same parameter
    names as the Phase89/90 H8 checkpoint. New MNR bottleneck residuals are
    zero-output after model initialization, so loading the H8 checkpoint starts
    from the H8 prediction surface and only trains a compact residual expert.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        mnr_bottleneck_dim=64,
        mnr_dropout=0.0,
        mnr_residual_scale=1.0,
        mnr_max_heading_delta=0.05,
        mnr_max_range_delta=0.25,
    ):
        super().__init__(
            net=net,
            heading_readout_layers=("mid", "late"),
            num_resconv_block=num_resconv_block,
        )
        output_dim = 4 * self.patch_size ** 2
        self.mnr_heading_bottleneck = MSRCompactBottleneck(
            output_dim, bottleneck_dim=mnr_bottleneck_dim, dropout=mnr_dropout
        )
        self.mnr_range_bottleneck = MSRCompactBottleneck(
            output_dim, bottleneck_dim=mnr_bottleneck_dim, dropout=mnr_dropout
        )
        self.mnr_heading_delta = nn.Linear(output_dim, 2)
        self.mnr_range_delta = nn.Linear(output_dim, 1)
        self.mnr_heading_scale = nn.Parameter(torch.tensor(float(mnr_residual_scale)))
        self.mnr_range_scale = nn.Parameter(torch.tensor(float(mnr_residual_scale)))
        self.mnr_max_heading_delta = float(mnr_max_heading_delta)
        self.mnr_max_range_delta = float(mnr_max_range_delta)

    def reset_after_model_init(self):
        nn.init.zeros_(self.mnr_heading_delta.weight)
        nn.init.zeros_(self.mnr_heading_delta.bias)
        nn.init.zeros_(self.mnr_range_delta.weight)
        nn.init.zeros_(self.mnr_range_delta.bias)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        base_heading = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        base_range = self.fc_range(range_feat)

        heading_residual_feat, heading_z = self.mnr_heading_bottleneck(heading_feat)
        range_residual_feat, range_z = self.mnr_range_bottleneck(range_feat)
        heading_delta = torch.tanh(self.mnr_heading_scale) * self.mnr_heading_delta(heading_residual_feat)
        range_delta = torch.tanh(self.mnr_range_scale) * self.mnr_range_delta(range_residual_feat)
        heading_delta = torch.clamp(
            heading_delta,
            min=-self.mnr_max_heading_delta,
            max=self.mnr_max_heading_delta,
        )
        range_delta = torch.clamp(
            range_delta,
            min=-self.mnr_max_range_delta,
            max=self.mnr_max_range_delta,
        )

        heading_vec = F.normalize(base_heading + heading_delta, dim=-1, eps=1e-6)
        range_value = base_range + range_delta
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "mnr_base_heading_vec": base_heading.detach(),
            "mnr_base_range_value": base_range.detach(),
            "mnr_heading_delta_norm": heading_delta.detach().norm(dim=-1),
            "mnr_range_delta_abs": range_delta.detach().abs().view(-1),
            "mnr_heading_z_norm": heading_z.detach().norm(dim=-1),
            "mnr_range_z_norm": range_z.detach().norm(dim=-1),
        }


class Phase96RegimeConditionedResidualPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    # Phase96-A0 H8-anchored shared-pose latent residual with a learned regime gate.

    def __init__(self, net, num_resconv_block=2, bottleneck_dim=64, dropout=0.0,
                 residual_scale=1.0, max_heading_delta=0.05, max_range_delta=0.25,
                 num_regime_experts=4, gate_hidden_dim=128):
        super().__init__(net=net, heading_readout_layers=("mid", "late"), num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        joint_dim = output_dim * 2
        self.phase96_num_regime_experts = int(num_regime_experts)
        self.phase96_gate = nn.Sequential(
            nn.Linear(joint_dim, int(gate_hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden_dim), self.phase96_num_regime_experts),
        )
        self.phase96_residual_bottleneck = MSRCompactBottleneck(joint_dim, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.phase96_heading_delta = nn.Linear(joint_dim, self.phase96_num_regime_experts * 2)
        self.phase96_range_delta = nn.Linear(joint_dim, self.phase96_num_regime_experts)
        self.phase96_heading_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase96_range_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase96_max_heading_delta = float(max_heading_delta)
        self.phase96_max_range_delta = float(max_range_delta)

    def reset_after_model_init(self):
        nn.init.zeros_(self.phase96_heading_delta.weight)
        nn.init.zeros_(self.phase96_heading_delta.bias)
        nn.init.zeros_(self.phase96_range_delta.weight)
        nn.init.zeros_(self.phase96_range_delta.bias)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        base_heading = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        base_range = self.fc_range(range_feat)
        joint_feat = torch.cat([heading_feat, range_feat], dim=-1)
        gate_logits = self.phase96_gate(joint_feat)
        gate = F.softmax(gate_logits, dim=-1)
        residual_feat, residual_z = self.phase96_residual_bottleneck(joint_feat)
        heading_delta_experts = self.phase96_heading_delta(residual_feat).view(-1, self.phase96_num_regime_experts, 2)
        range_delta_experts = self.phase96_range_delta(residual_feat).view(-1, self.phase96_num_regime_experts, 1)
        heading_delta = (gate.unsqueeze(-1) * heading_delta_experts).sum(dim=1)
        range_delta = (gate.unsqueeze(-1) * range_delta_experts).sum(dim=1)
        heading_delta = torch.tanh(self.phase96_heading_scale) * heading_delta
        range_delta = torch.tanh(self.phase96_range_scale) * range_delta
        heading_delta = torch.clamp(heading_delta, min=-self.phase96_max_heading_delta, max=self.phase96_max_heading_delta)
        range_delta = torch.clamp(range_delta, min=-self.phase96_max_range_delta, max=self.phase96_max_range_delta)
        heading_vec = F.normalize(base_heading + heading_delta, dim=-1, eps=1e-6)
        range_value = base_range + range_delta
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "phase96_base_heading_vec": base_heading.detach(),
            "phase96_base_range_value": base_range.detach(),
            "phase96_gate_entropy": (-(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=-1)).detach(),
            "phase96_gate_max": gate.max(dim=-1).values.detach(),
            "phase96_heading_delta_norm": heading_delta.detach().norm(dim=-1),
            "phase96_range_delta_abs": range_delta.detach().abs().view(-1),
            "phase96_residual_z_norm": residual_z.detach().norm(dim=-1),
        }


class Phase99StabilityAwareReadoutPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    """
    Phase99-A H8-anchored stability-aware readout.

    The base H8 mid-late heading and H0 range outputs remain the identity path.
    A learned scalar stability gate controls a bounded residual correction.
    Zero-initialized residual layers make the initial output exactly match H8.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        bottleneck_dim=64,
        dropout=0.0,
        residual_scale=1.0,
        max_heading_delta=0.05,
        max_range_delta=0.25,
        gate_hidden_dim=128,
        gate_init_bias=-5.0,
    ):
        super().__init__(net=net, heading_readout_layers=("mid", "late"), num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        joint_dim = output_dim * 2
        self.phase99_stability_gate = nn.Sequential(
            nn.Linear(joint_dim, int(gate_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        self.phase99_residual_bottleneck = MSRCompactBottleneck(
            joint_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.phase99_heading_delta = nn.Linear(joint_dim, 2)
        self.phase99_range_delta = nn.Linear(joint_dim, 1)
        self.phase99_heading_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase99_range_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase99_max_heading_delta = float(max_heading_delta)
        self.phase99_max_range_delta = float(max_range_delta)
        self.phase99_gate_init_bias = float(gate_init_bias)

    def reset_after_model_init(self):
        nn.init.zeros_(self.phase99_heading_delta.weight)
        nn.init.zeros_(self.phase99_heading_delta.bias)
        nn.init.zeros_(self.phase99_range_delta.weight)
        nn.init.zeros_(self.phase99_range_delta.bias)
        final_gate = self.phase99_stability_gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, self.phase99_gate_init_bias)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        base_heading = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        base_range = self.fc_range(range_feat)

        joint_feat = torch.cat([heading_feat, range_feat], dim=-1)
        stability_gate = torch.sigmoid(self.phase99_stability_gate(joint_feat))
        residual_feat, residual_z = self.phase99_residual_bottleneck(joint_feat)
        heading_delta = torch.tanh(self.phase99_heading_scale) * self.phase99_heading_delta(residual_feat)
        range_delta = torch.tanh(self.phase99_range_scale) * self.phase99_range_delta(residual_feat)
        heading_delta = torch.clamp(heading_delta, min=-self.phase99_max_heading_delta, max=self.phase99_max_heading_delta)
        range_delta = torch.clamp(range_delta, min=-self.phase99_max_range_delta, max=self.phase99_max_range_delta)
        heading_delta = stability_gate * heading_delta
        range_delta = stability_gate * range_delta

        if (not self.training) and float(heading_delta.detach().abs().max().cpu()) == 0.0:
            heading_vec = base_heading
        else:
            heading_vec = F.normalize(base_heading + heading_delta, dim=-1, eps=1e-6)
        range_value = base_range + range_delta
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "phase99_base_heading_vec": base_heading.detach(),
            "phase99_base_range_value": base_range.detach(),
            "phase99_stability_gate": stability_gate.detach().view(-1),
            "phase99_heading_delta_norm": heading_delta.detach().norm(dim=-1),
            "phase99_range_delta_abs": range_delta.detach().abs().view(-1),
            "phase99_residual_z_norm": residual_z.detach().norm(dim=-1),
        }


class Phase101AxiswiseTrajectoryReadoutPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    # Phase101-A1 SATR: H8-anchored axiswise trajectory-supervised readout.
    # The base H8 path is unchanged. Heading/range use independent gates and
    # independent checkpoint-mode routers over bounded residual experts.
    # Zero-initialized residual layers make the initial output exactly match H8.

    def __init__(
        self,
        net,
        num_resconv_block=2,
        bottleneck_dim=64,
        dropout=0.0,
        residual_scale=1.0,
        max_heading_delta=0.05,
        max_range_delta=0.25,
        router_hidden_dim=128,
        gate_init_bias=-6.0,
        num_modes=5,
        router_final_bias=2.0,
    ):
        super().__init__(net=net, heading_readout_layers=("mid", "late"), num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        joint_dim = output_dim * 2
        self.phase101_num_modes = int(num_modes)
        self.phase101_router_final_bias = float(router_final_bias)
        self.phase101_heading_gate = nn.Sequential(
            nn.Linear(joint_dim, int(router_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), 1),
        )
        self.phase101_range_gate = nn.Sequential(
            nn.Linear(joint_dim, int(router_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), 1),
        )
        self.phase101_heading_router = nn.Sequential(
            nn.Linear(joint_dim, int(router_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), self.phase101_num_modes),
        )
        self.phase101_range_router = nn.Sequential(
            nn.Linear(joint_dim, int(router_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), self.phase101_num_modes),
        )
        self.phase101_residual_bottleneck = MSRCompactBottleneck(
            joint_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.phase101_heading_delta = nn.Linear(joint_dim, self.phase101_num_modes * 2)
        self.phase101_range_delta = nn.Linear(joint_dim, self.phase101_num_modes)
        self.phase101_heading_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase101_range_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.phase101_max_heading_delta = float(max_heading_delta)
        self.phase101_max_range_delta = float(max_range_delta)
        self.phase101_gate_init_bias = float(gate_init_bias)

    def reset_after_model_init(self):
        nn.init.zeros_(self.phase101_heading_delta.weight)
        nn.init.zeros_(self.phase101_heading_delta.bias)
        nn.init.zeros_(self.phase101_range_delta.weight)
        nn.init.zeros_(self.phase101_range_delta.bias)
        for gate in (self.phase101_heading_gate[-1], self.phase101_range_gate[-1]):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, self.phase101_gate_init_bias)
        for router in (self.phase101_heading_router[-1], self.phase101_range_router[-1]):
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
            if self.phase101_num_modes > 0:
                router.bias.data[0] = self.phase101_router_final_bias

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        heading_feat = self._heading_features(decout, img_shape)
        range_feat = self._extract_features(decout, img_shape)
        base_heading = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        base_range = self.fc_range(range_feat)

        joint_feat = torch.cat([heading_feat, range_feat], dim=-1)
        heading_gate = torch.sigmoid(self.phase101_heading_gate(joint_feat))
        range_gate = torch.sigmoid(self.phase101_range_gate(joint_feat))
        heading_router = F.softmax(self.phase101_heading_router(joint_feat), dim=-1)
        range_router = F.softmax(self.phase101_range_router(joint_feat), dim=-1)
        residual_feat, residual_z = self.phase101_residual_bottleneck(joint_feat)

        heading_delta_modes = self.phase101_heading_delta(residual_feat).view(-1, self.phase101_num_modes, 2)
        range_delta_modes = self.phase101_range_delta(residual_feat).view(-1, self.phase101_num_modes, 1)
        heading_delta = (heading_router.unsqueeze(-1) * heading_delta_modes).sum(dim=1)
        range_delta = (range_router.unsqueeze(-1) * range_delta_modes).sum(dim=1)
        heading_delta = torch.tanh(self.phase101_heading_scale) * heading_delta
        range_delta = torch.tanh(self.phase101_range_scale) * range_delta
        heading_delta = torch.clamp(heading_delta, min=-self.phase101_max_heading_delta, max=self.phase101_max_heading_delta)
        range_delta = torch.clamp(range_delta, min=-self.phase101_max_range_delta, max=self.phase101_max_range_delta)
        heading_delta = heading_gate * heading_delta
        range_delta = range_gate * range_delta

        if (not self.training) and float(heading_delta.detach().abs().max().cpu()) == 0.0:
            heading_vec = base_heading
        else:
            heading_vec = F.normalize(base_heading + heading_delta, dim=-1, eps=1e-6)
        range_value = base_range + range_delta
        heading_entropy = -(heading_router * torch.log(heading_router.clamp_min(1e-8))).sum(dim=-1)
        range_entropy = -(range_router * torch.log(range_router.clamp_min(1e-8))).sum(dim=-1)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "phase101_base_heading_vec": base_heading.detach(),
            "phase101_base_range_value": base_range.detach(),
            "phase101_heading_gate": heading_gate.detach().view(-1),
            "phase101_range_gate": range_gate.detach().view(-1),
            "phase101_heading_router_train": heading_router,
            "phase101_range_router_train": range_router,
            "phase101_heading_gate_train": heading_gate,
            "phase101_range_gate_train": range_gate,
            "phase101_heading_router": heading_router.detach(),
            "phase101_range_router": range_router.detach(),
            "phase101_heading_router_entropy": heading_entropy.detach(),
            "phase101_range_router_entropy": range_entropy.detach(),
            "phase101_heading_router_argmax": heading_router.detach().argmax(dim=-1),
            "phase101_range_router_argmax": range_router.detach().argmax(dim=-1),
            "phase101_heading_delta_norm": heading_delta.detach().norm(dim=-1),
            "phase101_range_delta_abs": range_delta.detach().abs().view(-1),
            "phase101_residual_z_norm": residual_z.detach().norm(dim=-1),
        }


class SharedSelectableReadoutPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    """
    Phase88 capacity/readout control.

    Both heading and range consume the same selected decoder-layer fused
    feature. This separates multi-layer readout capacity from axis-specific
    heading specialization.
    """

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        shared_feat = self._heading_features(decout, img_shape)
        heading_vec = F.normalize(self.fc_heading(shared_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(shared_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class DecoderSplitPairUAVHead(RangeH0HeadingSelectableReadoutPairUAVHead):
    # Phase88 decoder-last1 split head. The main decout is the range branch;
    # heading_decout is produced by an independently copied last decoder block.
    def __init__(self, net, heading_readout_layers=None, num_resconv_block=2, heading_fusion_hidden_dim=None):
        self.decoder_split_heading_readout_layers = None if heading_readout_layers is None else tuple(heading_readout_layers)
        if self.decoder_split_heading_readout_layers is None:
            PairUAVHead.__init__(self, net=net, num_resconv_block=num_resconv_block)
        else:
            super().__init__(
                net=net,
                heading_readout_layers=self.decoder_split_heading_readout_layers,
                num_resconv_block=num_resconv_block,
                heading_fusion_hidden_dim=heading_fusion_hidden_dim,
            )

    def forward(self, decout, img_shape, heading_decout=None, **_unused_head_kwargs):
        range_decout = decout
        if heading_decout is None:
            heading_decout = decout
        range_feat = self._extract_features(range_decout, img_shape)
        if self.decoder_split_heading_readout_layers is None:
            heading_feat = self._extract_features(heading_decout, img_shape)
        else:
            heading_feat = self._heading_features(heading_decout, img_shape)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }

def _rotate_heading_vec(heading_vec, delta_rad):
    """Rotate a normalized [cos, sin] heading vector by a bounded residual."""
    delta_rad = delta_rad.view(-1, 1)
    cos_delta = torch.cos(delta_rad)
    sin_delta = torch.sin(delta_rad)
    x = heading_vec[:, 0:1]
    y = heading_vec[:, 1:2]
    rotated = torch.cat(
        [
            x * cos_delta - y * sin_delta,
            x * sin_delta + y * cos_delta,
        ],
        dim=-1,
    )
    return F.normalize(rotated, dim=-1, eps=1e-6)


class RangeH0HeadingH3ResidualPairUAVHead(RangeH0HeadingH3PairUAVHead):
    """
    Rank1-safe residual H5 variant.

    The H0 path remains the base predictor for heading and range. The H3
    branch predicts only a bounded heading-angle residual. With zero residual
    weights, this is an identity wrapper around a loaded H0 checkpoint.
    """

    def __init__(self, net, num_resconv_block=2, heading_residual_max_delta_deg=1.0):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))
        self.heading_residual_fc_delta = nn.Linear(output_dim, 1)
        self.reset_heading_residual_parameters()

    def reset_heading_residual_parameters(self):
        nn.init.zeros_(self.heading_residual_fc_delta.weight)
        nn.init.zeros_(self.heading_residual_fc_delta.bias)

    def reset_after_model_init(self):
        self.reset_heading_residual_parameters()

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        base_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(base_feat)

        residual_feat = self._heading_features(decout, img_shape)
        delta_rad = self.heading_residual_max_delta_rad * torch.tanh(
            self.heading_residual_fc_delta(residual_feat)
        )
        heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "heading_base_vec": base_heading_vec.detach(),
            "heading_residual_delta_rad": delta_rad,
            "heading_residual_delta_deg": delta_rad * (180.0 / math.pi),
        }


class RangeH0HeadingEarlyMidLateResidualPairUAVHead(RangeH0HeadingEarlyMidLatePairUAVHead):
    """
    Rank1-safe residual H8 variant.

    The loaded H0 path remains the base predictor. Early/mid/late decoder
    evidence only predicts a bounded heading residual, leaving range untouched.
    """

    def __init__(self, net, num_resconv_block=2, heading_residual_max_delta_deg=1.0):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.heading_residual_max_delta_rad = math.radians(float(heading_residual_max_delta_deg))
        self.heading_residual_fc_delta = nn.Linear(output_dim, 1)
        self.reset_heading_residual_parameters()

    def reset_heading_residual_parameters(self):
        nn.init.zeros_(self.heading_residual_fc_delta.weight)
        nn.init.zeros_(self.heading_residual_fc_delta.bias)

    def reset_after_model_init(self):
        self.reset_heading_residual_parameters()

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        base_feat = self._extract_features(decout, img_shape)
        base_heading_vec = F.normalize(self.fc_heading(base_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(base_feat)

        residual_feat = self._heading_features(decout, img_shape)
        delta_rad = self.heading_residual_max_delta_rad * torch.tanh(
            self.heading_residual_fc_delta(residual_feat)
        )
        heading_vec = _rotate_heading_vec(base_heading_vec, delta_rad)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
            "heading_base_vec": base_heading_vec.detach(),
            "heading_residual_delta_rad": delta_rad,
            "heading_residual_delta_deg": delta_rad * (180.0 / math.pi),
        }


class DualTaskTokenPairUAVHead(PairUAVHead):
    """
    H9: use learnable heading/range task tokens to cross-attend decoder tokens
    and produce axis-specific readouts without the convolutional H0 extractor.
    """

    def __init__(self, net, num_resconv_block=2, task_token_num_heads=8):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        token_dim = net.dec_embed_dim
        num_heads = int(task_token_num_heads)
        if token_dim % num_heads != 0:
            num_heads = 1

        self.task_tokens = nn.Parameter(torch.zeros(2, token_dim))
        nn.init.normal_(self.task_tokens, std=0.02)
        self.task_cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.task_norm = nn.LayerNorm(token_dim)
        self.task_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.heading_token_mlp = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.range_token_mlp = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def _task_features(self, decout):
        tokens = decout[-1]
        B = tokens.shape[0]
        task_tokens = self.task_tokens.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.task_cross_attn(task_tokens, tokens, tokens, need_weights=False)
        task_tokens = self.task_norm(task_tokens + attended)
        task_tokens = self.task_norm(task_tokens + self.task_ffn(task_tokens))
        heading_task = task_tokens[:, 0]
        range_task = task_tokens[:, 1]
        return self.heading_token_mlp(heading_task), self.range_token_mlp(range_task)

    def forward(self, decout, img_shape, **_unused_head_kwargs):
        del img_shape
        heading_feat, range_feat = self._task_features(decout)
        heading_vec = F.normalize(self.fc_heading(heading_feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(range_feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class LocalAlignmentPairUAVHead(PairUAVHead):
    """
    PairUAV heading/range head with a zero-initialized residual adapter fed by
    an 8D local cross-token alignment summary.
    """

    def __init__(self, net, num_resconv_block=2, alignment_dropout=0.0):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.alignment_adapter = nn.Sequential(
            nn.Linear(output_dim + 8, output_dim),
            nn.ReLU(),
            nn.Dropout(float(alignment_dropout)),
            nn.Linear(output_dim, output_dim),
        )
        self.alignment_adapter_scale = nn.Parameter(torch.zeros(()))

    def forward(self, decout, img_shape, paired_decout=None, **_unused_head_kwargs):
        feat = self._extract_features(decout, img_shape)
        alignment_summary = compute_local_alignment_summary(decout, paired_decout).to(device=feat.device, dtype=feat.dtype)
        adapter_delta = self.alignment_adapter(torch.cat([feat, alignment_summary], dim=-1))
        feat = feat + torch.tanh(self.alignment_adapter_scale) * adapter_delta

        heading_vec = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class FrozenMatcherFusionPairUAVHead(PairUAVHead):
    """
    PairUAV heading/range head with a zero-initialized residual adapter fed by
    frozen external matcher packet features.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        matcher_feature_dim=13,
        matcher_hidden_dim=64,
        matcher_dropout=0.0,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.matcher_feature_dim = int(matcher_feature_dim)
        self.matcher_adapter = nn.Sequential(
            nn.Linear(output_dim + self.matcher_feature_dim, int(matcher_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(matcher_dropout)),
            nn.Linear(int(matcher_hidden_dim), output_dim),
        )
        self.matcher_adapter_scale = nn.Parameter(torch.zeros(()))

    def _matcher_inputs(self, matcher_features, matcher_feature_mask, batch_size, device, dtype):
        if matcher_features is None:
            features = torch.zeros(batch_size, self.matcher_feature_dim, device=device, dtype=dtype)
        else:
            features = torch.as_tensor(matcher_features, device=device, dtype=dtype).view(batch_size, -1)
            if features.shape[1] != self.matcher_feature_dim:
                raise ValueError(
                    f"matcher_features dim {features.shape[1]} does not match expected {self.matcher_feature_dim}"
                )

        if matcher_feature_mask is None:
            mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            mask = torch.as_tensor(matcher_feature_mask, device=device, dtype=dtype).view(batch_size, -1)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
        return features, mask

    def forward(self, decout, img_shape, matcher_features=None, matcher_feature_mask=None, **_unused_head_kwargs):
        feat = self._extract_features(decout, img_shape)
        features, mask = self._matcher_inputs(
            matcher_features,
            matcher_feature_mask,
            batch_size=feat.shape[0],
            device=feat.device,
            dtype=feat.dtype,
        )
        adapter_delta = self.matcher_adapter(torch.cat([feat, features], dim=-1))
        feat = feat + torch.tanh(self.matcher_adapter_scale) * mask * adapter_delta

        heading_vec = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }


class AngleSpecialistPairUAVHead(PairUAVHead):
    """
    Rank1-compatible angle-only specialist.

    The base PairUAV heading/range modules keep the same names as PairUAVHead,
    so checkpoints trained with output_mode='pairuav_heading_range' load into
    this head. The specialist branch only adds a bounded heading residual and
    leaves range_value on the base path.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        angle_specialist_hidden_dim=256,
        angle_specialist_dropout=0.0,
        angle_specialist_max_residual=0.10,
        angle_specialist_init_scale=0.0,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        hidden_dim = int(angle_specialist_hidden_dim)
        self.angle_specialist_max_residual = float(angle_specialist_max_residual)
        self.angle_specialist_adapter = nn.Sequential(
            nn.Linear(output_dim + 8 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(angle_specialist_dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.angle_specialist_gate = nn.Linear(hidden_dim, 1)
        self.angle_specialist_delta_heading = nn.Linear(hidden_dim, 2)
        self.angle_specialist_scale = nn.Parameter(torch.tensor(float(angle_specialist_init_scale)))

    def forward(self, decout, img_shape, paired_decout=None, **_unused_head_kwargs):
        feat = self._extract_features(decout, img_shape)
        base_heading = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        base_range = self.fc_range(feat)
        alignment_summary = compute_local_alignment_summary(decout, paired_decout).to(
            device=feat.device,
            dtype=feat.dtype,
        )

        # The base prediction is used as an observability cue, not as a second
        # path for angle-loss gradients into the range head.
        base_observability = torch.cat([base_heading.detach(), base_range.detach()], dim=-1)
        specialist_input = torch.cat([feat, alignment_summary, base_observability], dim=-1)
        specialist_feat = self.angle_specialist_adapter(specialist_input)
        gate = torch.sigmoid(self.angle_specialist_gate(specialist_feat))
        scale = torch.tanh(self.angle_specialist_scale)
        delta = self.angle_specialist_delta_heading(specialist_feat)
        residual = torch.clamp(
            gate * scale * delta,
            min=-self.angle_specialist_max_residual,
            max=self.angle_specialist_max_residual,
        )
        heading_vec = F.normalize(base_heading + residual, dim=-1, eps=1e-6)
        return {
            "heading_vec": heading_vec,
            "range_value": base_range,
            "base_heading_vec": base_heading,
            "angle_specialist_gate": gate,
            "angle_specialist_residual": residual,
            "angle_specialist_scale": scale.view(1),
        }


class SelectiveCorrespondencePairUAVHead(PairUAVHead):
    """
    B-SCR head: keep base range unchanged and apply a bounded, gated heading residual
    from structured frozen correspondence evidence.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        bscr_global_dim=12,
        bscr_grid_size=4,
        bscr_topk=16,
        bscr_hidden_dim=128,
        bscr_dropout=0.0,
        bscr_max_residual=0.25,
        force_gate_off=False,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        self.bscr_global_dim = int(bscr_global_dim)
        self.bscr_grid_size = int(bscr_grid_size)
        self.bscr_topk = int(bscr_topk)
        self.bscr_spatial_dim = self.bscr_grid_size * self.bscr_grid_size * 4
        self.bscr_anchor_dim = self.bscr_topk * 5
        self.bscr_packet_dim = self.bscr_global_dim + self.bscr_spatial_dim + self.bscr_anchor_dim + 2
        self.bscr_max_residual = float(bscr_max_residual)
        self.force_gate_off = bool(force_gate_off)

        hidden_dim = int(bscr_hidden_dim)
        self.bscr_encoder = nn.Sequential(
            nn.Linear(self.bscr_packet_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(bscr_dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.bscr_gate = nn.Linear(hidden_dim, 1)
        self.bscr_delta_heading = nn.Linear(hidden_dim, 2)
        self.bscr_residual_scale = nn.Parameter(torch.zeros(()))

    def _bscr_inputs(
        self,
        bscr_global_stats,
        bscr_spatial_bins,
        bscr_topk_anchors,
        bscr_quality_mask,
        bscr_fallback_used,
        batch_size,
        device,
        dtype,
    ):
        if bscr_global_stats is None or bscr_spatial_bins is None or bscr_topk_anchors is None:
            raise ValueError("B-SCR mode requires bscr_global_stats, bscr_spatial_bins, and bscr_topk_anchors")

        global_stats = torch.as_tensor(bscr_global_stats, device=device, dtype=dtype).view(batch_size, -1)
        spatial = torch.as_tensor(bscr_spatial_bins, device=device, dtype=dtype).view(batch_size, -1)
        anchors = torch.as_tensor(bscr_topk_anchors, device=device, dtype=dtype).view(batch_size, -1)
        if global_stats.shape[1] != self.bscr_global_dim:
            raise ValueError(f"bscr_global_stats dim {global_stats.shape[1]} != {self.bscr_global_dim}")
        if spatial.shape[1] != self.bscr_spatial_dim:
            raise ValueError(f"bscr_spatial_bins flat dim {spatial.shape[1]} != {self.bscr_spatial_dim}")
        if anchors.shape[1] != self.bscr_anchor_dim:
            raise ValueError(f"bscr_topk_anchors flat dim {anchors.shape[1]} != {self.bscr_anchor_dim}")

        if bscr_quality_mask is None:
            quality = torch.ones(batch_size, 1, device=device, dtype=dtype)
        else:
            quality = torch.as_tensor(bscr_quality_mask, device=device, dtype=dtype).view(batch_size, -1)[:, :1]
        if bscr_fallback_used is None:
            fallback = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            fallback = torch.as_tensor(bscr_fallback_used, device=device, dtype=dtype).view(batch_size, -1)[:, :1]

        packet = torch.cat([global_stats, spatial, anchors, quality, fallback], dim=-1)
        packet = torch.nan_to_num(packet, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = quality * (1.0 - torch.clamp(fallback, 0.0, 1.0))
        return packet, valid_mask

    def forward(
        self,
        decout,
        img_shape,
        bscr_global_stats=None,
        bscr_spatial_bins=None,
        bscr_topk_anchors=None,
        bscr_quality_mask=None,
        bscr_fallback_used=None,
        **_unused_head_kwargs,
    ):
        feat = self._extract_features(decout, img_shape)
        heading_base = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        range_base = self.fc_range(feat)
        packet, valid_mask = self._bscr_inputs(
            bscr_global_stats,
            bscr_spatial_bins,
            bscr_topk_anchors,
            bscr_quality_mask,
            bscr_fallback_used,
            batch_size=feat.shape[0],
            device=feat.device,
            dtype=feat.dtype,
        )

        evidence = self.bscr_encoder(packet)
        gate = torch.sigmoid(self.bscr_gate(evidence)) * valid_mask
        if self.force_gate_off:
            gate = torch.zeros_like(gate)
        delta = self.bscr_delta_heading(evidence)
        residual = torch.clamp(
            gate * torch.tanh(self.bscr_residual_scale) * delta,
            min=-self.bscr_max_residual,
            max=self.bscr_max_residual,
        )
        heading_vec = F.normalize(heading_base + residual, dim=-1, eps=1e-6)
        return {
            "heading_vec": heading_vec,
            "range_value": range_base,
            "bscr_gate": gate,
            "bscr_heading_residual": residual,
        }


class TargetConditionedPairUAVHead(PairUAVHead):
    """
    PairUAV heading/range head with a lightweight target/group adapter.
    The residual adapter is zero-scaled at initialization so partial checkpoint
    loading starts close to the non-conditioned head.
    """

    def __init__(
        self,
        net,
        num_resconv_block=2,
        num_target_groups=4096,
        target_embed_dim=32,
        target_adapter_dropout=0.0,
    ):
        super().__init__(net=net, num_resconv_block=num_resconv_block)
        output_dim = 4 * self.patch_size ** 2
        self.num_target_groups = int(num_target_groups)
        self.target_embedding = nn.Embedding(self.num_target_groups, int(target_embed_dim))
        self.target_adapter = nn.Sequential(
            nn.Linear(output_dim + int(target_embed_dim), output_dim),
            nn.ReLU(),
            nn.Dropout(float(target_adapter_dropout)),
            nn.Linear(output_dim, output_dim),
        )
        self.adapter_scale = nn.Parameter(torch.zeros(()))

    def _target_ids(self, target_group_index, batch_size, device):
        if target_group_index is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        target_ids = torch.as_tensor(target_group_index, dtype=torch.long, device=device).view(-1)
        if target_ids.numel() != batch_size:
            raise ValueError(f"target_group_index has {target_ids.numel()} values for batch size {batch_size}")
        return torch.remainder(target_ids, self.num_target_groups)

    def forward(self, decout, img_shape, target_group_index=None):
        feat = self._extract_features(decout, img_shape)
        target_ids = self._target_ids(target_group_index, feat.shape[0], feat.device)
        target_feat = self.target_embedding(target_ids)
        adapter_delta = self.target_adapter(torch.cat([feat, target_feat], dim=-1))
        feat = feat + torch.tanh(self.adapter_scale) * adapter_delta

        heading_vec = F.normalize(self.fc_heading(feat), dim=-1, eps=1e-6)
        range_value = self.fc_range(feat)
        return {
            "heading_vec": heading_vec,
            "range_value": range_value,
        }
