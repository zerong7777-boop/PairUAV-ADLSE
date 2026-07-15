from copy import deepcopy
import os
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial
import reloc3r.utils.path_to_croco
from reloc3r.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
from models.blocks import Block, DecoderBlock
from reloc3r.pose_head import (
    AngleSpecialistPairUAVHead,
    AxisDecoupledPairUAVHead,
    AxisAsyncQueryBridgePairUAVHead,
    DecoderSplitPairUAVHead,
    DualTaskTokenPairUAVHead,
    EarlySplitPairUAVHead,
    FrozenMatcherFusionPairUAVHead,
    LocalAlignmentPairUAVHead,
    MidSplitPairUAVHead,
    MNRAnchoredResidualBottleneckPairUAVHead,
    PairUAVHead,
    Phase104bFDERPairUAVHead,
    Phase104cObservabilityFactorRouterHead,
    Phase104dPolarRelationMemoryHead,
    Phase104eProtectedAxisAsymmetricExpertHead,
    Phase96RegimeConditionedResidualPairUAVHead,
    Phase99StabilityAwareReadoutPairUAVHead,
    Phase101AxiswiseTrajectoryReadoutPairUAVHead,
    PoseHead,
    RangeAnchoredHeadingResidualPairUAVHead,
    RangeH0HeadingH2PairUAVHead,
    RangeH0HeadingH3PairUAVHead,
    RangeH0HeadingH3ResidualPairUAVHead,
    RangeH0HeadingEarlyMidLatePairUAVHead,
    RangeH0HeadingSelectableReadoutPairUAVHead,
    RangeH0HeadingEarlyMidLateResidualPairUAVHead,
    SelectiveCorrespondencePairUAVHead,
    SharedSelectableReadoutPairUAVHead,
    TargetConditionedPairUAVHead,
)
from reloc3r.trainable_policy import apply_trainable_policy
from reloc3r.utils.misc import freeze_all_params, transpose_to_landscape
from pdb import set_trace as bb
from huggingface_hub import PyTorchModelHubMixin


# parts of the code adapted from 
# 'https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/croco.py#L21'
# 'https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/model.py#L46'
class Reloc3rRelpose(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=512,          # input image size
                 patch_size=16,         # patch_size 
                 enc_embed_dim=1024,    # encoder feature dimension
                 enc_depth=24,          # encoder depth 
                 enc_num_heads=16,      # encoder number of heads in the transformer block 
                 dec_embed_dim=768,     # decoder feature dimension 
                 dec_depth=12,          # decoder depth 
                 dec_num_heads=12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                 output_mode='pose',    # downstream output type: pose | pairuav_heading_range
                 num_target_groups=4096,
                 target_embed_dim=32,
                 target_adapter_dropout=0.0,
                 alignment_dropout=0.0,
                 matcher_feature_dim=13,
                 matcher_hidden_dim=64,
                 matcher_dropout=0.0,
                 bscr_global_dim=12,
                 bscr_grid_size=4,
                 bscr_topk=16,
                 bscr_hidden_dim=128,
                 bscr_dropout=0.0,
                 bscr_max_residual=0.25,
                 bscr_force_gate_off=False,
                 angle_specialist_hidden_dim=256,
                 angle_specialist_dropout=0.0,
                 angle_specialist_max_residual=0.10,
                 angle_specialist_init_scale=0.0,
                 axis_branch_hidden_dim=256,
                 axis_branch_dropout=0.0,
                 axis_branch_init_scale=0.0,
                 heading_residual_max_delta_deg=1.0,
                 mnr_bottleneck_dim=64,
                 mnr_dropout=0.0,
                 mnr_residual_scale=1.0,
                 mnr_max_heading_delta=0.05,
                 mnr_max_range_delta=0.25,
                 phase104_task_token_num_heads=8,
                 phase104_bridge_hidden_dim=128,
                 phase104c_heading_residual_max_delta_deg=5.0,
                 phase104c_router_hidden_dim=256,
                 phase104d_heading_residual_max_delta_deg=5.0,
                 phase104d_slot_hidden_dim=256,
                 phase104d_heading_bin_count=12,
                 phase104d_range_bin_count=8,
                 phase104d_source_pool_grid=(12, 16),
                ):
        super(Reloc3rRelpose, self).__init__()

        # patchify and positional embedding
        self.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)

        # ViT encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        self.decoder_last1_split_modes = {
            'pairuav_decoder_last1_split_h0_heading_range',
            'pairuav_decoder_last1_split_h8_heading_range',
        }
        self._heading_dec_blocks_synced = False
        if output_mode in self.decoder_last1_split_modes:
            self.heading_dec_blocks = nn.ModuleList([deepcopy(self.dec_blocks[-1])])

        # downstream regression head
        self.output_mode = output_mode
        if self.output_mode == 'pose':
            self.pose_head = PoseHead(net=self)
        elif self.output_mode == 'pairuav_heading_range':
            self.pose_head = PairUAVHead(net=self)
        elif self.output_mode == 'pairuav_target_conditioned_heading_range':
            self.pose_head = TargetConditionedPairUAVHead(
                net=self,
                num_target_groups=num_target_groups,
                target_embed_dim=target_embed_dim,
                target_adapter_dropout=target_adapter_dropout,
            )
        elif self.output_mode == 'pairuav_local_alignment_heading_range':
            self.pose_head = LocalAlignmentPairUAVHead(
                net=self,
                alignment_dropout=alignment_dropout,
            )
        elif self.output_mode == 'pairuav_frozen_matcher_fusion_heading_range':
            self.pose_head = FrozenMatcherFusionPairUAVHead(
                net=self,
                matcher_feature_dim=matcher_feature_dim,
                matcher_hidden_dim=matcher_hidden_dim,
                matcher_dropout=matcher_dropout,
            )
        elif self.output_mode == 'pairuav_selective_correspondence_heading_range':
            self.pose_head = SelectiveCorrespondencePairUAVHead(
                net=self,
                bscr_global_dim=bscr_global_dim,
                bscr_grid_size=bscr_grid_size,
                bscr_topk=bscr_topk,
                bscr_hidden_dim=bscr_hidden_dim,
                bscr_dropout=bscr_dropout,
                bscr_max_residual=bscr_max_residual,
                force_gate_off=bscr_force_gate_off,
            )
        elif self.output_mode == 'pairuav_angle_specialist_heading_range':
            self.pose_head = AngleSpecialistPairUAVHead(
                net=self,
                angle_specialist_hidden_dim=angle_specialist_hidden_dim,
                angle_specialist_dropout=angle_specialist_dropout,
                angle_specialist_max_residual=angle_specialist_max_residual,
                angle_specialist_init_scale=angle_specialist_init_scale,
            )
        elif self.output_mode == 'pairuav_axis_decoupled_heading_range':
            self.pose_head = AxisDecoupledPairUAVHead(
                net=self,
                axis_branch_hidden_dim=axis_branch_hidden_dim,
                axis_branch_dropout=axis_branch_dropout,
                axis_branch_init_scale=axis_branch_init_scale,
            )
        elif self.output_mode == 'pairuav_mid_split_heading_range':
            self.pose_head = MidSplitPairUAVHead(net=self)
        elif self.output_mode == 'pairuav_early_split_heading_range':
            self.pose_head = EarlySplitPairUAVHead(net=self)
        elif self.output_mode == 'pairuav_range_h0_heading_h2_heading_range':
            self.pose_head = RangeH0HeadingH2PairUAVHead(net=self)
        elif self.output_mode == 'pairuav_range_h0_heading_h3_heading_range':
            self.pose_head = RangeH0HeadingH3PairUAVHead(net=self)
        elif self.output_mode == 'pairuav_range_h0_heading_early_mid_late_heading_range':
            self.pose_head = RangeH0HeadingEarlyMidLatePairUAVHead(net=self)
        elif self.output_mode == 'pairuav_decoder_last1_split_h0_heading_range':
            self.pose_head = DecoderSplitPairUAVHead(net=self, heading_readout_layers=None)
        elif self.output_mode == 'pairuav_decoder_last1_split_h8_heading_range':
            self.pose_head = DecoderSplitPairUAVHead(
                net=self,
                heading_readout_layers=("early", "mid", "late"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_late_only_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("late",),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_mid_late_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid", "late"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_late_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early", "late"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_mid_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early", "mid"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_mid_only_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid",),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_only_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early",),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_late_late_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("late", "late"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_mid_mid_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid", "mid"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_late_mid_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("late", "mid"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_mid_mid_mid_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid", "mid", "mid"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_early_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early", "early"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_mid_late_reduced_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid", "late"),
                heading_fusion_hidden_dim=512,
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_mid_late_reduced_heading_range':
            self.pose_head = RangeH0HeadingSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early", "mid", "late"),
                heading_fusion_hidden_dim=512,
            )
        elif self.output_mode == 'pairuav_shared_mid_late_capacity_heading_range':
            self.pose_head = SharedSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("mid", "late"),
            )
        elif self.output_mode == 'pairuav_shared_early_mid_late_capacity_heading_range':
            self.pose_head = SharedSelectableReadoutPairUAVHead(
                net=self,
                heading_readout_layers=("early", "mid", "late"),
            )
        elif self.output_mode == 'pairuav_range_h0_heading_h3_residual_heading_range':
            self.pose_head = RangeH0HeadingH3ResidualPairUAVHead(
                net=self,
                heading_residual_max_delta_deg=heading_residual_max_delta_deg,
            )
        elif self.output_mode == 'pairuav_range_h0_heading_early_mid_late_residual_heading_range':
            self.pose_head = RangeH0HeadingEarlyMidLateResidualPairUAVHead(
                net=self,
                heading_residual_max_delta_deg=heading_residual_max_delta_deg,
            )
        elif self.output_mode == 'pairuav_phase104_qbridge_fixed_h8_heading_range':
            self.pose_head = AxisAsyncQueryBridgePairUAVHead(
                net=self,
                heading_layers=("mid", "late"),
                range_layers=("late",),
                learnable_layer_weights=False,
                use_gated_bridge=False,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104_qbridge_learn_layer_no_bridge_heading_range':
            self.pose_head = AxisAsyncQueryBridgePairUAVHead(
                net=self,
                heading_layers=("early", "mid", "late"),
                range_layers=("early", "mid", "late"),
                learnable_layer_weights=True,
                use_gated_bridge=False,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104_axis_async_qbridge_heading_range':
            self.pose_head = AxisAsyncQueryBridgePairUAVHead(
                net=self,
                heading_layers=("mid", "late"),
                range_layers=("late",),
                learnable_layer_weights=False,
                use_gated_bridge=False,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104_axis_async_qbridge_gated_heading_range':
            self.pose_head = AxisAsyncQueryBridgePairUAVHead(
                net=self,
                heading_layers=("mid", "late"),
                range_layers=("late",),
                learnable_layer_weights=False,
                use_gated_bridge=True,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104_qbridge_learn_layer_gated_heading_range':
            self.pose_head = AxisAsyncQueryBridgePairUAVHead(
                net=self,
                heading_layers=("early", "mid", "late"),
                range_layers=("early", "mid", "late"),
                learnable_layer_weights=True,
                use_gated_bridge=True,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104_range_anchored_heading_residual_heading_range':
            self.pose_head = RangeAnchoredHeadingResidualPairUAVHead(
                net=self,
                heading_residual_max_delta_deg=heading_residual_max_delta_deg,
            )
        elif self.output_mode == 'pairuav_phase104b_fder_fixed_heading_range':
            self.pose_head = Phase104bFDERPairUAVHead(
                net=self,
                use_sample_router=False,
                heading_residual_max_delta_deg=heading_residual_max_delta_deg,
                task_token_num_heads=phase104_task_token_num_heads,
            )
        elif self.output_mode == 'pairuav_phase104b_fder_heading_range':
            self.pose_head = Phase104bFDERPairUAVHead(
                net=self,
                use_sample_router=True,
                heading_residual_max_delta_deg=heading_residual_max_delta_deg,
                task_token_num_heads=phase104_task_token_num_heads,
            )
        elif self.output_mode == 'pairuav_phase104c_offer_fixed_heading_range':
            self.pose_head = Phase104cObservabilityFactorRouterHead(
                net=self,
                use_observability_router=False,
                heading_residual_max_delta_deg=phase104c_heading_residual_max_delta_deg,
                router_hidden_dim=phase104c_router_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
            )
        elif self.output_mode == 'pairuav_phase104c_offer_heading_range':
            self.pose_head = Phase104cObservabilityFactorRouterHead(
                net=self,
                use_observability_router=True,
                heading_residual_max_delta_deg=phase104c_heading_residual_max_delta_deg,
                router_hidden_dim=phase104c_router_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
            )
        elif self.output_mode == 'pairuav_phase104e_paaer_hard_heading_range':
            self.pose_head = Phase104eProtectedAxisAsymmetricExpertHead(
                net=self,
                use_blend=False,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104e_paaer_blend_heading_range':
            self.pose_head = Phase104eProtectedAxisAsymmetricExpertHead(
                net=self,
                use_blend=True,
                task_token_num_heads=phase104_task_token_num_heads,
                bridge_hidden_dim=phase104_bridge_hidden_dim,
                blend_hidden_dim=phase104_bridge_hidden_dim,
            )
        elif self.output_mode == 'pairuav_phase104d_prm_r0_heading_range':
            self.pose_head = Phase104dPolarRelationMemoryHead(
                net=self,
                use_auxiliary=False,
                heading_bin_count=phase104d_heading_bin_count,
                range_bin_count=phase104d_range_bin_count,
                heading_residual_max_delta_deg=phase104d_heading_residual_max_delta_deg,
                slot_hidden_dim=phase104d_slot_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
                source_pool_grid=phase104d_source_pool_grid,
            )
        elif self.output_mode == 'pairuav_phase104d_prm_r1_aux_heading_range':
            self.pose_head = Phase104dPolarRelationMemoryHead(
                net=self,
                use_auxiliary=True,
                heading_bin_count=phase104d_heading_bin_count,
                range_bin_count=phase104d_range_bin_count,
                heading_residual_max_delta_deg=phase104d_heading_residual_max_delta_deg,
                slot_hidden_dim=phase104d_slot_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
                source_pool_grid=phase104d_source_pool_grid,
            )
        elif self.output_mode == 'pairuav_phase104d_prm_r2_direct_heading_range':
            self.pose_head = Phase104dPolarRelationMemoryHead(
                net=self,
                use_auxiliary=False,
                heading_readout_mode="direct_memory",
                heading_bin_count=phase104d_heading_bin_count,
                range_bin_count=phase104d_range_bin_count,
                heading_residual_max_delta_deg=phase104d_heading_residual_max_delta_deg,
                slot_hidden_dim=phase104d_slot_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
                source_pool_grid=phase104d_source_pool_grid,
            )
        elif self.output_mode == 'pairuav_phase104d_prm_r3_bounded_delta_heading_range':
            self.pose_head = Phase104dPolarRelationMemoryHead(
                net=self,
                use_auxiliary=False,
                heading_readout_mode="bounded_memory_delta",
                heading_bin_count=phase104d_heading_bin_count,
                range_bin_count=phase104d_range_bin_count,
                heading_residual_max_delta_deg=20.0,
                slot_hidden_dim=phase104d_slot_hidden_dim,
                task_token_num_heads=phase104_task_token_num_heads,
                source_pool_grid=phase104d_source_pool_grid,
            )
        elif self.output_mode == 'pairuav_dual_task_token_heading_range':
            self.pose_head = DualTaskTokenPairUAVHead(net=self)
        elif self.output_mode == 'pairuav_mnr_h8_residual_bottleneck_heading_range':
            self.pose_head = MNRAnchoredResidualBottleneckPairUAVHead(
                net=self,
                mnr_bottleneck_dim=mnr_bottleneck_dim,
                mnr_dropout=mnr_dropout,
                mnr_residual_scale=mnr_residual_scale,
                mnr_max_heading_delta=mnr_max_heading_delta,
                mnr_max_range_delta=mnr_max_range_delta,
            )
        elif self.output_mode == 'pairuav_phase101_axiswise_trajectory_readout':
            self.pose_head = Phase101AxiswiseTrajectoryReadoutPairUAVHead(
                net=self,
                bottleneck_dim=mnr_bottleneck_dim,
                dropout=mnr_dropout,
                residual_scale=mnr_residual_scale,
                max_heading_delta=mnr_max_heading_delta,
                max_range_delta=mnr_max_range_delta,
            )
        elif self.output_mode == 'pairuav_phase99_stability_readout':
            self.pose_head = Phase99StabilityAwareReadoutPairUAVHead(
                net=self,
                bottleneck_dim=mnr_bottleneck_dim,
                dropout=mnr_dropout,
                residual_scale=mnr_residual_scale,
                max_heading_delta=mnr_max_heading_delta,
                max_range_delta=mnr_max_range_delta,
            )
        elif self.output_mode == 'pairuav_phase96_regime_residual_heading_range':
            self.pose_head = Phase96RegimeConditionedResidualPairUAVHead(
                net=self,
                bottleneck_dim=mnr_bottleneck_dim,
                dropout=mnr_dropout,
                residual_scale=mnr_residual_scale,
                max_heading_delta=mnr_max_heading_delta,
                max_range_delta=mnr_max_range_delta,
            )
        else:
            raise ValueError(f'Unsupported output_mode={self.output_mode}')
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        self.initialize_weights() 

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # linears and layer norms
        self.apply(self._init_weights)
        if hasattr(self, "pose_head") and hasattr(self.pose_head, "reset_after_model_init"):
            self.pose_head.reset_after_model_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_encoder(self):
        freeze_all_params([self.patch_embed, self.enc_blocks])

    def freeze_except_angle_specialist(self):
        return apply_trainable_policy(self, "angle_specialist")

    def freeze_for_trainable_policy(self, policy):
        return apply_trainable_policy(self, policy)

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encoder(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk in self.dec_blocks:
            # img1 side
            f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _sync_heading_decoder_split(self):
        if not hasattr(self, 'heading_dec_blocks'):
            return
        if self._heading_dec_blocks_synced:
            return
        self.heading_dec_blocks[0].load_state_dict(self.dec_blocks[-1].state_dict())
        self._heading_dec_blocks_synced = True

    def _decoder_last1_split(self, f1, pos1, f2, pos2):
        self._sync_heading_decoder_split()
        common_output = [(f1, f2)]  # before projection

        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        common_output.append((f1, f2))

        for blk in self.dec_blocks[:-1]:
            f1, _ = blk(*common_output[-1][::+1], pos1, pos2)
            f2, _ = blk(*common_output[-1][::-1], pos2, pos1)
            common_output.append((f1, f2))

        range_f1, _ = self.dec_blocks[-1](*common_output[-1][::+1], pos1, pos2)
        range_f2, _ = self.dec_blocks[-1](*common_output[-1][::-1], pos2, pos1)
        heading_blk = self.heading_dec_blocks[0]
        heading_f1, _ = heading_blk(*common_output[-1][::+1], pos1, pos2)
        heading_f2, _ = heading_blk(*common_output[-1][::-1], pos2, pos1)

        del common_output[1]  # duplicate with common_output[0]
        range_output = list(common_output) + [(range_f1, range_f2)]
        heading_output = list(common_output) + [(heading_f1, heading_f2)]
        range_output[-1] = tuple(map(self.dec_norm, range_output[-1]))
        heading_output[-1] = tuple(map(self.dec_norm, heading_output[-1]))
        return zip(*range_output), zip(*heading_output)

    def _downstream_head(self, decout, img_shape, view=None, paired_decout=None, heading_decout=None):
        head_kwargs = {}
        if view is not None and 'target_group_index' in view:
            head_kwargs['target_group_index'] = view['target_group_index']
        if paired_decout is not None:
            head_kwargs['paired_decout'] = paired_decout
        if heading_decout is not None:
            head_kwargs['heading_decout'] = heading_decout
        if view is not None and 'matcher_features' in view:
            head_kwargs['matcher_features'] = view['matcher_features']
        if view is not None and 'matcher_feature_mask' in view:
            head_kwargs['matcher_feature_mask'] = view['matcher_feature_mask']
        for key in (
            'bscr_global_stats',
            'bscr_spatial_bins',
            'bscr_topk_anchors',
            'bscr_quality_mask',
            'bscr_fallback_used',
            'phase104d_mask_slot',
        ):
            if view is not None and key in view:
                head_kwargs[key] = view[key]
        return self.head(decout, img_shape, **head_kwargs)

    def forward(self, view1, view2):
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(view1, view2)  # B,S,D

        if self.output_mode in getattr(self, 'decoder_last1_split_modes', set()):
            (range_dec1, range_dec2), (heading_dec1, heading_dec2) = self._decoder_last1_split(feat1, pos1, feat2, pos2)
            with torch.cuda.amp.autocast(enabled=False):
                range_dec1 = [tok.float() for tok in range_dec1]
                range_dec2 = [tok.float() for tok in range_dec2]
                heading_dec1 = [tok.float() for tok in heading_dec1]
                heading_dec2 = [tok.float() for tok in heading_dec2]
                pose1 = self._downstream_head(
                    range_dec1,
                    shape1,
                    view=view1,
                    paired_decout=range_dec2,
                    heading_decout=heading_dec1,
                )
                pose2 = self._downstream_head(
                    range_dec2,
                    shape2,
                    view=view2,
                    paired_decout=range_dec1,
                    heading_decout=heading_dec2,
                )
            return pose1, pose2

        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            dec1 = [tok.float() for tok in dec1]
            dec2 = [tok.float() for tok in dec2]
            pose1 = self._downstream_head(dec1, shape1, view=view1, paired_decout=dec2)
            pose2 = self._downstream_head(dec2, shape2, view=view2, paired_decout=dec1)  # relative camera pose from 2 to 1. 
            
        return pose1, pose2


def setup_reloc3r_relpose_model(model_args, device):
    if '224' in model_args:
        ckpt_path = 'siyan824/reloc3r-224'
    elif '512' in model_args:
        ckpt_path = 'siyan824/reloc3r-512'
    reloc3r_relpose = Reloc3rRelpose.from_pretrained(ckpt_path)
    reloc3r_relpose.to(device)
    reloc3r_relpose.eval()
    print('Model loaded from ', ckpt_path)
    return reloc3r_relpose


@torch.no_grad()
def inference_relpose(batch, model, device, use_amp=False): 
    # to device. 
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split():  
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        _, pose2 = model(view1, view2)
    pose2to1 = pose2["pose"]
    return pose2to1
