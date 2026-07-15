from __future__ import annotations

import torch
import torch.nn.functional as F

from reloc3r.loss import PairUAVOfficialMetricAwareLoss


class PairUAVTailWeightedOfficialLoss(PairUAVOfficialMetricAwareLoss):
    """Official-like PairUAV loss with extra range pressure on high-|range| samples."""

    def __init__(
        self,
        heading_weight=1.0,
        range_weight=1.0,
        angle_floor_deg=1.0,
        distance_floor=1.0,
        absolute_heading_weight=0.05,
        absolute_range_weight=0.10,
        tail_start=80.0,
        tail_end=120.0,
        tail_max_weight=3.0,
    ):
        super().__init__(
            heading_weight=heading_weight,
            range_weight=range_weight,
            angle_floor_deg=angle_floor_deg,
            distance_floor=distance_floor,
            absolute_heading_weight=absolute_heading_weight,
            absolute_range_weight=absolute_range_weight,
        )
        self.tail_start = float(tail_start)
        self.tail_end = float(tail_end)
        self.tail_max_weight = float(tail_max_weight)

    def _range_tail_weight(self, target_range):
        denom = max(self.tail_end - self.tail_start, 1e-6)
        ramp = (target_range.abs() - self.tail_start) / denom
        ramp = torch.clamp(ramp, min=0.0, max=1.0)
        return 1.0 + (self.tail_max_weight - 1.0) * ramp

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):
        pred_heading_deg = self._pred_heading_deg(pose2)
        pred_range = pose2["range_value"].view(-1)
        target_heading_deg = self._target_heading_deg(gt2, pred_heading_deg.device, pred_heading_deg.dtype)
        target_range = self._target_range(gt2, pred_range.device, pred_range.dtype)

        angle_abs_deg = self._wrapped_abs_angle_deg(pred_heading_deg, target_heading_deg)
        target_angle_norm = torch.remainder(target_heading_deg, 360.0).abs()
        angle_den = torch.clamp(target_angle_norm, min=self.angle_floor_deg)
        angle_rel = angle_abs_deg / angle_den

        distance_abs = (pred_range - target_range).abs()
        distance_den = torch.clamp(target_range.abs(), min=self.distance_floor)
        distance_rel = distance_abs / distance_den
        tail_weight = self._range_tail_weight(target_range).to(device=distance_rel.device, dtype=distance_rel.dtype)
        weighted_distance_rel = distance_rel * tail_weight

        official_like = (
            self.heading_weight * angle_rel.mean()
            + self.range_weight * weighted_distance_rel.mean()
        )
        stabilizer = (
            self.absolute_heading_weight * F.smooth_l1_loss(angle_abs_deg, torch.zeros_like(angle_abs_deg), beta=1.0)
            + self.absolute_range_weight
            * (tail_weight * F.smooth_l1_loss(distance_abs, torch.zeros_like(distance_abs), beta=1.0, reduction="none")).mean()
        )
        total = official_like + stabilizer

        tail_mask = target_range.abs() >= self.tail_start
        if bool(tail_mask.any()):
            tail_distance_abs = float(distance_abs.detach()[tail_mask].mean())
            tail_distance_rel = float(distance_rel.detach()[tail_mask].mean())
        else:
            tail_distance_abs = 0.0
            tail_distance_rel = 0.0

        details = {
            "pairuav_tail_angle_rel": float(angle_rel.detach().mean()),
            "pairuav_tail_distance_rel": float(distance_rel.detach().mean()),
            "pairuav_tail_weighted_distance_rel": float(weighted_distance_rel.detach().mean()),
            "pairuav_tail_angle_abs_deg": float(angle_abs_deg.detach().mean()),
            "pairuav_tail_distance_abs": float(distance_abs.detach().mean()),
            "pairuav_tail_high_abs_range_distance_abs": tail_distance_abs,
            "pairuav_tail_high_abs_range_distance_rel": tail_distance_rel,
            "pairuav_tail_weight_mean": float(tail_weight.detach().mean()),
            "pairuav_tail_weight_max": float(tail_weight.detach().max()),
            "pairuav_tail_official_like": float(official_like.detach()),
        }
        return total, details
