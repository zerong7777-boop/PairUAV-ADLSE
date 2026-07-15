from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryPolicyEntry:
    pair_id: str
    angle_policy_weight: float
    teacher_aux_allowed: bool
    teacher_heading_split_fusion: float | None


class GeometryPolicyTable:
    def __init__(self, entries: Iterable[GeometryPolicyEntry]):
        self._entries = {entry.pair_id: entry for entry in entries}

    @classmethod
    def from_csv(cls, path):
        entries = []
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "pair_id" not in reader.fieldnames:
                raise ValueError(f"Geometry policy CSV must include pair_id: {path}")
            for row_index, row in enumerate(reader, start=2):
                pair_id = str(row.get("pair_id", "")).strip()
                if not pair_id:
                    raise ValueError(f"Geometry policy CSV row {row_index} has empty pair_id")
                entries.append(
                    GeometryPolicyEntry(
                        pair_id=pair_id,
                        angle_policy_weight=_float_from_row(
                            row,
                            ("angle_policy_weight", "angle_weight", "sample_weight"),
                            default=1.0,
                        ),
                        teacher_aux_allowed=_bool_from_row(
                            row,
                            ("teacher_aux_allowed", "teacher_allowed", "split_fusion_teacher_allowed"),
                            default=False,
                        ),
                        teacher_heading_split_fusion=_optional_float_from_row(
                            row,
                            (
                                "teacher_heading_split_fusion",
                                "split_fusion_teacher_heading",
                                "teacher_heading_deg",
                                "split_fusion_heading_deg",
                            ),
                        ),
                    )
                )
        return cls(entries)

    def get(self, pair_id):
        return self._entries[str(pair_id)]

    def __contains__(self, pair_id):
        return str(pair_id) in self._entries

    def __len__(self):
        return len(self._entries)


def _float_from_row(row, keys, default):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return float(default)


def _optional_float_from_row(row, keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def _bool_from_row(row, keys, default=False):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "allowed"}
    return bool(default)


def circular_heading_difference_deg(pred_deg, target_deg):
    return torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0


def circular_heading_loss_deg(pred_deg, target_deg, reduction="mean"):
    diff = circular_heading_difference_deg(pred_deg, target_deg)
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=1.0, reduction="none")
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean() if loss.numel() else loss.new_zeros(())
    raise ValueError(f"Unsupported reduction: {reduction}")


class PairUAVGeometryPolicyLoss(nn.Module):
    def __init__(self, policy_table, teacher_aux_weight=0.1, missing_policy_weight=1.0):
        super().__init__()
        self.policy_table = policy_table
        self.teacher_aux_weight = float(teacher_aux_weight)
        self.missing_policy_weight = float(missing_policy_weight)

    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, gt1=None, gt2=None, pose1=None, pose2=None, **kwargs):
        batch = kwargs.get("batch") or kwargs.get("view2") or gt2
        prediction = kwargs.get("prediction") or kwargs.get("predictions") or kwargs.get("pose2") or pose2
        if batch is None:
            raise ValueError("PairUAVGeometryPolicyLoss requires gt2/view2 batch with heading targets")
        if prediction is None:
            raise ValueError("PairUAVGeometryPolicyLoss requires pose2/prediction with heading predictions")

        pred_heading = _prediction_heading_deg(prediction)
        target_heading = _target_heading_deg(batch, pred_heading.device, pred_heading.dtype)
        pair_ids = _pair_ids_from_batch(batch, pred_heading.shape[0])
        entries = self._entries_for_pair_ids(pair_ids, device=pred_heading.device, dtype=pred_heading.dtype)

        per_sample = circular_heading_loss_deg(pred_heading, target_heading, reduction="none")
        primary = per_sample * entries["angle_weight"]
        total = primary.mean() if primary.numel() else pred_heading.new_zeros(())

        teacher_mask = entries["teacher_allowed"] & torch.isfinite(entries["teacher_heading"])
        teacher_count = int(teacher_mask.sum().item())
        teacher_loss = pred_heading.new_zeros(())
        if self.teacher_aux_weight > 0.0 and teacher_count:
            aux = circular_heading_loss_deg(pred_heading[teacher_mask], entries["teacher_heading"][teacher_mask], reduction="mean")
            teacher_loss = aux * self.teacher_aux_weight
            total = total + teacher_loss

        angle_abs = circular_heading_difference_deg(pred_heading, target_heading).abs()
        details = {
            "pairuav_geometry_policy_loss": float(total.detach()),
            "pairuav_geometry_policy_primary_loss": float((primary.mean() if primary.numel() else total).detach()),
            "pairuav_geometry_policy_teacher_aux_loss": float(teacher_loss.detach()),
            "pairuav_geometry_policy_teacher_aux_count": teacher_count,
            "pairuav_geometry_policy_angle_abs_deg": float(angle_abs.mean().detach()) if angle_abs.numel() else 0.0,
            "pairuav_geometry_policy_angle_weight_mean": float(entries["angle_weight"].mean().detach())
            if entries["angle_weight"].numel()
            else 0.0,
        }
        return total, details

    def _entries_for_pair_ids(self, pair_ids, device, dtype):
        weights = []
        teacher_allowed = []
        teacher_headings = []
        missing = []
        for pair_id in pair_ids:
            try:
                entry = self.policy_table.get(pair_id)
            except KeyError:
                missing.append(pair_id)
                weights.append(self.missing_policy_weight)
                teacher_allowed.append(False)
                teacher_headings.append(float("nan"))
                continue
            weights.append(float(entry.angle_policy_weight))
            teacher_allowed.append(bool(entry.teacher_aux_allowed))
            teacher_headings.append(
                float(entry.teacher_heading_split_fusion)
                if entry.teacher_heading_split_fusion is not None
                else float("nan")
            )
        if missing and self.training:
            example = missing[0]
            raise KeyError(
                f"Missing geometry policy entry for pair_id={example!r}; "
                "expected policy CSV pair_id to match PairUAV sample_id '<group_id>/<json_id>'."
            )
        return {
            "angle_weight": torch.as_tensor(weights, device=device, dtype=dtype),
            "teacher_allowed": torch.as_tensor(teacher_allowed, device=device, dtype=torch.bool),
            "teacher_heading": torch.as_tensor(teacher_headings, device=device, dtype=dtype),
        }


def _prediction_heading_deg(prediction):
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            raise ValueError("Empty prediction sequence")
        prediction = prediction[-1]
    if not isinstance(prediction, dict):
        raise TypeError(f"Prediction must be a dict, got {type(prediction).__name__}")

    for key in ("pred_heading_deg", "heading_deg"):
        if key in prediction:
            return torch.as_tensor(prediction[key], dtype=_tensor_dtype(prediction[key]), device=_tensor_device(prediction[key])).view(-1)
    for key in ("heading", "pred_heading"):
        if key in prediction:
            value = torch.as_tensor(prediction[key], dtype=_tensor_dtype(prediction[key]), device=_tensor_device(prediction[key]))
            if value.ndim > 0 and value.shape[-1] == 2:
                value = F.normalize(value, dim=-1, eps=1e-6)
                return torch.rad2deg(torch.atan2(value[..., 1], value[..., 0])).view(-1)
            return value.view(-1)
    for key in ("heading_vec", "pred_heading_vec", "pred_heading_vector"):
        if key in prediction:
            heading_vec = torch.as_tensor(prediction[key], dtype=_tensor_dtype(prediction[key]), device=_tensor_device(prediction[key]))
            heading_vec = F.normalize(heading_vec, dim=-1, eps=1e-6)
            return torch.rad2deg(torch.atan2(heading_vec[..., 1], heading_vec[..., 0])).view(-1)

    nested_keys = ("pose2", "prediction", "pred", "output")
    for key in nested_keys:
        if key in prediction and isinstance(prediction[key], dict):
            return _prediction_heading_deg(prediction[key])
    raise KeyError(
        "Could not find heading prediction. Tried heading, pred_heading, pred_heading_deg, "
        "heading_deg, heading_vec, pred_heading_vec."
    )


def _target_heading_deg(batch, device, dtype):
    if "heading_deg" in batch:
        return torch.as_tensor(batch["heading_deg"], device=device, dtype=dtype).view(-1)
    if "heading_cos" in batch and "heading_sin" in batch:
        cos = torch.as_tensor(batch["heading_cos"], device=device, dtype=dtype)
        sin = torch.as_tensor(batch["heading_sin"], device=device, dtype=dtype)
        return torch.rad2deg(torch.atan2(sin, cos)).view(-1)
    raise KeyError("Batch is missing ground-truth heading target: expected heading_deg or heading_cos/heading_sin")


def _pair_ids_from_batch(batch, batch_size):
    for key in ("pair_id", "sample_id", "canonical_pair_id"):
        if key in batch:
            values = _value_list(batch[key])
            if values and all(str(v) for v in values):
                return [str(v) for v in values]

    if "group_id" in batch and "json_id" in batch:
        groups = _value_list(batch["group_id"])
        json_ids = _value_list(batch["json_id"])
        if len(groups) == 1 and batch_size > 1:
            groups = groups * batch_size
        if len(json_ids) == 1 and batch_size > 1:
            json_ids = json_ids * batch_size
        pair_ids = [f"{groups[i]}/{json_ids[i]}" for i in range(min(len(groups), len(json_ids), batch_size))]
        if len(pair_ids) != batch_size:
            raise KeyError(f"Could not derive {batch_size} pair ids from group_id/json_id batch fields")
        return pair_ids

    if "group_id" in batch and "json_path" in batch:
        groups = _value_list(batch["group_id"])
        paths = _value_list(batch["json_path"])
        pair_ids = [f"{groups[i]}/{Path(str(paths[i])).stem}" for i in range(min(len(groups), len(paths), batch_size))]
        if len(pair_ids) != batch_size:
            raise KeyError(f"Could not derive {batch_size} pair ids from group_id/json_path batch fields")
        return pair_ids

    raise KeyError(
        "Batch is missing pair identity. Expected pair_id/sample_id or group_id plus json_id/json_path."
    )


def _value_list(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return [str(value.item())]
        return [str(v.item()) for v in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _tensor_device(value):
    return value.device if torch.is_tensor(value) else None


def _tensor_dtype(value):
    return value.dtype if torch.is_tensor(value) and value.is_floating_point() else torch.float32
