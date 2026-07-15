# references: DUSt3R: https://github.com/naver/dust3r


from copy import copy, deepcopy
import math
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'+bb()
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)

        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class RelativeCameraPoseRegression(Criterion, MultiLoss): 
    def __init__(self, criterion):
        super().__init__(criterion)
        self.PoseLoss = Reloc3rPoseLoss() 

    def get_poses(self, gt1, gt2, pose1, pose2):
        gt_pose2to1 = torch.inverse(gt1['camera_pose']) @ gt2['camera_pose']
        gt_pose1to2 = torch.inverse(gt2['camera_pose']) @ gt1['camera_pose']
        pr_pose2to1 = pose2['pose'] 
        pr_pose1to2 = pose1['pose']
        return gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, {}

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):

        gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, monitoring = self.get_poses(gt1, gt2, pose1, pose2)

        # compute loss
        loss_pose2, loss_Terr2, loss_Rerr2 = self.PoseLoss(pr_pose2to1, gt_pose2to1)
        loss_pose1, loss_Terr1, loss_Rerr1 = self.PoseLoss(pr_pose1to2, gt_pose1to2)

        # record and return details
        self_name = type(self).__name__
        details = {
                   self_name+'_terr1': float(loss_Terr1*180 / math.pi),
                   self_name+'_rerr1': float(loss_Rerr1*180 / math.pi),
                   self_name+'_terr2': float(loss_Terr2*180 / math.pi),
                   self_name+'_rerr2': float(loss_Rerr2*180 / math.pi),
                  }
        return loss_pose1 + loss_pose2, dict(pose_loss = float(loss_pose1 + loss_pose2), **(details | monitoring))


class Reloc3rPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_pred, pose_gt):
        t = pose_pred[:,0:3,-1]
        tgt = pose_gt[:,0:3,-1]
        R = pose_pred[:, :3, :3]
        Rgt = pose_gt[:, :3, :3]

        trans_loss = self.transl_ang_loss(t, tgt)
        rot_loss = self.rot_ang_loss(R, Rgt)
        loss =  trans_loss + rot_loss
        return loss, trans_loss, rot_loss


class PairUAVHeadingRangeLoss(MultiLoss):
    def __init__(self, heading_weight=1.0, range_weight=1.0, beta=0.1, supervise_inverse=False):
        super().__init__()
        self.heading_weight = float(heading_weight)
        self.range_weight = float(range_weight)
        self.beta = float(beta)
        self.supervise_inverse = bool(supervise_inverse)

    def get_name(self):
        return type(self).__name__

    @staticmethod
    def _heading_target(view):
        heading_cos = torch.as_tensor(view['heading_cos'], dtype=torch.float32)
        heading_sin = torch.as_tensor(view['heading_sin'], dtype=torch.float32)
        return torch.stack((heading_cos, heading_sin), dim=-1)

    def _single_direction_loss(self, prediction, target_heading_vec, target_range_value):
        pred_heading = prediction['heading_vec']
        pred_range = prediction['range_value'].view(-1, 1)
        heading_loss = F.smooth_l1_loss(pred_heading, target_heading_vec, beta=self.beta)
        range_loss = F.smooth_l1_loss(pred_range, target_range_value.view(-1, 1), beta=self.beta)
        total = self.heading_weight * heading_loss + self.range_weight * range_loss
        return total, heading_loss, range_loss

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):
        target_heading_vec = self._heading_target(gt2).to(device=pose2['heading_vec'].device, dtype=pose2['heading_vec'].dtype)
        target_range_value = torch.as_tensor(
            gt2['range_value'],
            dtype=pose2['range_value'].dtype,
            device=pose2['range_value'].device,
        )

        loss_main, heading_loss, range_loss = self._single_direction_loss(pose2, target_heading_vec, target_range_value)
        total = loss_main
        details = {
            'pairuav_heading_loss': float(heading_loss),
            'pairuav_range_loss': float(range_loss),
            'pairuav_loss_main': float(loss_main),
        }

        if self.supervise_inverse:
            inv_heading_vec = -target_heading_vec
            inv_range_value = -target_range_value
            loss_inv, heading_loss_inv, range_loss_inv = self._single_direction_loss(pose1, inv_heading_vec, inv_range_value)
            total = total + loss_inv
            details.update({
                'pairuav_heading_loss_inv': float(heading_loss_inv),
                'pairuav_range_loss_inv': float(range_loss_inv),
                'pairuav_loss_inv': float(loss_inv),
            })

        return total, details


class PairUAVOfficialMetricAwareLoss(MultiLoss):
    def __init__(
        self,
        heading_weight=1.0,
        range_weight=1.0,
        angle_floor_deg=1.0,
        distance_floor=1.0,
        absolute_heading_weight=0.05,
        absolute_range_weight=0.05,
    ):
        super().__init__()
        self.heading_weight = float(heading_weight)
        self.range_weight = float(range_weight)
        self.angle_floor_deg = float(angle_floor_deg)
        self.distance_floor = float(distance_floor)
        self.absolute_heading_weight = float(absolute_heading_weight)
        self.absolute_range_weight = float(absolute_range_weight)

    def get_name(self):
        return type(self).__name__

    @staticmethod
    def _target_heading_deg(view, device, dtype):
        return torch.as_tensor(view["heading_deg"], device=device, dtype=dtype).view(-1)

    @staticmethod
    def _target_range(view, device, dtype):
        return torch.as_tensor(view["range_value"], device=device, dtype=dtype).view(-1)

    @staticmethod
    def _pred_heading_deg(prediction):
        heading_vec = F.normalize(prediction["heading_vec"], dim=-1)
        return torch.rad2deg(torch.atan2(heading_vec[:, 1], heading_vec[:, 0]))

    @staticmethod
    def _wrapped_abs_angle_deg(pred_deg, target_deg):
        diff = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
        return diff.abs()

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

        official_like = self.heading_weight * angle_rel.mean() + self.range_weight * distance_rel.mean()
        stabilizer = (
            self.absolute_heading_weight * F.smooth_l1_loss(angle_abs_deg, torch.zeros_like(angle_abs_deg), beta=1.0)
            + self.absolute_range_weight * F.smooth_l1_loss(distance_abs, torch.zeros_like(distance_abs), beta=1.0)
        )
        total = official_like + stabilizer
        details = {
            "pairuav_official_angle_rel": float(angle_rel.mean()),
            "pairuav_official_distance_rel": float(distance_rel.mean()),
            "pairuav_official_angle_abs_deg": float(angle_abs_deg.mean()),
            "pairuav_official_distance_abs": float(distance_abs.mean()),
            "pairuav_official_like": float(official_like),
        }
        return total, details

    def transl_ang_loss(self, t, tgt, eps=1e-6):
        """
        Args: 
            t: estimated translation vector [B, 3]
            tgt: ground-truth translation vector [B, 3]
        Returns: 
            T_err: translation direction angular error 
        """
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t_normed = t / (t_norm + eps)
        tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
        tgt_normed = tgt / (tgt_norm + eps)
        cosine = torch.sum(t_normed * tgt_normed, dim=1)
        T_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return T_err.mean()

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()




class PairUAVPhase101SATRTeacherLoss(PairUAVOfficialMetricAwareLoss):
    """Official-like PairUAV loss plus Phase101 SATR train-only teacher losses."""

    def __init__(
        self,
        *args,
        teacher_csv,
        router_weight=0.02,
        identity_weight=0.05,
        gate_weight=0.001,
        final_mode_id=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_csv = str(teacher_csv)
        self.router_weight = float(router_weight)
        self.identity_weight = float(identity_weight)
        self.gate_weight = float(gate_weight)
        self.final_mode_id = int(final_mode_id)
        self.teacher_by_path = self._load_teacher_csv(Path(self.teacher_csv))

    def get_name(self):
        return type(self).__name__

    @staticmethod
    def _case_to_mode(case_id):
        case_id = str(case_id)
        if case_id == "final":
            return 0
        if case_id == "step450000":
            return 1
        if case_id == "step400000":
            return 2
        if case_id == "step350000":
            return 3
        return 4

    @classmethod
    def _load_teacher_csv(cls, path):
        if not path.is_file():
            raise FileNotFoundError(f"Phase101 teacher CSV not found: {path}")
        table = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                json_path = str(row.get("json_path", ""))
                if not json_path:
                    continue
                heading_case = row.get("teacher_heading_case", row.get("best_heading_case", "final"))
                range_case = row.get("teacher_range_case", row.get("best_range_case", "final"))
                table[json_path] = {
                    "heading_mode": cls._case_to_mode(heading_case),
                    "range_mode": cls._case_to_mode(range_case),
                    "final_best": 1 if row.get("best_final_case", "") == "final" else 0,
                    "heading_case": str(heading_case),
                    "range_case": str(range_case),
                }
        if not table:
            raise ValueError(f"No teacher rows loaded from {path}")
        return table

    @staticmethod
    def _json_paths(view):
        paths = view.get("json_path", [])
        if isinstance(paths, (str, Path)):
            return [str(paths)]
        return [str(item) for item in paths]

    def _teacher_tensors(self, view, device):
        paths = self._json_paths(view)
        heading_modes = []
        range_modes = []
        final_best = []
        missing = 0
        for path in paths:
            row = self.teacher_by_path.get(path)
            if row is None:
                missing += 1
                heading_modes.append(self.final_mode_id)
                range_modes.append(self.final_mode_id)
                final_best.append(1)
            else:
                heading_modes.append(int(row["heading_mode"]))
                range_modes.append(int(row["range_mode"]))
                final_best.append(int(row["final_best"]))
        return (
            torch.as_tensor(heading_modes, device=device, dtype=torch.long),
            torch.as_tensor(range_modes, device=device, dtype=torch.long),
            torch.as_tensor(final_best, device=device, dtype=torch.float32),
            missing,
        )

    @staticmethod
    def _router_nll(router_probs, target):
        log_probs = torch.log(router_probs.clamp_min(1e-8))
        return F.nll_loss(log_probs, target)

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):
        base_loss, details = super().compute_loss(gt1, gt2, pose1, pose2, **kw)
        if "phase101_heading_router_train" not in pose2 or "phase101_range_router_train" not in pose2:
            raise KeyError("Phase101 teacher loss requires phase101 router train tensors in pose2")
        device = pose2["heading_vec"].device
        heading_target, range_target, final_best, missing = self._teacher_tensors(gt2, device)
        heading_router = pose2["phase101_heading_router_train"]
        range_router = pose2["phase101_range_router_train"]
        heading_router_loss = self._router_nll(heading_router, heading_target)
        range_router_loss = self._router_nll(range_router, range_target)
        router_loss = 0.5 * (heading_router_loss + range_router_loss)

        base_heading = pose2["phase101_base_heading_vec"].to(device=device, dtype=pose2["heading_vec"].dtype)
        base_range = pose2["phase101_base_range_value"].to(device=device, dtype=pose2["range_value"].dtype).view(-1)
        pred_range = pose2["range_value"].view(-1)
        heading_identity = ((pose2["heading_vec"] - base_heading).square().sum(dim=-1) * final_best).sum() / final_best.sum().clamp_min(1.0)
        range_identity = (((pred_range - base_range).square()) * final_best).sum() / final_best.sum().clamp_min(1.0)
        identity_loss = heading_identity + range_identity

        gate_loss = torch.zeros((), device=device, dtype=pose2["heading_vec"].dtype)
        if "phase101_heading_gate_train" in pose2:
            gate_loss = gate_loss + pose2["phase101_heading_gate_train"].mean()
        if "phase101_range_gate_train" in pose2:
            gate_loss = gate_loss + pose2["phase101_range_gate_train"].mean()
        gate_loss = 0.5 * gate_loss

        total = base_loss + self.router_weight * router_loss + self.identity_weight * identity_loss + self.gate_weight * gate_loss
        details = dict(details)
        details.update({
            "phase101_router_loss": float(router_loss.detach()),
            "phase101_heading_router_loss": float(heading_router_loss.detach()),
            "phase101_range_router_loss": float(range_router_loss.detach()),
            "phase101_identity_loss": float(identity_loss.detach()),
            "phase101_gate_loss": float(gate_loss.detach()),
            "phase101_teacher_missing": float(missing),
            "phase101_final_best_frac": float(final_best.mean().detach()),
            "phase101_router_weight": self.router_weight,
            "phase101_identity_weight": self.identity_weight,
            "phase101_gate_weight": self.gate_weight,
        })
        return total, details


class PairUAVOfficialMetricAwarePolarLoss(PairUAVOfficialMetricAwareLoss):
    """Official-like PairUAV loss plus polar vector recomposition consistency."""

    def __init__(self, *args, polar_vector_weight=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.polar_vector_weight = float(polar_vector_weight)

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):
        base_loss, details = super().compute_loss(gt1, gt2, pose1, pose2, **kw)
        pred_heading_deg = self._pred_heading_deg(pose2)
        pred_range = pose2["range_value"].view(-1)
        target_heading_deg = self._target_heading_deg(gt2, pred_heading_deg.device, pred_heading_deg.dtype)
        target_range = self._target_range(gt2, pred_range.device, pred_range.dtype)

        pred_rad = torch.deg2rad(pred_heading_deg)
        target_rad = torch.deg2rad(target_heading_deg)
        pred_xy = torch.stack((pred_range * torch.cos(pred_rad), pred_range * torch.sin(pred_rad)), dim=-1)
        target_xy = torch.stack((target_range * torch.cos(target_rad), target_range * torch.sin(target_rad)), dim=-1)
        scale = torch.clamp(target_range.abs(), min=self.distance_floor).unsqueeze(-1)
        vector_loss = F.smooth_l1_loss(pred_xy / scale, target_xy / scale, beta=0.05)
        total = base_loss + self.polar_vector_weight * vector_loss
        details = dict(details)
        details["pairuav_polar_vector_loss"] = float(vector_loss)
        details["pairuav_polar_vector_weight"] = self.polar_vector_weight
        return total, details

def loss_of_one_batch(batch, model, criterion, device, use_amp=False, ret=None):
    view1, view2 = batch
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split(): 
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):

        pose1, pose2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pose1, pose2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pose1=pose1, pose2=pose2, loss=loss)
    return result[ret] if ret else result

