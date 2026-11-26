"""Training losses for E2E driving and World Models."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.core.registry import Registry

LOG_2 = 0.6931471805599453


@Registry.register_loss("laplace_nll")
class LaplaceNLL(nn.Module):
    """Heteroscedastic Laplace NLL: |y - mu| / b + log(2b), b = exp(log_b)."""

    def __init__(self, min_log_b: float = -5.0, max_log_b: float = 5.0) -> None:
        super().__init__()
        self.min_log_b = min_log_b
        self.max_log_b = max_log_b

    def forward(
        self,
        mu: torch.Tensor,
        log_b: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        clamped = torch.clamp(log_b, self.min_log_b, self.max_log_b)
        nll = torch.abs(target - mu) / torch.exp(clamped) + clamped + LOG_2
        if mask is None:
            return nll.mean()
        weights = mask.to(dtype=nll.dtype)
        return (nll * weights).sum() / weights.sum().clamp_min(1.0)


@Registry.register_loss("mhp_laplace")
class MHPLaplaceLoss(nn.Module):
    """Winner-takes-all Multi-Hypothesis Planning Laplace NLL + mode CE."""

    def __init__(
        self,
        cls_weight: float = 1.0,
        min_log_b: float = -5.0,
        max_log_b: float = 5.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.laplace = LaplaceNLL(min_log_b=min_log_b, max_log_b=max_log_b)

    def forward(
        self,
        mu: torch.Tensor,
        log_b: torch.Tensor,
        mode_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = mu.shape[0]
        with torch.no_grad():
            errors = torch.abs(mu - target.unsqueeze(1)).mean(dim=(-2, -1))
            best_idx = torch.argmin(errors, dim=1)

        batch_idx = torch.arange(batch, device=mu.device)
        reg_loss = self.laplace(mu[batch_idx, best_idx], log_b[batch_idx, best_idx], target)
        cls_loss = F.cross_entropy(mode_logits, best_idx)
        return {
            "loss": reg_loss + self.cls_weight * cls_loss,
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
        }


@Registry.register_loss("flow_matching")
class FlowMatchingLoss(nn.Module):
    """Rectified Flow velocity-matching loss (target velocity = x1 - x0)."""

    def forward(
        self,
        pred_velocity: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        err = F.mse_loss(pred_velocity, x1 - x0, reduction="none")
        if mask is None:
            return err.mean()
        weights = mask.to(dtype=err.dtype)
        return (err * weights).sum() / weights.sum().clamp_min(1.0)


@Registry.register_loss("waypoint")
class WaypointLoss(nn.Module):
    """Smooth-L1 waypoint loss with 2nd-order smoothness and control terms."""

    def __init__(
        self,
        smooth_weight: float = 0.25,
        ctrl_weight: float = 0.5,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ctrl_weight = ctrl_weight
        self.beta = beta

    def forward(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
        pred_ctrl: Optional[torch.Tensor] = None,
        gt_ctrl: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        wp_loss = F.smooth_l1_loss(pred_xy, gt_xy, beta=self.beta)

        if pred_xy.shape[1] >= 3:
            pred_curv = pred_xy[:, 2:] - 2.0 * pred_xy[:, 1:-1] + pred_xy[:, :-2]
            gt_curv = gt_xy[:, 2:] - 2.0 * gt_xy[:, 1:-1] + gt_xy[:, :-2]
            smooth_loss = F.l1_loss(pred_curv, gt_curv)
        else:
            smooth_loss = pred_xy.new_tensor(0.0)

        if pred_ctrl is not None and gt_ctrl is not None:
            ctrl_loss = F.smooth_l1_loss(pred_ctrl, gt_ctrl, beta=self.beta)
        else:
            ctrl_loss = pred_xy.new_tensor(0.0)

        total = wp_loss + self.smooth_weight * smooth_loss + self.ctrl_weight * ctrl_loss
        return {
            "loss": total,
            "wp_loss": wp_loss,
            "smooth_loss": smooth_loss,
            "ctrl_loss": ctrl_loss,
        }
