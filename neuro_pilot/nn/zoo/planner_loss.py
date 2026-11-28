"""Loss and backbone feature helpers for single-camera multi-task Zoo planners."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo import MHPLaplaceLoss, download_weights, resolve_weight_key
from neuro_pilot.utils.losses import DetectionLoss, FDATLoss, HeatmapLoss
from neuro_pilot.utils.planner_metrics import sym_curve_dist


def forward_with_p4p5(
    model: nn.Module, img: torch.Tensor, command: torch.Tensor
) -> Tuple[Dict[str, Any], Any, torch.Tensor, torch.Tensor]:
    """Run GraphModel returning `(det_out, traj_out, P4, P5)` for joint training."""
    inputs = {"img": img, "command": command}
    cache: Dict[int, Any] = {}
    prev: Any = inputs
    for idx, (layer, src) in enumerate(zip(model.layers, model._sources)):
        inp = model._resolve_ref(src, prev, cache, inputs)
        out = (
            layer(*inp)
            if isinstance(inp, list) and getattr(layer, "unpack_inputs", False)
            else layer(inp)
        )
        if idx in model._save_indices or idx in (4, 6, 23, 24):
            cache[idx] = out
        prev = out
    return cache[23], cache[24], cache[4], cache[6]


def forward_backbone_p4p5(
    model: nn.Module, img: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run only backbone layers `0..6` to extract `(P4, P5)` for unlabeled distillation."""
    x = p4 = p5 = img
    for idx in range(7):
        x = model.layers[idx](x)
        if idx == 4:
            p4 = x
        elif idx == 6:
            p5 = x
    return p4, p5


def load_matching_weights(
    model: nn.Module, ckpt_path: str | Path, prefixes: Optional[Tuple[str, ...]] = None
) -> int:
    """Load shape-matched tensors from a local checkpoint or auto-downloaded hub key."""
    p = Path(ckpt_path)
    if not p.is_file():
        key = resolve_weight_key(str(ckpt_path))
        if key is None:
            raise FileNotFoundError(f"Checkpoint not found locally or on hub: {ckpt_path}")
        p = download_weights(key)
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    src = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    dst = model.state_dict()
    matched = 0
    for k, v in src.items():
        if prefixes and not k.startswith(prefixes):
            continue
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            matched += 1
    model.load_state_dict(dst)
    return matched


def _smoothness_loss(
    wp_f: torch.Tensor, gt: torch.Tensor, mu: torch.Tensor, t_len: int
) -> torch.Tensor:
    """Density-invariant velocity, acceleration, tangent cosine, and jerk penalty."""
    scale_t = max(1.0, (t_len - 1) / 9.0)
    v_p = (wp_f[:, 1:] - wp_f[:, :-1]) * scale_t
    v_g = (gt[:, 1:] - gt[:, :-1]) * scale_t
    a_p = (v_p[:, 1:] - v_p[:, :-1]) * scale_t
    a_g = (v_g[:, 1:] - v_g[:, :-1]) * scale_t
    j_p = (a_p[:, 1:] - a_p[:, :-1]) * scale_t
    cos_sim = (F.normalize(v_p, dim=-1, eps=1e-5) * F.normalize(v_g, dim=-1, eps=1e-5)).sum(-1)
    seg_w = torch.ones_like(cos_sim)
    seg_w[:, -max(2, t_len // 5) :] = 2.5
    v_all = (mu[:, :, 1:] - mu[:, :, :-1]) * scale_t
    a_all = (v_all[:, :, 1:] - v_all[:, :, :-1]) * scale_t
    j_all = (a_all[:, :, 1:] - a_all[:, :, :-1]) * scale_t
    return (
        20.0 * F.smooth_l1_loss(v_p, v_g, beta=0.008)
        + 32.0 * F.smooth_l1_loss(a_p, a_g, beta=0.008)
        + 4.0 * ((1.0 - cos_sim) * seg_w).mean()
        + 25.0 * F.smooth_l1_loss(j_p, torch.zeros_like(j_p), beta=0.005)
        + 10.0 * F.smooth_l1_loss(j_all, torch.zeros_like(j_all), beta=0.006)
    )


def mhp_traj_loss(
    traj_out: Dict[str, torch.Tensor],
    gt_wp: torch.Tensor,
    cmd: torch.Tensor,
    epoch: int,
    mhp_fn: MHPLaplaceLoss,
    fdat_fn: FDATLoss,
    hm_fn: HeatmapLoss,
    smooth_weight: float,
) -> torch.Tensor:
    """Compute joint MHP + differentiable MoE + entry-lane + physics smoothness loss."""
    wp_f, gt = traj_out["waypoints"].float(), gt_wp.float()
    mu, mu0 = traj_out["mu"].float(), traj_out["mu0"].float()
    log_b, m_logits = traj_out["log_b"].float(), traj_out["mode_logits"].float()
    t_len = gt.shape[1]
    w_h = torch.linspace(0.85, 1.85, t_len, device=gt.device, dtype=gt.dtype).view(1, t_len, 1)
    w_h = w_h / w_h.mean()

    mhp = mhp_fn(mu, log_b, m_logits, gt)
    with torch.no_grad():
        mode_mse_w = (((mu - gt.unsqueeze(1)) ** 2) * w_h.unsqueeze(1)).mean(dim=(-2, -1))
        best_k = mode_mse_w.argmin(dim=1)
        soft_tgt = F.softmax(-mode_mse_w / 0.003, dim=-1)
    cmd_w = torch.where(cmd != 0, 2.0, 1.0)
    kl_per = F.kl_div(F.log_softmax(m_logits, dim=-1), soft_tgt, reduction="none").sum(dim=-1)
    l_mode_kl = (kl_per * cmd_w).mean()
    b_idx = torch.arange(mu.shape[0], device=mu.device)
    mu_best, mu0_best = mu[b_idx, best_k], mu0[b_idx, best_k]
    if epoch > 6:
        with torch.no_grad():
            hard_w = 1.0 + (torch.abs(wp_f - gt).mean(dim=(1, 2)) / 0.05).clamp(0.0, 2.0)
        cmd_w = cmd_w * hard_w

    tail_k = max(2, t_len // 4)
    sl1_all = F.smooth_l1_loss(wp_f, gt, beta=0.005, reduction="none")
    sl1_tail = F.smooth_l1_loss(wp_f[:, -tail_k:], gt[:, -tail_k:], beta=0.005, reduction="none")
    sl1_best = F.smooth_l1_loss(mu_best, gt, beta=0.005, reduction="none")
    l_fdat = (fdat_fn(wp_f, gt, (cmd != 0).float()) * cmd_w).mean()
    l_l1 = ((sl1_all * w_h).mean((1, 2)) * cmd_w).mean()
    l_mse = ((((wp_f - gt) ** 2) * w_h).mean((1, 2)) * cmd_w).mean()
    l_ent_mse = ((((wp_f[:, -tail_k:] - gt[:, -tail_k:]) ** 2)).mean((1, 2)) * cmd_w).mean()
    l_ent_l1 = (sl1_tail.mean((1, 2)) * cmd_w).mean()
    l_crv = sym_curve_dist(wp_f, gt).mean()
    l_mono = F.relu(wp_f[:, 1:, 1] - wp_f[:, :-1, 1]).mean()
    l_wta = (
        8.0 * ((sl1_best * w_h).mean((1, 2)) * cmd_w).mean()
        + 40.0 * ((((mu_best - gt) ** 2) * w_h).mean((1, 2)) * cmd_w).mean()
        + 4.0 * F.smooth_l1_loss(mu0_best, gt, beta=0.02)
    )
    l_smooth = _smoothness_loss(wp_f, gt, mu, t_len)
    hm_logits = traj_out["heatmap"].float().clamp(-15.0, 15.0)
    gt_hm = hm_fn.generate_heatmap(gt * 2.0 - 1.0, hm_logits.shape[-2], hm_logits.shape[-1])
    l_hm = hm_fn(hm_logits, gt_hm).mean()
    return 0.70 * (
        0.45 * (mhp["loss"] + 3.5)
        + 2.0 * l_mode_kl
        + l_wta
        + 3.0 * l_fdat
        + 12.0 * l_l1
        + 160.0 * l_mse
        + 100.0 * l_ent_mse
        + 8.0 * l_ent_l1
        + 10.0 * l_crv
        + 12.0 * l_mono
        + 1.2 * l_hm
        + smooth_weight * l_smooth
    )


def build_planner_optim(
    model: nn.Module,
    proj: Optional[nn.Module],
    epochs: int,
    opt_fn: Callable[[str, Any], Any],
) -> Tuple[torch.optim.Optimizer, Any, list, list, list]:
    """Create 3-group AdamW (backbone, det/neck, traj) and warmup + cosine LR schedule."""
    bb_p = [p for i in range(1, 10) for p in model.layers[i].parameters()]
    traj_p = list(model.layers[24].parameters())
    special = {id(p) for p in bb_p} | {id(p) for p in traj_p}
    det_p = [p for p in model.parameters() if id(p) not in special]
    if proj is not None:
        det_p.extend(list(proj.parameters()))
    lr_head, lr_bb = float(opt_fn("lr_head", 1.2e-3)), float(opt_fn("lr_backbone", 2e-4))
    is_mhp = getattr(model, "model_key", "") in ("pilot_mhp", "bfmc_mhp")
    lr_traj = float(opt_fn("lr_traj", lr_head * 0.5 if is_mhp else lr_head))
    opt = torch.optim.AdamW([
        {"params": bb_p, "lr": lr_bb, "weight_decay": 1e-2},
        {"params": det_p, "lr": lr_head, "weight_decay": 2e-2},
        {"params": traj_p, "lr": lr_traj, "weight_decay": 1e-2},
    ])
    warm = max(0, int(opt_fn("warmup_epochs", 3)))

    def _lr_fn(ep: int) -> float:
        if warm > 0 and ep < warm:
            return float(ep + 1) / float(warm)
        prog = float(ep - warm) / max(1.0, float(epochs - warm))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn), bb_p, det_p, traj_p


def build_planner_losses(
    model: nn.Module, device: torch.device, det_cls_weight: float = 0.85
) -> Tuple[DetectionLoss, FDATLoss, MHPLaplaceLoss, HeatmapLoss]:
    """Instantiate joint detection, FDAT, MHP Laplace, and heatmap losses."""
    det_proxy = SimpleNamespace(
        heads={"detect": model.layers[23]}, parameters=model.layers[23].parameters
    )
    return (
        DetectionLoss(det_proxy, config=SimpleNamespace(box=7.5, cls_det=det_cls_weight, dfl=1.5)),
        FDATLoss(base_loss_type="wing").to(device),
        MHPLaplaceLoss(cls_weight=1.5, min_log_b=-3.5, max_log_b=1.0).to(device),
        HeatmapLoss(dice_weight=5.0, device=device).to(device),
    )
