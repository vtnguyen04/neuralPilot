"""Generic single-camera multi-task planner trainer (MHP/MoE, DINOv3 REPA, EMA, RunMonitor)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuro_pilot.data.datasets.planner import (
    PlannerDataset,
    UnlabeledFrameDataset,
    build_balanced_sampler,
    planner_collate_fn,
)
from neuro_pilot.engine.callbacks import CallbackList
from neuro_pilot.engine.monitor import RunMonitor
from neuro_pilot.engine.planner_eval import PlannerValidator
from neuro_pilot.nn.zoo import MHPLaplaceLoss
from neuro_pilot.nn.zoo.distill import DinoV3Teacher, DistillProj, dense_distill_loss
from neuro_pilot.nn.zoo.planner_loss import (
    build_planner_losses,
    build_planner_optim,
    forward_backbone_p4p5,
    forward_with_p4p5,
    load_matching_weights,
    mhp_traj_loss,
)
from neuro_pilot.tasks.base import save_graph_checkpoint
from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.losses import FDATLoss, HeatmapLoss
from neuro_pilot.utils.torch_utils import ModelEMA


class PlannerTrainer:
    """Framework trainer for single-camera multi-task planners (`pilot_mhp`, `pilot_det`)."""

    def __init__(
        self,
        model: nn.Module,
        overrides: Optional[Dict[str, Any]] = None,
        task_name: str = "planner",
    ) -> None:
        self.model, self.cfg, self.task_name = model, dict(overrides or {}), task_name
        self.device = next(model.parameters()).device
        self.save_dir = Path(self.cfg.get("save_dir", "runs/planner"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best, self.best_traj = self.save_dir / "best.pt", self.save_dir / "best_traj.pt"
        self.last = self.save_dir / "last.pt"
        self.monitor = RunMonitor(self.save_dir, task_name=task_name)
        self.callbacks = CallbackList(self.cfg.get("callbacks"))
        self.validator = PlannerValidator(model, overrides=self.cfg)
        self.epoch = 0

    def _opt(self, key: str, default: Any) -> Any:
        if key in self.cfg and self.cfg[key] is not None:
            return self.cfg[key]
        for sub in ("trainer", "data", "loss"):
            sec = self.cfg.get(sub)
            if isinstance(sec, dict) and key in sec and sec[key] is not None:
                return sec[key]
        return default

    def _build_loaders(self) -> Tuple[PlannerDataset, DataLoader, DataLoader, Optional[DataLoader]]:
        img_w, img_h = int(self._opt("img_w", 640)), int(self._opt("img_h", 512))
        bs = int(self._opt("batch_size", self._opt("batch", 8)))
        train_json, val_json = self._opt("train_json", ""), self._opt("val_json", "")
        if not train_json or not val_json:
            raise ValueError("PlannerTrainer requires `train_json` and `val_json`.")
        train_ds = PlannerDataset(
            train_json, img_w, img_h, augment=True,
            copy_paste_prob=float(self._opt("copy_paste", 0.50)),
            affine_prob=float(self._opt("affine_prob", 0.65)),
            affine_strength=float(self._opt("affine_strength", 1.0)),
        )
        use_bal = bool(self._opt("balance_commands", True))
        sampler = build_balanced_sampler(train_ds.records) if use_bal else None
        pin, nw = self.device.type == "cuda", int(self._opt("num_workers", 4))
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
            num_workers=nw, pin_memory=pin, drop_last=len(train_ds) >= bs,
            collate_fn=planner_collate_fn,
        )
        val_loader = DataLoader(
            PlannerDataset(val_json, img_w, img_h, augment=False), batch_size=bs, shuffle=False,
            num_workers=min(nw, 2), pin_memory=pin, collate_fn=planner_collate_fn,
        )
        unlab_json = self._opt("unlabeled_json", "")
        unlab_loader = (
            DataLoader(
                UnlabeledFrameDataset(unlab_json, img_w, img_h), batch_size=bs,
                shuffle=True, num_workers=2, pin_memory=pin, drop_last=True,
            )
            if bool(self._opt("distill", False)) and unlab_json else None
        )
        return train_ds, train_loader, val_loader, unlab_loader

    def _init_weights(self) -> bool:
        resume = self._opt("resume_weights", "")
        if resume:
            n = load_matching_weights(self.model, resume)
            logger.info(f"({self.task_name}) Warm-started ({n} tensors) from {resume}")
            return True
        if bool(self._opt("_warm_started", False)):
            return True
        backbone_pt = self._opt("pretrained_backbone", self._opt("pilot_s_pt", ""))
        if backbone_pt:
            prefixes = tuple(f"layers.{i}." for i in range(1, 10))
            n = load_matching_weights(self.model, backbone_pt, prefixes=prefixes)
            logger.info(f"({self.task_name}) Loaded {n} backbone tensors from {backbone_pt}")
        return False

    def _eval_and_save(
        self, epoch: int, ev_m: nn.Module, val_loader: DataLoader,
        img_w: int, img_h: int, state: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.validator.model = ev_m
        val_m = self.validator(dataloader=val_loader)
        if epoch >= 5 and (not math.isfinite(val_m["mse"]) or val_m["mse"] > 0.25):
            raise RuntimeError(f"Early-stop guardrail at Ep {epoch}: MSE={val_m['mse']}")
        extra = {"img_w": img_w, "img_h": img_h, "metrics": {"epoch": epoch, **val_m}}
        save_graph_checkpoint(ev_m, self.last, extra=extra)
        if val_m["mse"] < state["best_mse"]:
            state["best_mse"] = val_m["mse"]
            save_graph_checkpoint(ev_m, self.best_traj, extra=extra)
        if val_m["score"] > state["best_score"]:
            state["best_score"], state["best_metrics"] = val_m["score"], extra["metrics"]
            save_graph_checkpoint(ev_m, self.best, extra=extra)
        self.callbacks.on_val_end(self)
        return val_m

    def train(self) -> Dict[str, Any]:
        """Execute full training loop with EMA, RunMonitor, callbacks, and val grid export."""
        torch.manual_seed(int(self._opt("seed", 42)))
        epochs = int(self._opt("epochs", self._opt("max_epochs", 40)))
        freeze_ep = int(self._opt("freeze_epochs", 3))
        eval_every = max(1, int(self._opt("eval_every", 2)))
        polish_ep = int(self._opt("polish_epochs", 0))
        img_w, img_h = int(self._opt("img_w", 640)), int(self._opt("img_h", 512))
        train_ds, train_loader, val_loader, unlab_loader = self._build_loaders()
        resumed = self._init_weights()
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        teacher = (
            DinoV3Teacher(str(self._opt("dinov3_path", ""))).to(self.device, dtype=amp_dtype).eval()
            if bool(self._opt("distill", False)) else None
        )
        proj = DistillProj(128, 256, teacher.dim).to(self.device) if teacher is not None else None
        opt, sched, bb_p, det_p, traj_p = build_planner_optim(self.model, proj, epochs, self._opt)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda" and not use_bf16))
        ema = ModelEMA(self.model, decay=float(self._opt("ema_decay", 0.994)), tau=300)
        losses_fn = build_planner_losses(
            self.model, self.device, float(self._opt("det_cls_weight", 0.85))
        )
        weights = (
            float(self._opt("det_weight", 1.8)),
            float(self._opt("distill_weight", 1.2)),
            float(self._opt("smooth_weight", 1.0)),
        )
        meta = {"model_key": getattr(self.model, "model_key", ""), **self.cfg}
        self.monitor.start(epochs, meta=meta)
        self.callbacks.on_train_start(self)
        unlab_it = iter(unlab_loader) if unlab_loader is not None else None
        state: Dict[str, Any] = {"best_score": -1.0, "best_mse": 1e9, "best_metrics": {}}
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            self.callbacks.on_epoch_start(self)
            for p in bb_p:
                p.requires_grad_(epoch > freeze_ep)
            if polish_ep > 0 and epoch > (epochs - polish_ep):
                train_ds.affine_prob = min(train_ds.affine_prob, 0.25)
                train_ds.affine_strength = min(train_ds.affine_strength, 0.35)
            ep_t0 = time.perf_counter()
            losses, samples, unlab_it = self._run_epoch(
                epoch, resumed, train_loader, unlab_loader, unlab_it, teacher, proj,
                losses_fn, opt, scaler, ema, (bb_p, det_p, traj_p), amp_dtype, weights,
            )
            sched.step()
            val_m = None
            if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
                ev_m = ema.ema if (epoch >= 6 or resumed) else self.model
                val_m = self._eval_and_save(epoch, ev_m, val_loader, img_w, img_h, state)
            self.monitor.log_epoch(
                epoch, epochs, losses, val_m, lr=opt.param_groups[0]["lr"],
                samples=samples, epoch_sec=time.perf_counter() - ep_t0, device=self.device,
            )
            self.callbacks.on_epoch_end(self)
        val_json = self._opt("val_json", "")
        if val_json and bool(self._opt("save_val_grid", True)):
            self.validator.render_val_grid(val_json, self.save_dir / "val_grid.jpg", self.device)
        self.monitor.finish(state["best_metrics"])
        self.callbacks.on_train_end(self)
        return state["best_metrics"]

    def _run_epoch(
        self, epoch, resumed, loader, unlab_loader, unlab_it, teacher, proj,
        losses_fn, opt, scaler, ema, param_groups, amp_dtype, weights,
    ) -> Tuple[Dict[str, float], int, Any]:
        det_fn, fdat_fn, mhp_fn, hm_fn = losses_fn
        bb_p, det_p, traj_p = param_groups
        det_w, dist_w, smooth_w = weights
        self.model.train()
        if proj is not None:
            proj.train()
        s_det = s_traj = s_dist = 0.0
        n_steps = nan_streak = samples = 0
        for batch in loader:
            yuv6 = batch["yuv6"].to(self.device, non_blocking=True)
            rgb01 = batch["rgb01"].to(self.device, non_blocking=True)
            cmd = batch["command"].to(self.device, non_blocking=True)
            gt_wp = batch["waypoints"].to(self.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.device.type == "cuda"):
                det_out, traj_out, p4, p5 = forward_with_p4p5(self.model, yuv6, cmd)
                l_det = self._det_loss(det_out, batch, det_fn)
                l_traj = self._traj_loss(
                    traj_out, gt_wp, cmd, epoch + (3 if resumed else 0),
                    mhp_fn, fdat_fn, hm_fn, smooth_w,
                )
                l_dist, unlab_it = self._distill_loss(
                    teacher, proj, rgb01, p4, p5, unlab_loader, unlab_it, n_steps
                )
                loss = det_w * l_det + l_traj + dist_w * l_dist
            if not torch.isfinite(loss):
                nan_streak += 1
                opt.zero_grad(set_to_none=True)
                if nan_streak >= 8:
                    raise RuntimeError(f"Aborted at Ep {epoch}: {nan_streak} non-finite losses")
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            for grp, cap in ((bb_p, 2.0), (det_p, 5.0), (traj_p, 2.0)):
                torch.nn.utils.clip_grad_norm_(grp, cap)
            scaler.step(opt)
            scaler.update()
            ema.update(self.model)
            s_det += float(l_det.detach())
            s_traj += float(l_traj.detach())
            s_dist += float(l_dist.detach())
            samples, n_steps = samples + int(yuv6.shape[0]), n_steps + 1
        d = max(1, n_steps)
        return {"det": s_det / d, "traj": s_traj / d, "dist": s_dist / d}, samples, unlab_it

    @staticmethod
    def _det_loss(det_out: Dict[str, Any], batch: Dict[str, Any], det_fn: Any) -> torch.Tensor:
        if batch["cls"].numel() == 0:
            sc = det_out["scores"].float()
            return 0.5 * F.binary_cross_entropy_with_logits(sc, torch.zeros_like(sc))
        det_fp32 = {
            "boxes": det_out["boxes"].float(), "scores": det_out["scores"].float(),
            "feats": [f.float() for f in det_out["feats"]],
        }
        return det_fn(det_fp32, batch)[0].sum()

    @staticmethod
    def _traj_loss(
        traj_out: Any, gt_wp: torch.Tensor, cmd: torch.Tensor, eff_ep: int,
        mhp_fn: MHPLaplaceLoss, fdat_fn: FDATLoss, hm_fn: HeatmapLoss, smooth_w: float,
    ) -> torch.Tensor:
        if isinstance(traj_out, dict):
            return mhp_traj_loss(traj_out, gt_wp, cmd, eff_ep, mhp_fn, fdat_fn, hm_fn, smooth_w)
        gt = gt_wp.float()
        return (
            12.0 * fdat_fn(traj_out.float(), gt, (cmd != 0).float()).mean()
            + 40.0 * F.smooth_l1_loss(traj_out.float(), gt, beta=0.005)
            + 60.0 * F.mse_loss(traj_out.float(), gt)
        )

    def _distill_loss(
        self, teacher, proj, rgb01, p4, p5, unlab_loader, unlab_it, n_steps: int
    ) -> Tuple[torch.Tensor, Any]:
        if teacher is None or proj is None:
            return torch.tensor(0.0, device=self.device), unlab_it
        with torch.no_grad():
            t_feat = teacher(rgb01)
        l_dist = dense_distill_loss(proj(p4, p5), t_feat)
        if unlab_it is not None and unlab_loader is not None and (n_steps % 2 == 0):
            try:
                u_yuv, u_rgb = next(unlab_it)
            except StopIteration:
                unlab_it = iter(unlab_loader)
                u_yuv, u_rgb = next(unlab_it)
            u_p4, u_p5 = forward_backbone_p4p5(self.model, u_yuv.to(self.device))
            with torch.no_grad():
                u_t = teacher(u_rgb.to(self.device))
            l_dist = l_dist + 0.5 * dense_distill_loss(proj(u_p4, u_p5), u_t)
        return l_dist, unlab_it
