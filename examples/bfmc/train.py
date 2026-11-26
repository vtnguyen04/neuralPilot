"""BFMC Multi-Task Fine-Tuning Example (reuses core DetectionLoss, ModelEMA, Zoo Distill & MHP)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.bfmc.dataset import BFMCDataset, UnlabeledFrameDataset, bfmc_collate_fn
from examples.bfmc.metrics import evaluate_bfmc, sym_curve_dist
from neuro_pilot.nn.zoo import MHPLaplaceLoss, build_zoo_model
from neuro_pilot.nn.zoo.distill import DinoV3Teacher, DistillProj, dense_distill_loss
from neuro_pilot.utils.losses import DetectionLoss, FDATLoss, HeatmapLoss
from neuro_pilot.utils.torch_utils import ModelEMA


def _forward_with_intermediates(
    model: nn.Module, img: torch.Tensor, command: torch.Tensor
) -> Tuple[Dict[str, Any], Any, torch.Tensor, torch.Tensor]:
    """Execute GraphModel returning raw det dict, traj output, P4 (layer 4), and P5 (layer 6)."""
    inputs = {"img": img, "command": command}
    cache: Dict[int, Any] = {}
    prev: Any = inputs
    for idx, (layer, src) in enumerate(zip(model.layers, model._sources)):
        inp = model._resolve_ref(src, prev, cache, inputs)
        out = layer(*inp) if isinstance(inp, list) and getattr(layer, "unpack_inputs", False) else layer(inp)
        if idx in model._save_indices or idx in (4, 6, 23, 24):
            cache[idx] = out
        prev = out
    return cache[23], cache[24], cache[4], cache[6]


def _forward_backbone_p4p5(model: nn.Module, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = p4 = p5 = img
    for idx in range(7):
        x = model.layers[idx](x)
        if idx == 4:
            p4 = x
        elif idx == 6:
            p5 = x
    return p4, p5


def _load_matching_weights(model: nn.Module, ckpt_path: str | Path, prefixes: Optional[Tuple[str, ...]] = None) -> int:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
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


def _mhp_traj_loss(
    traj_out: Dict[str, torch.Tensor],
    gt_wp: torch.Tensor,
    cmd: torch.Tensor,
    epoch: int,
    mhp_fn: MHPLaplaceLoss,
    fdat_fn: FDATLoss,
    hm_fn: HeatmapLoss,
    smooth_weight: float,
) -> torch.Tensor:
    mu, mu0 = traj_out["mu"].float(), traj_out["mu0"].float()
    log_b, m_logits = traj_out["log_b"].float(), traj_out["mode_logits"].float()
    gt = gt_wp.float()
    mhp = mhp_fn(mu, log_b, m_logits, gt)
    with torch.no_grad():
        mode_err = torch.abs(mu - gt.unsqueeze(1)).mean(dim=(-2, -1))
        best_k = mode_err.argmin(dim=1)
        soft_tgt = F.softmax(-mode_err / 0.012, dim=-1)
    l_mode_kl = F.kl_div(F.log_softmax(m_logits, dim=-1), soft_tgt, reduction="batchmean")
    b_idx = torch.arange(mu.shape[0], device=mu.device)
    mu_best, mu0_best = mu[b_idx, best_k], mu0[b_idx, best_k]

    cmd_w = torch.where(cmd != 0, 1.5, 1.0)
    if epoch > 8:
        with torch.no_grad():
            hard_w = 1.0 + (torch.abs(mu_best - gt).mean(dim=(1, 2)) / 0.06).clamp(0.0, 1.5)
        cmd_w = cmd_w * hard_w
    gate = (cmd != 0).float()
    l_fdat = (fdat_fn(mu_best, gt, gate) * cmd_w).mean()
    l_l1 = (F.smooth_l1_loss(mu_best, gt, beta=0.005, reduction="none").mean(dim=(1, 2)) * cmd_w).mean()
    l_mse = (((mu_best - gt) ** 2).mean(dim=(1, 2)) * cmd_w).mean()
    l_crv = sym_curve_dist(mu_best, gt).mean()
    l_mono = F.relu(mu_best[:, 1:, 1] - mu_best[:, :-1, 1] + 0.005).mean()
    l_c0 = F.smooth_l1_loss(mu0_best, gt, beta=0.02)

    # S-curve-safe physics smoothness: match GT 1st/2nd derivatives + all-K 3rd-order jerk
    v_p, v_g = mu_best[:, 1:] - mu_best[:, :-1], gt[:, 1:] - gt[:, :-1]
    a_p, a_g = v_p[:, 1:] - v_p[:, :-1], v_g[:, 1:] - v_g[:, :-1]
    seg_cos = 1.0 - (F.normalize(v_p, dim=-1, eps=1e-5) * F.normalize(v_g, dim=-1, eps=1e-5)).sum(-1)
    seg_w = torch.ones_like(seg_cos)
    seg_w[:, -2:] = 2.5
    v_all = mu[:, :, 1:] - mu[:, :, :-1]
    a_all = v_all[:, :, 1:] - v_all[:, :, :-1]
    j_all = a_all[:, :, 1:] - a_all[:, :, :-1]
    l_smooth = (
        20.0 * F.smooth_l1_loss(v_p, v_g, beta=0.005)
        + 35.0 * F.smooth_l1_loss(a_p, a_g, beta=0.005)
        + 3.5 * (seg_cos * seg_w).mean()
        + 25.0 * F.smooth_l1_loss(j_all, torch.zeros_like(j_all), beta=0.004)
    )
    hm_logits = traj_out["heatmap"].float().clamp(-15.0, 15.0)
    gt_hm = hm_fn.generate_heatmap(gt * 2.0 - 1.0, hm_logits.shape[-2], hm_logits.shape[-1])
    l_hm = hm_fn(hm_logits, gt_hm).mean()
    return 0.75 * (
        0.8 * (mhp["loss"] + 3.5) + 2.5 * l_mode_kl + 3.0 * l_fdat + 15.0 * l_l1 + 25.0 * l_mse
        + 10.0 * l_crv + 15.0 * l_mono + 5.0 * l_c0 + 1.2 * l_hm + smooth_weight * l_smooth
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        BFMCDataset(args.train_json, args.img_w, args.img_h, augment=True, copy_paste_prob=args.copy_paste),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, collate_fn=bfmc_collate_fn,
    )
    val_loader = DataLoader(
        BFMCDataset(args.val_json, args.img_w, args.img_h, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=bfmc_collate_fn,
    )
    unlab_loader = unlab_iter = None
    if args.distill and args.unlabeled_json:
        unlab_loader = DataLoader(
            UnlabeledFrameDataset(args.unlabeled_json, args.img_w, args.img_h),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )
        unlab_iter = iter(unlab_loader)

    model = build_zoo_model(args.arch).to(device)
    if args.resume_weights:
        n_ok = _load_matching_weights(model, args.resume_weights)
        print(f"[INIT] Warm-started {args.arch} ({n_ok}/{len(model.state_dict())} tensors) from {args.resume_weights}")
    else:
        prefixes = tuple(f"layers.{i}." for i in range(1, 10))
        n_ok = _load_matching_weights(model, args.pilot_s_pt, prefixes=prefixes)
        print(f"[INIT] Loaded {n_ok} OpenPilot pilot_s backbone tensors into {args.arch}")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    teacher = DinoV3Teacher(args.dinov3_path).to(device=device, dtype=amp_dtype).eval() if args.distill else None
    proj = DistillProj(128, 256, teacher.dim).to(device) if teacher is not None else None

    det_proxy = SimpleNamespace(
        heads={"detect": model.layers[23]}, parameters=model.layers[23].parameters
    )
    det_loss_fn = DetectionLoss(det_proxy, config=SimpleNamespace(box=7.5, cls_det=0.6, dfl=1.5))
    fdat_fn = FDATLoss(base_loss_type="wing").to(device)
    mhp_fn = MHPLaplaceLoss(cls_weight=1.5, min_log_b=-3.5, max_log_b=1.0).to(device)
    hm_fn = HeatmapLoss(dice_weight=5.0, device=device).to(device)

    bb_params = [p for i in range(1, 10) for p in model.layers[i].parameters()]
    traj_params = list(model.layers[24].parameters())
    special = {id(p) for p in bb_params} | {id(p) for p in traj_params}
    det_params = [p for p in model.parameters() if id(p) not in special] + (list(proj.parameters()) if proj else [])

    lr_traj = args.lr_head * 0.5 if args.arch in ("pilot_mhp", "bfmc_mhp") else args.lr_head
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr_backbone, "weight_decay": 1e-2},
        {"params": det_params, "lr": args.lr_head, "weight_decay": 1e-2},
        {"params": traj_params, "lr": lr_traj, "weight_decay": 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))
    ema = ModelEMA(model, decay=args.ema_decay, tau=300)

    best_score, best_mse, best_metrics, history = -1.0, 1e9, {}, []
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        for p in bb_params:
            p.requires_grad_(epoch > args.freeze_epochs)
        model.train()
        if proj is not None:
            proj.train()
        sum_det = sum_traj = sum_dist = 0.0
        n_steps = nan_streak = 0

        for batch in train_loader:
            yuv6 = batch["yuv6"].to(device, non_blocking=True)
            rgb01 = batch["rgb01"].to(device, non_blocking=True)
            cmd = batch["command"].to(device, non_blocking=True)
            gt_wp = batch["waypoints"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                det_out, traj_out, p4, p5 = _forward_with_intermediates(model, yuv6, cmd)
                if batch["cls"].numel() == 0:
                    sc = det_out["scores"].float()
                    l_det = 0.5 * F.binary_cross_entropy_with_logits(sc, torch.zeros_like(sc))
                else:
                    det_fp32 = {
                        "boxes": det_out["boxes"].float(),
                        "scores": det_out["scores"].float(),
                        "feats": [f.float() for f in det_out["feats"]],
                    }
                    l_det = det_loss_fn(det_fp32, batch)[0].sum()
                if isinstance(traj_out, dict):
                    l_traj = _mhp_traj_loss(traj_out, gt_wp, cmd, epoch, mhp_fn, fdat_fn, hm_fn, args.smooth_weight)
                else:
                    gt = gt_wp.float()
                    l_traj = (
                        12.0 * fdat_fn(traj_out.float(), gt, (cmd != 0).float()).mean()
                        + 40.0 * F.smooth_l1_loss(traj_out.float(), gt, beta=0.005)
                        + 60.0 * F.mse_loss(traj_out.float(), gt)
                    )
                l_dist = torch.tensor(0.0, device=device)
                if teacher is not None and proj is not None:
                    with torch.no_grad():
                        t_feat = teacher(rgb01)
                    l_dist = dense_distill_loss(proj(p4, p5), t_feat)
                    if unlab_iter is not None and unlab_loader is not None and (n_steps % 2 == 0):
                        try:
                            u_yuv, u_rgb = next(unlab_iter)
                        except StopIteration:
                            unlab_iter = iter(unlab_loader)
                            u_yuv, u_rgb = next(unlab_iter)
                        u_p4, u_p5 = _forward_backbone_p4p5(model, u_yuv.to(device, non_blocking=True))
                        with torch.no_grad():
                            u_t = teacher(u_rgb.to(device, non_blocking=True))
                        l_dist = l_dist + 0.5 * dense_distill_loss(proj(u_p4, u_p5), u_t)
                loss = args.det_weight * l_det + l_traj + args.distill_weight * l_dist

            if not torch.isfinite(loss):
                nan_streak += 1
                opt.zero_grad(set_to_none=True)
                if nan_streak >= 8:
                    raise RuntimeError(f"Aborted at Ep {epoch}: {nan_streak} consecutive non-finite losses!")
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            sum_det += float(l_det.detach())
            sum_traj += float(l_traj.detach())
            sum_dist += float(l_dist.detach())
            n_steps += 1

        sched.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            ev_model = ema.ema if (epoch >= 6 or args.resume_weights) else model
            m = evaluate_bfmc(ev_model, val_loader, device, img_w=args.img_w, img_h=args.img_h)
            print(
                f"Ep {epoch:02d}/{args.epochs} ({time.perf_counter() - t0:5.0f}s) | "
                f"L_det={sum_det/n_steps:.3f} L_traj={sum_traj/n_steps:.3f} L_dist={sum_dist/n_steps:.3f} | "
                f"Score={m['score']:.2f} MSE={m['mse']:.5f} ADE={m['ade']:.4f} Lat={m['lat_err']:.4f} "
                f"Kink={m['end_kink_deg']:.1f}deg mAP50={m['map50']:.4f}",
                flush=True,
            )
            history.append({"epoch": epoch, **m})
            if epoch >= 5 and (not math.isfinite(m["mse"]) or m["mse"] > 0.25):
                raise RuntimeError(f"Early-stop guardrail triggered at Ep {epoch}: Val MSE={m['mse']}")
            payload = {
                "model_key": model.model_key, "state_dict": ev_model.state_dict(),
                "img_w": args.img_w, "img_h": args.img_h, "metrics": {"epoch": epoch, **m},
            }
            if m["mse"] < best_mse:
                best_mse = m["mse"]
                torch.save(payload, save_dir / "best_traj.pt")
            if m["score"] > best_score:
                best_score, best_metrics = m["score"], payload["metrics"]
                torch.save(payload, save_dir / "best.pt")

    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    (save_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    return best_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="pilot_mhp", choices=["pilot_det", "pilot_mhp", "bfmc", "bfmc_mhp"])
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--unlabeled-json", default="")
    p.add_argument("--pilot-s-pt", required=True)
    p.add_argument("--resume-weights", default="")
    p.add_argument("--dinov3-path", default="")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--img-w", type=int, default=640)
    p.add_argument("--img-h", type=int, default=512)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-backbone", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=1.2e-3)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--copy-paste", type=float, default=0.45)
    p.add_argument("--distill", action="store_true")
    p.add_argument("--distill-weight", type=float, default=1.2)
    p.add_argument("--det-weight", type=float, default=1.25)
    p.add_argument("--smooth-weight", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.992)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
