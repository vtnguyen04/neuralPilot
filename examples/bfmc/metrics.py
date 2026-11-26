"""Single-source-of-truth BFMC labels, curve geometry, NMS, and frozen multi-task evaluator."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

CLS_NAMES = (
    "Stop", "Pedestrian", "Car", "Priority", "Crosswalk", "Oneway", "Stop-Line",
    "Parking", "Highway-Entry", "Roundabout", "Highway-Exit", "Traffic-Light", "No-Entry",
)
CMD_NAMES = ("FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "STRAIGHT")


def sym_curve_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Differentiable per-sample symmetric point-to-polyline geometric lane distance `[B]`."""
    def _one_way(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
        p1, p2 = poly[:, None, :-1, :], poly[:, None, 1:, :]
        v = p2 - p1
        w = pts[:, :, None, :] - p1
        t = ((w * v).sum(-1) / ((v * v).sum(-1) + 1e-6)).clamp(0.0, 1.0)
        proj = p1 + t.unsqueeze(-1) * v
        return torch.sqrt(((pts[:, :, None, :] - proj) ** 2).sum(-1) + 1e-6).min(dim=-1).values.mean(dim=-1)

    return 0.5 * (_one_way(pred, gt) + _one_way(gt, pred))


def frenet_lateral_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute per-sample mean absolute cross-track (normal) error `[B]`."""
    tang = np.zeros_like(gt)
    tang[:, 1:-1] = gt[:, 2:] - gt[:, :-2]
    tang[:, 0] = gt[:, 1] - gt[:, 0]
    tang[:, -1] = gt[:, -1] - gt[:, -2]
    tang = tang / (np.linalg.norm(tang, axis=-1, keepdims=True) + 1e-6)
    normal = np.stack([-tang[..., 1], tang[..., 0]], axis=-1)
    return np.abs(((pred - gt) * normal).sum(axis=-1)).mean(axis=-1)


def traj_smoothness(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample mean 2nd-order jerk L2 and final-segment kink angle (deg)."""
    d1 = pred[:, 1:, :] - pred[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    jerk = np.linalg.norm(d2, axis=-1).mean(axis=-1)
    seg = np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-6
    v7, v8 = d1[:, -2, :] / seg[:, -2, :], d1[:, -1, :] / seg[:, -1, :]
    end_kink = np.degrees(np.arccos(np.clip((v7 * v8).sum(axis=-1), -1.0, 1.0)))
    return jerk, end_kink


def decode_nms(
    decoded: torch.Tensor,
    img_w: int = 640,
    img_h: int = 512,
    conf_thres: float = 0.01,
    iou_thres: float = 0.55,
    max_det: int = 100,
) -> List[np.ndarray]:
    """Run class-aware batched NMS on `[B, 4+C, N]` outputs -> list of `[K, 6] xyxy,conf,cls`."""
    results = []
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=decoded.device, dtype=decoded.dtype)
    for b in range(decoded.shape[0]):
        pred = decoded[b].T
        cxcywh, probs = pred[:, :4], pred[:, 4:]
        scores, cls_ids = probs.max(dim=1)
        mask = scores > conf_thres
        if not mask.any():
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue
        cx, cy, bw, bh = cxcywh[mask].unbind(dim=1)
        xyxy = torch.stack((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5), dim=1)
        xyxy = (xyxy / scale).clamp(0.0, 1.0)
        sc, cl = scores[mask], cls_ids[mask]
        keep = batched_nms(xyxy.float(), sc.float(), cl, iou_thres)[:max_det]
        results.append(torch.cat((xyxy[keep], sc[keep, None], cl[keep, None].float()), dim=1).cpu().numpy())
    return results


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    aa = np.maximum(1e-6, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    ab = np.maximum(1e-6, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)


def _compute_ap(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    tp = tp[np.argsort(-conf)]
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1.0 - tp)
    rec, prec = tp_c / (n_gt + 1e-9), tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], prec, [0.0])))))
    return float(np.trapz(np.interp(np.linspace(0.0, 1.0, 101), mrec, mpre), np.linspace(0.0, 1.0, 101)))


def _evaluate_detection(
    all_preds: List[np.ndarray], all_gts: List[Tuple[np.ndarray, np.ndarray]], num_classes: int = 13
) -> Dict[str, Any]:
    iou_thrs = np.linspace(0.50, 0.95, 10)
    stats_tp: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    stats_conf: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    p25_tp = p25_fp = p25_gt = 0

    for preds, (gt_cls, gt_xyxy) in zip(all_preds, all_gts):
        for c in gt_cls:
            gt_counts[int(c)] += 1
        p25_gt += len(gt_cls)
        if len(preds) == 0:
            continue
        p_xyxy, p_conf, p_cls = preds[:, :4], preds[:, 4], preds[:, 5].astype(np.int64)
        tp_mat = np.zeros((len(preds), len(iou_thrs)), dtype=np.float32)
        for c in np.unique(p_cls):
            pm, gm = np.where(p_cls == c)[0], np.where(gt_cls == c)[0]
            if len(gm) == 0:
                continue
            ious = _iou_matrix(p_xyxy[pm], gt_xyxy[gm])
            for ti, thr in enumerate(iou_thrs):
                matched = set()
                for pi in np.argsort(-p_conf[pm]):
                    bg = int(np.argmax(ious[pi]))
                    if ious[pi, bg] >= thr and bg not in matched:
                        tp_mat[pm[pi], ti] = 1.0
                        matched.add(bg)
        for c in range(num_classes):
            pm = np.where(p_cls == c)[0]
            if len(pm) > 0:
                stats_tp[c].append(tp_mat[pm])
                stats_conf[c].append(p_conf[pm])
        hi = p_conf >= 0.25
        p25_tp += int(tp_mat[hi, 0].sum())
        p25_fp += int((1.0 - tp_mat[hi, 0]).sum())

    ap50_per_cls: Dict[str, float] = {}
    ap50_vals, ap95_vals = [], []
    for c in range(num_classes):
        if gt_counts[c] == 0:
            continue
        if not stats_tp[c]:
            ap50_per_cls[CLS_NAMES[c]] = 0.0
            ap50_vals.append(0.0)
            ap95_vals.append(0.0)
            continue
        tp_c, cf_c = np.concatenate(stats_tp[c]), np.concatenate(stats_conf[c])
        aps = [_compute_ap(tp_c[:, ti], cf_c, int(gt_counts[c])) for ti in range(len(iou_thrs))]
        ap50_per_cls[CLS_NAMES[c]] = round(aps[0], 4)
        ap50_vals.append(aps[0])
        ap95_vals.append(float(np.mean(aps)))

    return {
        "map50": float(np.mean(ap50_vals)) if ap50_vals else 0.0,
        "map50_95": float(np.mean(ap95_vals)) if ap95_vals else 0.0,
        "precision25": float(p25_tp / max(1, p25_tp + p25_fp)),
        "recall25": float(p25_tp / max(1, p25_gt)),
        "per_class_ap50": ap50_per_cls,
    }


@torch.no_grad()
def evaluate_bfmc(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device, img_w: int = 640, img_h: int = 512
) -> Dict[str, Any]:
    """Run frozen video-grouped BFMC evaluation (Trajectory ADE/FDE/Lat/Curve/Kink + COCO mAP)."""
    was_training = model.training
    model.eval()
    ades, fdes, lats, mses, curves, jerks, kinks, min_mses, cmds = [], [], [], [], [], [], [], [], []
    all_preds: List[np.ndarray] = []
    all_gts: List[Tuple[np.ndarray, np.ndarray]] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for batch in val_loader:
        yuv6 = batch["yuv6"].to(device, non_blocking=True)
        cmd = batch["command"].to(device, non_blocking=True)
        gt_t = batch["waypoints"]
        gt_wp = gt_t.numpy()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(img=yuv6, command=cmd)
        pred_t = out["waypoints"].float().cpu()
        pred_wp = pred_t.numpy()
        diff = pred_wp - gt_wp
        dist = np.linalg.norm(diff, axis=-1)
        ades.append(dist.mean(axis=-1))
        fdes.append(dist[:, -1])
        lats.append(frenet_lateral_err(pred_wp, gt_wp))
        mses.append((diff ** 2).mean(axis=(1, 2)))
        curves.append(sym_curve_dist(pred_t, gt_t.float()).numpy())
        jk, ek = traj_smoothness(pred_wp)
        jerks.append(jk)
        kinks.append(ek)
        if "mu" in out:
            mu_np = out["mu"].float().cpu().numpy()
            min_mses.append(((mu_np - gt_wp[:, None, :, :]) ** 2).mean(axis=(2, 3)).min(axis=1))
        cmds.append(batch["command"].numpy())
        all_preds.extend(decode_nms(out["det"].float(), img_w=img_w, img_h=img_h))

        b_idx, b_cls, b_box = batch["batch_idx"].numpy(), batch["cls"].numpy(), batch["bboxes"].numpy()
        for i in range(yuv6.shape[0]):
            m = b_idx == i
            if np.any(m):
                bx = b_box[m]
                xyxy = np.stack(
                    [bx[:, 0] - bx[:, 2] * 0.5, bx[:, 1] - bx[:, 3] * 0.5,
                     bx[:, 0] + bx[:, 2] * 0.5, bx[:, 1] + bx[:, 3] * 0.5], axis=1,
                )
                all_gts.append((b_cls[m].astype(np.int64), np.clip(xyxy, 0.0, 1.0)))
            else:
                all_gts.append((np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)))

    if was_training:
        model.train()

    ade_arr, cmd_arr = np.concatenate(ades), np.concatenate(cmds)
    ade, fde = float(ade_arr.mean()), float(np.concatenate(fdes).mean())
    lat, mse = float(np.concatenate(lats).mean()), float(np.concatenate(mses).mean())
    crv = float(np.concatenate(curves).mean())
    det_res = _evaluate_detection(all_preds, all_gts)
    res: Dict[str, Any] = {
        "score": round(50.0 * det_res["map50"] + 50.0 * max(0.0, 1.0 - ade / 0.12), 3),
        "mse": round(mse, 6),
        "ade": round(ade, 5),
        "ade_px640": round(ade * 640.0, 2),
        "fde": round(fde, 5),
        "fde_px640": round(fde * 640.0, 2),
        "lat_err": round(lat, 5),
        "lat_px640": round(lat * 640.0, 2),
        "curve_err": round(crv, 5),
        "curve_px640": round(crv * 640.0, 2),
        "jerk_l2": round(float(np.concatenate(jerks).mean()), 5),
        "end_kink_deg": round(float(np.concatenate(kinks).mean()), 2),
        "cmd_ade": {
            CMD_NAMES[c]: round(float(ade_arr[cmd_arr == c].mean()), 5)
            for c in range(4) if np.any(cmd_arr == c)
        },
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in det_res.items()},
    }
    if min_mses:
        res["min_mse5"] = round(float(np.concatenate(min_mses).mean()), 6)
    return res
