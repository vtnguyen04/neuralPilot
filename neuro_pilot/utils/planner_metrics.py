"""Pure trajectory geometry and multi-class detection mAP metric helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms


def sym_curve_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Differentiable per-sample symmetric point-to-polyline geometric lane distance `[B]`."""
    def _one_way(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
        p1, p2 = poly[:, None, :-1, :], poly[:, None, 1:, :]
        v = p2 - p1
        w = pts[:, :, None, :] - p1
        t = ((w * v).sum(-1) / ((v * v).sum(-1) + 1e-6)).clamp(0.0, 1.0)
        proj = p1 + t.unsqueeze(-1) * v
        return torch.sqrt(((pts[:, :, None, :] - proj) ** 2).sum(-1) + 1e-6).min(-1).values.mean(-1)

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
    """Compute scale-invariant 2nd-order jerk L2 and final-quarter chord kink angle (deg)."""
    t_len = pred.shape[1]
    scale = max(1.0, (t_len - 1) / 9.0)
    d1 = (pred[:, 1:, :] - pred[:, :-1, :]) * scale
    d2 = (d1[:, 1:, :] - d1[:, :-1, :]) * scale
    jerk = np.linalg.norm(d2, axis=-1).mean(axis=-1)
    span = max(1, (t_len - 1) // 8)
    v_prev = pred[:, -1 - span, :] - pred[:, -1 - 2 * span, :]
    v_last = pred[:, -1, :] - pred[:, -1 - span, :]
    v_prev = v_prev / (np.linalg.norm(v_prev, axis=-1, keepdims=True) + 1e-6)
    v_last = v_last / (np.linalg.norm(v_last, axis=-1, keepdims=True) + 1e-6)
    end_kink = np.degrees(np.arccos(np.clip((v_prev * v_last).sum(axis=-1), -1.0, 1.0)))
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
        out = torch.cat((xyxy[keep], sc[keep, None], cl[keep, None].float()), dim=1)
        results.append(out.cpu().numpy())
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
    grid = np.linspace(0.0, 1.0, 101)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(np.interp(grid, mrec, mpre), grid))


def evaluate_detection(
    all_preds: List[np.ndarray],
    all_gts: List[Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute COCO-style `mAP@50`, `mAP@50:95`, precision/recall@0.25, and per-class AP50."""
    num_classes = len(class_names)
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
    ap50_list, ap5095_list = [], []
    for c in range(num_classes):
        if gt_counts[c] == 0:
            continue
        if not stats_tp[c]:
            ap50_per_cls[class_names[c]] = 0.0
            ap50_list.append(0.0)
            ap5095_list.append(0.0)
            continue
        tp_all = np.concatenate(stats_tp[c], axis=0)
        cf_all = np.concatenate(stats_conf[c], axis=0)
        aps = [_compute_ap(tp_all[:, ti], cf_all, int(gt_counts[c])) for ti in range(len(iou_thrs))]
        ap50_per_cls[class_names[c]] = round(aps[0], 4)
        ap50_list.append(aps[0])
        ap5095_list.append(float(np.mean(aps)))

    return {
        "map50": float(np.mean(ap50_list)) if ap50_list else 0.0,
        "map50_95": float(np.mean(ap5095_list)) if ap5095_list else 0.0,
        "precision25": p25_tp / max(1, p25_tp + p25_fp),
        "recall25": p25_tp / max(1, p25_gt),
        "per_class_ap50": ap50_per_cls,
    }
