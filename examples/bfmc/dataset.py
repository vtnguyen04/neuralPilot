"""BFMC dataset, right-hand-safe geometric/photometric augmentations, and unlabeled loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from neuro_pilot.engine.predictors.utils import bgr_to_yuv6

# Train-only copy-paste classes (symmetric patches allow safe horizontal flip)
_COPY_PASTE_IDS = {0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12}
_SYM_PATCH_IDS = {1, 2, 3, 4, 11}


def _bgr_to_tensors(bgr: np.ndarray, w: int, h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if bgr.shape[1] != w or bgr.shape[0] != h:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    yuv6 = torch.from_numpy(bgr_to_yuv6(bgr, width=w, height=h))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return yuv6, torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def _photometric_aug(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    if random.random() < 0.85:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-8.0, 8.0)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.75, 1.30), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.70, 1.35), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.40:
        h, w = img.shape[:2]
        ang = random.uniform(0.0, 2.0 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
        gain = 1.0 + ramp * random.uniform(-0.35, 0.35)
        img = np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    if random.random() < 0.12:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.15:
        noise = np.random.normal(0.0, random.uniform(3.0, 9.0), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    b1x1, b1y1 = box[0] - box[2] * 0.5, box[1] - box[3] * 0.5
    b1x2, b1y2 = box[0] + box[2] * 0.5, box[1] + box[3] * 0.5
    b2x1, b2y1 = boxes[:, 0] - boxes[:, 2] * 0.5, boxes[:, 1] - boxes[:, 3] * 0.5
    b2x2, b2y2 = boxes[:, 0] + boxes[:, 2] * 0.5, boxes[:, 1] + boxes[:, 3] * 0.5
    inter = np.maximum(0.0, np.minimum(b1x2, b2x2) - np.maximum(b1x1, b2x1)) * np.maximum(
        0.0, np.minimum(b1y2, b2y2) - np.maximum(b1y1, b2y1)
    )
    a1 = max(1e-6, (b1x2 - b1x1) * (b1y2 - b1y1))
    a2 = np.maximum(1e-6, (b2x2 - b2x1) * (b2y2 - b2y1))
    return inter / (a1 + a2 - inter + 1e-6)


class BFMCDataset(Dataset):
    """Multi-task BFMC dataset returning aligned YUV6 + RGB01, command, waypoints, and bboxes."""

    def __init__(
        self,
        manifest_path: str | Path,
        img_w: int = 640,
        img_h: int = 512,
        augment: bool = False,
        copy_paste_prob: float = 0.45,
    ) -> None:
        self.records: List[Dict[str, Any]] = json.loads(Path(manifest_path).read_text())
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.augment = bool(augment)
        self.copy_paste_prob = float(copy_paste_prob) if augment else 0.0
        self.crop_bank: Dict[int, List[Tuple[np.ndarray, float, float]]] = {}
        if self.copy_paste_prob > 0:
            self._build_crop_bank()

    def _build_crop_bank(self) -> None:
        for rec in self.records:
            rare = [b for b in rec["bboxes"] if int(b[0]) in _COPY_PASTE_IDS]
            if not rare:
                continue
            bgr = cv2.imread(rec["img_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            for cid, cx, cy, bw, bh in rare:
                x1, y1 = max(0, int((cx - bw * 0.5) * w)), max(0, int((cy - bh * 0.5) * h))
                x2, y2 = min(w, int((cx + bw * 0.5) * w)), min(h, int((cy + bh * 0.5) * h))
                if (x2 - x1) >= 12 and (y2 - y1) >= 12:
                    self.crop_bank.setdefault(int(cid), []).append(
                        (bgr[y1:y2, x1:x2].copy(), bw, bh)
                    )

    def __len__(self) -> int:
        return len(self.records)

    def _apply_affine(
        self, bgr: np.ndarray, wps: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sx, sy = random.uniform(0.86, 1.14), random.uniform(0.88, 1.12)
        tx, ty = random.uniform(-0.08, 0.08), random.uniform(-0.06, 0.06)
        rad = np.deg2rad(random.uniform(-5.5, 5.5) if random.random() < 0.6 else 0.0)
        ca, sa = float(np.cos(rad)), float(np.sin(rad))

        dx, dy = (wps[:, 0] - 0.5) * sx, (wps[:, 1] - 0.5) * sy
        new_wps = np.stack((ca * dx - sa * dy + 0.5 + tx, sa * dx + ca * dy + 0.5 + ty), axis=-1)
        if np.any(new_wps[:, 0] < -0.05) or np.any(new_wps[:, 0] > 1.05):
            return bgr, wps, boxes
        if np.any(new_wps[:, 1] < 0.05) or np.any(new_wps[:, 1] > 1.05):
            return bgr, wps, boxes

        h, w = bgr.shape[:2]
        a00, a01 = ca * sx, -sa * sy * (w / h)
        a10, a11 = sa * sx * (h / w), ca * sy
        b0 = (0.5 + tx) * w - a00 * (0.5 * w) - a01 * (0.5 * h)
        b1 = (0.5 + ty) * h - a10 * (0.5 * w) - a11 * (0.5 * h)
        warped = cv2.warpAffine(
            bgr, np.array([[a00, a01, b0], [a10, a11, b1]], dtype=np.float32),
            (w, h), borderMode=cv2.BORDER_REPLICATE,
        )
        if len(boxes) == 0:
            return warped, new_wps, boxes
        kept = []
        for cid, cx, cy, bw, bh in boxes:
            corners = np.array(
                [
                    [cx - bw * 0.5, cy - bh * 0.5], [cx + bw * 0.5, cy - bh * 0.5],
                    [cx - bw * 0.5, cy + bh * 0.5], [cx + bw * 0.5, cy + bh * 0.5],
                ],
                dtype=np.float32,
            )
            cdx, cdy = (corners[:, 0] - 0.5) * sx, (corners[:, 1] - 0.5) * sy
            nx = np.clip(ca * cdx - sa * cdy + 0.5 + tx, 0.0, 1.0)
            ny = np.clip(sa * cdx + ca * cdy + 0.5 + ty, 0.0, 1.0)
            x1, x2, y1, y2 = float(nx.min()), float(nx.max()), float(ny.min()), float(ny.max())
            if (x2 - x1) >= 0.015 and (y2 - y1) >= 0.018:
                kept.append([cid, (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1])
        return warped, new_wps.astype(np.float32), np.asarray(kept, dtype=np.float32).reshape(-1, 5)

    def _apply_copy_paste(self, bgr: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.crop_bank:
            return bgr, boxes
        h, w = bgr.shape[:2]
        out_img = bgr.copy()
        extra, cids = [], list(self.crop_bank.keys())
        for _ in range(random.randint(1, 2)):
            cid = random.choice(cids)
            crop, bw_n, bh_n = random.choice(self.crop_bank[cid])
            if cid in _SYM_PATCH_IDS and random.random() < 0.5:
                crop = cv2.flip(crop, 1)
            scale = random.uniform(0.85, 1.15)
            pw = max(14, min(int(w * 0.28), int(bw_n * w * scale)))
            ph = max(14, min(int(h * 0.35), int(bh_n * h * scale)))
            cx_n = random.uniform(0.12, 0.88)
            cy_n = random.uniform(0.24, 0.54) if cid == 2 else random.uniform(0.16, 0.44)
            cand = np.array([cx_n, cy_n, pw / w, ph / h], dtype=np.float32)
            existing = boxes[:, 1:5] if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            if len(existing) > 0 and _box_iou_np(cand, existing).max() > 0.05:
                continue
            x1, y1 = max(0, int(cx_n * w - pw * 0.5)), max(0, int(cy_n * h - ph * 0.5))
            x2, y2 = min(w, x1 + pw), min(h, y1 + ph)
            if (x2 - x1) < 12 or (y2 - y1) < 12:
                continue
            patch = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            alpha = np.ones((y2 - y1, x2 - x1, 1), dtype=np.float32)
            alpha[0, :] = alpha[-1, :] = alpha[:, 0] = alpha[:, -1] = 0.5
            roi = out_img[y1:y2, x1:x2].astype(np.float32)
            out_img[y1:y2, x1:x2] = (alpha * patch + (1.0 - alpha) * roi).astype(np.uint8)
            extra.append([cid, (x1 + x2) * 0.5 / w, (y1 + y2) * 0.5 / h, (x2 - x1) / w, (y2 - y1) / h])
        if extra:
            arr = np.asarray(extra, dtype=np.float32)
            boxes = np.vstack([boxes, arr]) if len(boxes) > 0 else arr
        return out_img, boxes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        bgr = cv2.resize(cv2.imread(rec["img_path"], cv2.IMREAD_COLOR), (self.img_w, self.img_h))
        wps = np.asarray(rec["waypoints"], dtype=np.float32).copy()
        boxes = (
            np.asarray(rec["bboxes"], dtype=np.float32).reshape(-1, 5).copy()
            if rec["bboxes"]
            else np.zeros((0, 5), dtype=np.float32)
        )
        if self.augment:
            # Never flip the full road image: BFMC enforces right-hand traffic & CCW roundabouts
            if random.random() < 0.75:
                bgr, wps, boxes = self._apply_affine(bgr, wps, boxes)
            if random.random() < self.copy_paste_prob:
                bgr, boxes = self._apply_copy_paste(bgr, boxes)
            bgr = _photometric_aug(bgr)
        yuv6, rgb01 = _bgr_to_tensors(bgr, self.img_w, self.img_h)
        return {
            "yuv6": yuv6,
            "rgb01": rgb01,
            "command": torch.tensor(int(rec["command"]), dtype=torch.long),
            "waypoints": torch.from_numpy(wps),
            "cls": torch.from_numpy(boxes[:, 0]) if len(boxes) else torch.zeros((0,)),
            "bboxes": torch.from_numpy(boxes[:, 1:5]) if len(boxes) else torch.zeros((0, 4)),
        }


class UnlabeledFrameDataset(Dataset):
    """Unlabeled driving frames returning aligned `(yuv6, rgb01)` pairs for REPA distillation."""

    def __init__(self, json_path: str | Path, img_w: int = 640, img_h: int = 512) -> None:
        self.paths: List[str] = json.loads(Path(json_path).read_text())
        self.img_w, self.img_h = int(img_w), int(img_h)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bgr = cv2.resize(cv2.imread(self.paths[idx], cv2.IMREAD_COLOR), (self.img_w, self.img_h))
        return _bgr_to_tensors(_photometric_aug(bgr), self.img_w, self.img_h)


def bfmc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length GT boxes with batch indices for `DetectionLoss`."""
    b_idx = [torch.full((len(x["cls"]),), i, dtype=torch.float32) for i, x in enumerate(batch)]
    return {
        "yuv6": torch.stack([x["yuv6"] for x in batch]),
        "rgb01": torch.stack([x["rgb01"] for x in batch]),
        "command": torch.stack([x["command"] for x in batch]),
        "waypoints": torch.stack([x["waypoints"] for x in batch]),
        "batch_idx": torch.cat(b_idx) if b_idx else torch.zeros((0,)),
        "cls": torch.cat([x["cls"] for x in batch]),
        "bboxes": torch.cat([x["bboxes"] for x in batch]),
    }
