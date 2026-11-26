"""Shared image preprocessing, geometry, and NMS helpers for driving predictors."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

ImageLike = Union[str, Path, np.ndarray]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_bgr(source: ImageLike) -> np.ndarray:
    """Load an image path or validate an existing BGR ndarray."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {source}")
        return img
    if isinstance(source, np.ndarray):
        return source
    raise TypeError(f"Expected image path or numpy ndarray, got {type(source).__name__}")


def crop_2to1(bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Center-crop BGR frame to 2:1 aspect ratio over full width and resize."""
    height, width = bgr.shape[:2]
    crop_h = min(height, max(1, width // 2))
    y0 = max(0, (height - crop_h) // 2)
    cropped = bgr[y0 : y0 + crop_h, :]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def to_rgb_01(bgr: np.ndarray, out_w: int, out_h: int) -> torch.Tensor:
    """Convert BGR frame to `[1, 3, out_h, out_w]` RGB float32 tensor in `[0, 1]`."""
    resized = crop_2to1(bgr, out_w, out_h)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))


def to_imagenet_tensor(bgr: np.ndarray, out_w: int, out_h: int) -> torch.Tensor:
    """Convert BGR frame to `[1, 3, out_h, out_w]` ImageNet-normalized float32 tensor."""
    resized = crop_2to1(bgr, out_w, out_h)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(norm[None]))


def bgr_to_yuv6(bgr: np.ndarray, width: int = 512, height: int = 256) -> np.ndarray:
    """Pack a BGR frame into openpilot's `(6, H/2, W/2)` uint8 YUV420 tensor."""
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y_plane = yuv[:height, :]
    u_plane = yuv[height : height + height // 4, :].reshape(height // 2, width // 2)
    v_plane = yuv[height + height // 4 : height + height // 2, :].reshape(height // 2, width // 2)
    out = np.empty((6, height // 2, width // 2), dtype=np.uint8)
    out[0] = y_plane[0::2, 0::2]
    out[1] = y_plane[1::2, 0::2]
    out[2] = y_plane[0::2, 1::2]
    out[3] = y_plane[1::2, 1::2]
    out[4] = u_plane
    out[5] = v_plane
    return out


def sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def nms_xywh(
    raw_out: np.ndarray,
    conf_thres: float = 0.45,
    iou_thres: float = 0.45,
) -> List[Tuple[float, float, float, float, float, int]]:
    """Decode `[1, 4+C, N]` box predictions with OpenCV NMS into `(x1,y1,x2,y2,score,cls)`."""
    preds = raw_out[0].T
    boxes, probs = preds[:, :4], preds[:, 4:]
    cls_ids = np.argmax(probs, axis=1)
    scores = probs[np.arange(len(probs)), cls_ids]
    keep = scores > conf_thres
    if not np.any(keep):
        return []
    kept_b, kept_s, kept_c = boxes[keep], scores[keep], cls_ids[keep]
    x1 = kept_b[:, 0] - kept_b[:, 2] / 2.0
    y1 = kept_b[:, 1] - kept_b[:, 3] / 2.0
    w_box, h_box = kept_b[:, 2], kept_b[:, 3]
    idxs = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, w_box, h_box], axis=1).tolist(),
        kept_s.tolist(),
        conf_thres,
        iou_thres,
    )
    if len(idxs) == 0:
        return []
    return [
        (
            float(x1[i]),
            float(y1[i]),
            float(x1[i] + w_box[i]),
            float(y1[i] + h_box[i]),
            float(kept_s[i]),
            int(kept_c[i]),
        )
        for i in np.array(idxs).reshape(-1)
    ]


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Normalize a 2D depth prediction to an Inferno BGR heatmap for visualization."""
    lo, hi = np.percentile(depth_map, [2, 98])
    norm = np.clip((depth_map - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
