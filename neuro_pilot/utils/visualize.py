"""Unified task-aware visualizers for all NeuroPilot tasks (zero-DRY over overlay/viz)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from neuro_pilot.utils.overlay import (
    render_depth_panel,
    render_perception_boxes,
    render_seg_overlay,
)

DEFAULT_CMD_NAMES: Tuple[str, ...] = ("FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "STRAIGHT")
DEFAULT_CLS_NAMES: Tuple[str, ...] = (
    "Stop", "Pedestrian", "Car", "Priority", "Crosswalk", "Oneway", "Stop-Line",
    "Parking", "Highway-Entry", "Roundabout", "Highway-Exit", "Traffic-Light", "No-Entry",
)
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (60, 70, 255), (255, 80, 220), (80, 230, 120), (0, 210, 255), (255, 220, 80),
    (255, 140, 40), (255, 255, 110), (255, 150, 90), (120, 255, 180),
    (255, 170, 0), (120, 180, 255), (90, 255, 255), (40, 40, 240),
)


def _draw_corridor_2d(
    vis: np.ndarray, wps_norm: np.ndarray, w: int, h: int
) -> None:
    """Draw smooth perspective 2D drivable corridor with continuous right-hand normal."""
    pts = np.asarray(wps_norm, dtype=np.float32)
    if pts.ndim != 2 or len(pts) < 2:
        return
    px_pts = np.column_stack([
        np.clip(pts[:, 0] * w, 0.0, w - 1.0),
        np.clip(pts[:, 1] * h, 0.0, h - 1.0),
    ])
    tang = np.gradient(px_pts, axis=0)
    tang_u = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6)
    nx, ny = -tang_u[:, 1], tang_u[:, 0]
    ox, oy = nx, 0.42 * ny
    ts = np.linspace(0.0, 1.0, len(px_pts), dtype=np.float32)
    half_w = (1.0 - 0.65 * ts) * (w * 0.065)
    half_w = np.maximum(half_w, 6.0)
    left = np.column_stack([
        np.clip(px_pts[:, 0] - half_w * ox, 0, w - 1),
        np.clip(px_pts[:, 1] - half_w * oy, 0, h - 1),
    ])
    right = np.column_stack([
        np.clip(px_pts[:, 0] + half_w * ox, 0, w - 1),
        np.clip(px_pts[:, 1] + half_w * oy, 0, h - 1),
    ])
    poly = np.vstack([left, right[::-1]]).astype(np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly], (255, 200, 40), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, vis, 0.72, 0.0, vis)
    ipts = np.round(px_pts).astype(np.int32)
    for i in range(len(ipts) - 1):
        cv2.line(vis, tuple(ipts[i]), tuple(ipts[i + 1]), (255, 235, 90), 3, cv2.LINE_AA)
    step = max(1, len(ipts) // 11)
    for idx in list(range(0, len(ipts), step)) + [len(ipts) - 1]:
        cv2.circle(vis, tuple(ipts[idx]), 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_planner_frame(
    frame_bgr: np.ndarray,
    output: Dict[str, Any],
    command: int = 0,
    gt_waypoints: Optional[np.ndarray] = None,
    conf_thres: float = 0.30,
    class_names: Sequence[str] = DEFAULT_CLS_NAMES,
    cmd_names: Sequence[str] = DEFAULT_CMD_NAMES,
    tag: str = "",
) -> np.ndarray:
    """Render 2D planner trajectory corridor, optional gold GT path, boxes, and HUD."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    if gt_waypoints is not None and len(gt_waypoints) >= 2:
        gpts = np.round(np.asarray(gt_waypoints, dtype=np.float32) * [w, h]).astype(np.int32)
        for i in range(len(gpts) - 1):
            cv2.line(vis, tuple(gpts[i]), tuple(gpts[i + 1]), (0, 215, 255), 2, cv2.LINE_AA)

    wps = output.get("waypoints")
    if wps is not None:
        _draw_corridor_2d(vis, np.asarray(wps, dtype=np.float32), w, h)

    dets = output.get("detections")
    if dets is not None and len(dets) > 0:
        for row in dets:
            x1, y1, x2, y2, conf, cid = row[:6]
            if float(conf) < conf_thres:
                continue
            ci = int(cid)
            col = _PALETTE[ci % len(_PALETTE)]
            if max(float(x2), float(y2)) <= 1.5:
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            else:
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis, p1, p2, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.rectangle(vis, p1, p2, col, 2, cv2.LINE_AA)
            cname = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
            label = f"{cname} {float(conf):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 6, p1[1] - 4)
            cv2.rectangle(vis, (p1[0], ty - th - 5), (p1[0] + tw + 6, ty + 2), (18, 22, 28), -1)
            cv2.putText(vis, label, (p1[0] + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    cmd_txt = cmd_names[command] if 0 <= command < len(cmd_names) else str(command)
    hud = f"{tag} | CMD: {cmd_txt}" if tag else f"CMD: {cmd_txt}"
    cv2.rectangle(vis, (0, 0), (w, 26), (16, 20, 26), -1)
    cv2.putText(vis, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 245, 255), 1, cv2.LINE_AA)
    return vis


def render_seg_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], **_: Any
) -> np.ndarray:
    """Render semantic segmentation class map overlay."""
    cmap = output.get("class_map")
    return render_seg_overlay(frame_bgr, cmap) if cmap is not None else frame_bgr.copy()


def render_depth_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], **_: Any
) -> np.ndarray:
    """Render monocular depth panel blended or side-by-side."""
    depth_vis = output.get("depth_vis")
    if depth_vis is not None:
        return cv2.resize(depth_vis, (frame_bgr.shape[1], frame_bgr.shape[0]))
    depth = output.get("depth")
    if depth is not None:
        return render_depth_panel(depth, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return frame_bgr.copy()


def render_lane_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], src_wh: Tuple[int, int] = (1024, 512), **_: Any
) -> np.ndarray:
    """Render ego-lane polyline points onto `frame_bgr`."""
    vis = frame_bgr.copy()
    pts = output.get("points") or []
    if len(pts) < 2:
        return vis
    h, w = vis.shape[:2]
    sx, sy = w / float(src_wh[0]), h / float(src_wh[1])
    scaled = [(int(x * sx), int(y * sy)) for x, y in pts]
    for i in range(len(scaled) - 1):
        cv2.line(vis, scaled[i], scaled[i + 1], (255, 220, 60), 3, cv2.LINE_AA)
    return vis


def render_cipo_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
) -> np.ndarray:
    """Render vehicle/CIPO boxes and optional distance/curvature HUD."""
    boxes = output.get("boxes")
    vis = (
        render_perception_boxes(frame_bgr, boxes, src_wh=kwargs.get("src_wh", (1024, 512)))
        if boxes is not None
        else frame_bgr.copy()
    )
    if "distance_m" in output:
        txt = f"Lead: {output['distance_m']:.1f}m | Curv: {output.get('curvature', 0.0):.4f}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (16, 20, 26), -1)
        cv2.putText(vis, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 180), 1)
    return vis


def render_world_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], **_: Any
) -> np.ndarray:
    """Render side-by-side input vs reconstructed world-model frame."""
    recon = output.get("image_bgr")
    if recon is None:
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    return np.hstack([frame_bgr, cv2.resize(recon, (w, h))])


def render_driver_frame(
    frame_bgr: np.ndarray, output: Dict[str, Any], **_: Any
) -> np.ndarray:
    """Render driver monitoring summary HUD onto `frame_bgr`."""
    vis = frame_bgr.copy()
    vec = output.get("outputs")
    n_vals = int(np.asarray(vec).size) if vec is not None else 0
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (16, 20, 26), -1)
    cv2.putText(
        vis, f"Driver Monitoring | outputs={n_vals}", (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 245, 255), 1, cv2.LINE_AA,
    )
    return vis


def render_image_grid(tiles: List[np.ndarray], cols: int = 3) -> np.ndarray:
    """Stack equal-sized BGR tiles into a clean grid padded with dark frames if needed."""
    if not tiles:
        return np.zeros((320, 512, 3), dtype=np.uint8)
    cols = max(1, int(cols))
    padded = list(tiles)
    while len(padded) % cols != 0:
        padded.append(np.zeros_like(padded[0]))
    return np.vstack([np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)])


def visualize_prediction(
    task_name: str, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
) -> np.ndarray:
    """Dispatch task-specific visualization cleanly by canonical task name."""
    t = task_name.lower()
    if t in {"planner", "e2e_planning", "bfmc", "trajectory"}:
        return render_planner_frame(frame_bgr, output, **kwargs)
    if t in {"seg", "segmentation"}:
        return render_seg_frame(frame_bgr, output, **kwargs)
    if t == "depth":
        return render_depth_frame(frame_bgr, output, **kwargs)
    if t == "lane":
        return render_lane_frame(frame_bgr, output, **kwargs)
    if t in {"cipo", "drive", "detection", "detect"}:
        return render_cipo_frame(frame_bgr, output, **kwargs)
    if t in {"world", "world_model"}:
        return render_world_frame(frame_bgr, output, **kwargs)
    if t in {"driver", "dmonitoring"}:
        return render_driver_frame(frame_bgr, output, **kwargs)
    return render_planner_frame(frame_bgr, output, **kwargs)
