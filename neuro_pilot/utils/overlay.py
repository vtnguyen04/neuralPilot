"""Modern 2026 zero-text perception, segmentation, multi-lane, and depth visualizers."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from neuro_pilot.utils.viz import sample_gradient_bgr

# Rich 2026 Studio Jewel-Tone Perception Palette (BGR — zero plain flat white)
_BOX_CLASS_BGR = {
    0: (248, 189, 56),   # Traffic vehicle: vivid electric sky-cyan (#38bdf8)
    1: (153, 211, 52),   # CIPO lead vehicle: luminous emerald-mint (#34d399)
    2: (94, 63, 244),    # Vulnerable road user: vivid rose-magenta (#f43f5e)
    3: (36, 191, 251),   # Heavy / special vehicle: luminous amber-gold (#fbbf24)
}
_SHADOW_BGR = (14, 16, 22)

# Multi-lane 3-tier jewel gradients (RGB stops -> interpolated along distance for rich depth)
_EGO_LANE_RGB = np.array(
    [[34, 211, 238], [14, 165, 233], [37, 99, 235]], dtype=np.float32
)  # Electric cyan -> ocean azure -> deep cobalt
_ADJ_LANE_RGB = np.array(
    [[216, 180, 254], [192, 132, 252], [124, 58, 237]], dtype=np.float32
)  # Luminous orchid -> amethyst -> rich royal purple
_CURB_EDGE_RGB = np.array(
    [[253, 186, 116], [251, 146, 60], [244, 63, 94]], dtype=np.float32
)  # Warm sunset gold -> coral -> crimson-rose
_LANE_STOP_T = np.array([0.0, 0.5, 1.0], dtype=np.float32)

# Cinema complementary 19-class Cityscapes palette (deep sapphire road vs warm coral/amber vehicles)
_CITYSCAPES_RGB = np.array(
    [
        [37, 99, 235],    # 0:  road (deep sapphire azure)
        [139, 92, 246],   # 1:  sidewalk (studio violet)
        [51, 65, 85],     # 2:  building (recessive midnight slate)
        [71, 85, 105],    # 3:  wall
        [100, 116, 139],  # 4:  fence
        [148, 163, 184],  # 5:  pole
        [250, 204, 21],   # 6:  traffic light (luminous gold)
        [251, 191, 36],   # 7:  traffic sign (warm amber)
        [17, 94, 89],     # 8:  vegetation (subtle deep pine)
        [19, 78, 74],     # 9:  terrain
        [15, 23, 42],     # 10: sky (transparent)
        [244, 63, 94],    # 11: person (electric rose)
        [236, 72, 153],   # 12: rider (vivid magenta)
        [249, 115, 22],   # 13: car (luminous sunset coral-orange)
        [251, 146, 60],   # 14: truck (warm amber-coral)
        [245, 158, 11],   # 15: bus (rich golden amber)
        [168, 85, 247],   # 16: train
        [232, 121, 249],  # 17: motorcycle (electric fuchsia)
        [251, 113, 133],  # 18: bicycle (bright coral-rose)
    ],
    dtype=np.uint8,
)
CITYSCAPES_BGR = _CITYSCAPES_RGB[:, ::-1].copy()
_CITYSCAPES_ALPHA = np.array(
    [
        0.38, 0.34, 0.12, 0.16, 0.18, 0.34, 0.65, 0.60, 0.16, 0.16,
        0.00, 0.62, 0.62, 0.50, 0.50, 0.50, 0.48, 0.58, 0.58,
    ],
    dtype=np.float32,
)


def _draw_corner_brackets(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    frac: float = 0.25,
) -> None:
    """Draw razor-sharp AR corner brackets with dark contrast hairline."""
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    clen = max(6, min(20, int(round(min(bw, bh) * frac))))
    corners = (
        ((x1, y1), (x1 + clen, y1), (x1, y1 + clen)),
        ((x2, y1), (x2 - clen, y1), (x2, y1 + clen)),
        ((x1, y2), (x1 + clen, y2), (x1, y2 - clen)),
        ((x2, y2), (x2 - clen, y2), (x2, y2 - clen)),
    )
    for p, p_h, p_v in corners:
        cv2.line(img, p, p_h, _SHADOW_BGR, thickness + 2, cv2.LINE_AA)
        cv2.line(img, p, p_v, _SHADOW_BGR, thickness + 2, cv2.LINE_AA)
    for p, p_h, p_v in corners:
        cv2.line(img, p, p_h, color, thickness, cv2.LINE_AA)
        cv2.line(img, p, p_v, color, thickness, cv2.LINE_AA)


def render_perception_boxes(
    frame_bgr: np.ndarray,
    boxes: Sequence[Tuple[float, float, float, float, float, int]],
    src_wh: Tuple[int, int] = (1024, 512),
) -> np.ndarray:
    """Render sleek zero-text jewel-tone AR corner brackets for traffic and CIPO."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    sx, sy = w / float(src_wh[0]), h / float(src_wh[1])

    overlay = out.copy()
    valid_boxes = []
    for x1, y1, x2, y2, _score, cid in boxes:
        ix1 = int(np.clip(round(x1 * sx), 0, w - 1))
        iy1 = int(np.clip(round(y1 * sy), 0, h - 1))
        ix2 = int(np.clip(round(x2 * sx), 0, w - 1))
        iy2 = int(np.clip(round(y2 * sy), 0, h - 1))
        bw, bh = ix2 - ix1, iy2 - iy1
        if bw < 22 or bh < 20:
            continue
        is_cipo = cid == 1
        col = _BOX_CLASS_BGR.get(int(cid), _BOX_CLASS_BGR[0])
        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), col, -1)
        valid_boxes.append((ix1, iy1, ix2, iy2, is_cipo, col, min(bw, bh)))

    if valid_boxes:
        cv2.addWeighted(overlay, 0.13, out, 0.87, 0.0, dst=out)
    for ix1, iy1, ix2, iy2, is_cipo, col, mdim in valid_boxes:
        th = 3 if is_cipo else (2 if mdim >= 38 else 1)
        _draw_corner_brackets(out, ix1, iy1, ix2, iy2, col, thickness=th)
        if is_cipo:
            cx, tick_w = (ix1 + ix2) // 2, max(12, (ix2 - ix1) // 3)
            y_bar = min(h - 3, iy2 + 6)
            cv2.line(out, (cx - tick_w, y_bar), (cx + tick_w, y_bar), _SHADOW_BGR, 5, cv2.LINE_AA)
            cv2.line(out, (cx - tick_w, y_bar), (cx + tick_w, y_bar), col, 3, cv2.LINE_AA)
    return out


def render_seg_overlay(
    frame_bgr: np.ndarray,
    class_map: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Render smooth cinema-palette 19-class semantic segmentation with luminous contours."""
    h, w = frame_bgr.shape[:2]
    cls_up = cv2.resize(class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    cls_safe = np.clip(cls_up, 0, len(CITYSCAPES_BGR) - 1)
    color_mask = CITYSCAPES_BGR[cls_safe]
    a3 = np.clip(_CITYSCAPES_ALPHA[cls_safe] * float(alpha), 0.0, 0.68)[..., None]
    out = frame_bgr.astype(np.float32) * (1.0 - a3) + color_mask.astype(np.float32) * a3

    # Crisp luminous edge only along dynamic objects & drivable boundaries
    fg = np.isin(cls_safe, (0, 1, 6, 7, 11, 12, 13, 14, 15, 17, 18))
    edges = np.zeros((h, w), dtype=np.uint8)
    edges[1:, :] |= ((cls_safe[1:, :] != cls_safe[:-1, :]) & (fg[1:, :] | fg[:-1, :])).astype(
        np.uint8
    )
    edges[:, 1:] |= ((cls_safe[:, 1:] != cls_safe[:, :-1]) & (fg[:, 1:] | fg[:, :-1])).astype(
        np.uint8
    )
    edge_a = (cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 0.6) * 0.34)[..., None]
    tint = np.array([255.0, 240.0, 220.0], dtype=np.float32)
    return np.clip(out * (1.0 - edge_a) + tint * edge_a, 0, 255).astype(np.uint8)


def render_depth_panel(
    depth_map: np.ndarray,
    out_wh: Tuple[int, int],
    colormap: int = cv2.COLORMAP_MAGMA,
) -> np.ndarray:
    """Render a refined full-contrast Magma monocular 3D depth panel (zero text)."""
    lo, hi = np.percentile(depth_map, [2, 98])
    norm = np.clip((depth_map - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) ** 0.92
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), colormap)
    return cv2.resize(vis, out_wh, interpolation=cv2.INTER_LINEAR)


def render_proximity_gauge(
    frame_bgr: np.ndarray,
    distance_m: float,
    max_dist_m: float = 80.0,
) -> np.ndarray:
    """Draw a minimal text-free frosted longitudinal CIPO proximity bar."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bar_w, bar_h = int(w * 0.26), 4
    x0, y0 = (w - bar_w) // 2, h - 16
    frac = float(np.clip(1.0 - (distance_m / max_dist_m), 0.06, 1.0))
    fill_w = int(round(bar_w * frac))

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y0 + bar_h), (20, 24, 34), -1)
    cv2.addWeighted(overlay, 0.58, out, 0.42, 0.0, dst=out)
    col = sample_gradient_bgr(frac)
    cv2.rectangle(out, (x0, y0), (x0 + fill_w, y0 + bar_h), col, -1)
    return out


def _sample_curve_3d(
    x_idxs: np.ndarray,
    yz: np.ndarray,
    min_x: float = 2.5,
    max_x: float = 48.0,
    num_samples: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a smooth 3D road curve `[N, 3]` (`+X fwd, +Y right, +Z down`)."""
    mask = x_idxs <= min(float(x_idxs[-1]), max_x + 15.0)
    xf = x_idxs[mask].astype(np.float64)
    yf, zf = yz[mask, 0].astype(np.float64), yz[mask, 1].astype(np.float64)
    xd = np.linspace(min_x, min(max_x, float(xf[-1])), num_samples, dtype=np.float64)
    if len(xf) >= 4:
        yd = CubicSpline(xf, yf, bc_type="natural")(xd)
        zd = CubicSpline(xf, zf, bc_type="natural")(xd)
    else:
        yd, zd = np.interp(xd, xf, yf), np.interp(xd, xf, zf)
    t_norm = np.clip((xd - min_x) / max(max_x - min_x, 1e-3), 0.0, 1.0)
    return np.stack([xd, yd, zd], axis=1), t_norm


def render_multilane_3d(
    frame_bgr: np.ndarray,
    lane_lines: np.ndarray,
    lane_probs: np.ndarray,
    road_edges: np.ndarray,
    x_idxs: np.ndarray,
    project_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    *,
    lane_stds: np.ndarray | None = None,
    max_dist_m: float = 48.0,
    lane_prob_thres: float = 0.28,
    fill_ego_lane: bool = True,
) -> np.ndarray:
    """Render 2x supersampled 3-tier jewel-tone 3D multi-lane (cyan ego, violet adj, coral curb)."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    ss = 2
    sw, sh = w * ss, h * ss

    def _effective_max_x(li: int) -> float:
        if lane_stds is None:
            return max_dist_m
        y_std = lane_stds[li, :, 0]
        high_u = np.where((x_idxs >= 8.0) & (y_std > 0.75))[0]
        return float(min(max_dist_m, x_idxs[high_u[0]])) if len(high_u) > 0 else max_dist_m

    # 1) Glass-clear ego-lane carpet deepening from light aqua near to rich sapphire far
    if fill_ego_lane and float(lane_probs[1]) >= 0.35 and float(lane_probs[2]) >= 0.35:
        ego_x = min(_effective_max_x(1), _effective_max_x(2))
        pts_l, _ = _sample_curve_3d(x_idxs, lane_lines[1], max_x=ego_x)
        pts_r, t_n = _sample_curve_3d(x_idxs, lane_lines[2], max_x=ego_x)
        uv_l, ok_l = project_fn(pts_l)
        uv_r, ok_r = project_fn(pts_r)
        ok = ok_l & ok_r
        fill_buf = np.zeros_like(out)
        fill_a = np.zeros((h, w), dtype=np.float32)
        for i in range(len(t_n) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            tm = float(0.5 * (t_n[i] + t_n[i + 1]))
            col = sample_gradient_bgr(tm**0.7, _EGO_LANE_RGB, _LANE_STOP_T)
            quad = np.array([uv_l[i], uv_r[i], uv_r[i + 1], uv_l[i + 1]], dtype=np.int32)
            a_seg = float(0.08 + 0.18 * (tm**0.75)) * (1.0 if tm < 0.88 else 0.7)
            cv2.fillConvexPoly(fill_buf, quad, col, cv2.LINE_AA)
            cv2.fillConvexPoly(fill_a, quad, a_seg, cv2.LINE_AA)
        fill_a = cv2.GaussianBlur(fill_a, (0, 0), sigmaX=3.5, sigmaY=2.5)[..., None]
        out = np.clip(out * (1.0 - fill_a) + fill_buf * fill_a, 0, 255).astype(np.uint8)

    # 2) 2x supersampled crisp jewel-tone lane curves with subtle dark backing & neon aura
    shadow_a = np.zeros((sh, sw), dtype=np.float32)
    aura_col = np.zeros((sh, sw, 3), dtype=np.uint8)
    aura_a = np.zeros((sh, sw), dtype=np.float32)
    core_col = np.zeros((sh, sw, 3), dtype=np.uint8)
    core_a = np.zeros((sh, sw), dtype=np.float32)

    def _draw_curve(
        yz: np.ndarray,
        stops_rgb: np.ndarray,
        base_alpha: float,
        base_thick: float,
        curve_max_x: float,
    ) -> None:
        if curve_max_x <= 4.0:
            return
        pts_3d, t_n = _sample_curve_3d(x_idxs, yz, max_x=curve_max_x)
        uv, ok = project_fn(pts_3d)
        uv_ss = np.round(np.nan_to_num(uv, nan=0.0) * ss).astype(np.int32)
        for i in range(len(t_n) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            p0, p1 = tuple(uv_ss[i]), tuple(uv_ss[i + 1])
            if not (-sw <= p0[0] <= 2 * sw and -sh <= p0[1] <= 2 * sh):
                continue
            tm = float(0.5 * (t_n[i] + t_n[i + 1]))
            col = sample_gradient_bgr(tm**0.7, stops_rgb, _LANE_STOP_T)
            th = max(2, int(round(base_thick * (1.0 - 0.38 * tm) * ss)))
            al = float(base_alpha * (0.85 + 0.15 * tm) * (1.0 - 0.25 * (tm**2)))
            cv2.line(shadow_a, p0, p1, al * 0.36, th + 3 * ss, cv2.LINE_AA)
            cv2.line(aura_col, p0, p1, col, th + 2 * ss, cv2.LINE_AA)
            cv2.line(aura_a, p0, p1, al * 0.45, th + 2 * ss, cv2.LINE_AA)
            cv2.line(core_col, p0, p1, col, th, cv2.LINE_AA)
            cv2.line(core_a, p0, p1, al, th, cv2.LINE_AA)

    for ri in range(min(2, len(road_edges))):
        _draw_curve(road_edges[ri], _CURB_EDGE_RGB, 0.72, 2.0, 36.0)

    for li in range(min(4, len(lane_lines))):
        prob = float(lane_probs[li])
        if prob < lane_prob_thres:
            continue
        is_ego = li in (1, 2)
        stops = _EGO_LANE_RGB if is_ego else _ADJ_LANE_RGB
        al = min(0.96, 0.56 + 0.42 * prob) if is_ego else min(0.84, 0.44 + 0.40 * prob)
        _draw_curve(lane_lines[li], stops, al, 3.0 if is_ego else 2.2, _effective_max_x(li))

    sh_ds = cv2.GaussianBlur(
        cv2.resize(shadow_a, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 2.0
    )[..., None]
    au_c = cv2.GaussianBlur(
        cv2.resize(aura_col, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 3.0
    )
    au_a = cv2.GaussianBlur(
        cv2.resize(aura_a, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 3.2
    )[..., None]
    co_c = cv2.resize(core_col, (w, h), interpolation=cv2.INTER_AREA)
    co_a = cv2.resize(core_a, (w, h), interpolation=cv2.INTER_AREA)[..., None]

    out = np.clip(out * (1.0 - sh_ds) + 12.0 * sh_ds, 0, 255).astype(np.uint8)
    out = np.clip(out * (1.0 - au_a) + au_c * au_a, 0, 255).astype(np.uint8)
    return np.clip(out * (1.0 - co_a) + co_c * co_a, 0, 255).astype(np.uint8)


def compose_grid_2x2(
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
    divider_px: int = 4,
    divider_bgr: Tuple[int, int, int] = (14, 16, 22),
) -> np.ndarray:
    """Compose four equal-sized frames into a crisp zero-text 2x2 showcase grid."""
    h, w = tl.shape[:2]
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA) for p in (tl, tr, bl, br)]
    v_seam = np.full((h, divider_px, 3), divider_bgr, dtype=np.uint8)
    top = np.hstack([panels[0], v_seam, panels[1]])
    bot = np.hstack([panels[2], v_seam, panels[3]])
    h_seam = np.full((divider_px, top.shape[1], 3), divider_bgr, dtype=np.uint8)
    return np.vstack([top, h_seam, bot])
