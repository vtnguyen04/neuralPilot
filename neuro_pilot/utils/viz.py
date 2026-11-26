"""Cinema-grade, zero-text 3D trajectory & waypoint visualizer for autonomous driving."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# Depth-intensifying studio gradient: crystal aqua near -> electric sapphire -> deep royal violet.
AZURE_COBALT_STOPS_RGB = np.array(
    [
        [103, 232, 249],  # t = 0.00 (#67e8f9) crystal icy aqua (light & airy near hood)
        [34, 211, 238],   # t = 0.22 (#22d3ee) vivid electric cyan
        [14, 165, 233],   # t = 0.45 (#0ea5e9) luminous ocean azure
        [37, 99, 235],    # t = 0.72 (#2563eb) rich deep electric cobalt
        [79, 70, 229],    # t = 0.88 (#4f46e5) deep sapphire indigo
        [109, 40, 217],   # t = 1.00 (#6d28d9) rich royal violet at far horizon
    ],
    dtype=np.float32,
)
AZURE_COBALT_STOP_T = np.linspace(0.0, 1.0, len(AZURE_COBALT_STOPS_RGB), dtype=np.float32)

# Backward-compatible aliases
ORANGE_RUST_STOPS_RGB = AZURE_COBALT_STOPS_RGB
ORANGE_RUST_STOP_T = AZURE_COBALT_STOP_T

DEFAULT_WAYPOINT_TIMES: Tuple[float, ...] = (
    0.30,
    0.65,
    1.10,
    1.65,
    2.30,
    3.10,
    4.00,
)


def sample_gradient_bgr(
    t: float,
    stops_rgb: np.ndarray = AZURE_COBALT_STOPS_RGB,
    stop_t: np.ndarray = AZURE_COBALT_STOP_T,
) -> Tuple[int, int, int]:
    """Interpolate a smooth BGR color along `[0, 1]` from the depth-intensifying gradient."""
    tc = float(np.clip(t, 0.0, 1.0))
    r = float(np.interp(tc, stop_t, stops_rgb[:, 0]))
    g = float(np.interp(tc, stop_t, stops_rgb[:, 1]))
    b = float(np.interp(tc, stop_t, stops_rgb[:, 2]))
    return (int(round(b)), int(round(g)), int(round(r)))


def _perceptual_t(t_norm: float) -> float:
    """Stretch near/mid perspective foreshortening so color deepens steadily toward the horizon."""
    return float(np.clip(t_norm, 0.0, 1.0) ** 0.65)


def _interpolate_dense_plan(
    t_idxs: np.ndarray,
    plan_xyz: np.ndarray,
    horizon_s: float = 4.5,
    num_samples: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a smooth cubic spline over the 3D plan and sample dense points up to `horizon_s`."""
    t_max = min(float(t_idxs[-1]), float(horizon_s))
    mask = t_idxs <= min(float(t_idxs[-1]), horizon_s + 1.5)
    t_fit = t_idxs[mask].astype(np.float64)
    p_fit = plan_xyz[mask].astype(np.float64)
    t_dense = np.linspace(float(t_fit[0]), t_max, num_samples, dtype=np.float64)
    if len(t_fit) >= 4:
        cs_x = CubicSpline(t_fit, p_fit[:, 0], bc_type="natural")
        cs_y = CubicSpline(t_fit, p_fit[:, 1], bc_type="natural")
        cs_z = CubicSpline(t_fit, p_fit[:, 2], bc_type="natural")
        dense_xyz = np.stack([cs_x(t_dense), cs_y(t_dense), cs_z(t_dense)], axis=1)
    else:
        dense_xyz = np.stack(
            [np.interp(t_dense, t_fit, p_fit[:, k]) for k in range(3)],
            axis=1,
        )
    return t_dense, dense_xyz


def _alpha_composite(dst_bgr: np.ndarray, layer_bgr: np.ndarray, alpha_map: np.ndarray) -> None:
    """In-place float alpha blend using a single-channel `[0, 1]` alpha mask."""
    a = alpha_map[..., None]
    blended = dst_bgr.astype(np.float32) * (1.0 - a) + layer_bgr.astype(np.float32) * a
    np.clip(blended, 0.0, 255.0, out=blended)
    dst_bgr[:] = blended.astype(np.uint8)


def _draw_ribbon_layer(
    canvas_w: int,
    canvas_h: int,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    valid: np.ndarray,
    t_norm: np.ndarray,
    near_alpha: float = 0.16,
    far_alpha: float = 0.56,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a crystal-clear 3D road ribbon that deepens in opacity & richness with distance."""
    color_buf = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    alpha_buf = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    n = len(t_norm)
    for i in range(n - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        quad = np.array(
            [pts_left[i], pts_right[i], pts_right[i + 1], pts_left[i + 1]],
            dtype=np.int32,
        )
        tm = 0.5 * (t_norm[i] + t_norm[i + 1])
        pt = _perceptual_t(tm)
        col = sample_gradient_bgr(pt)
        fade_near = min(1.0, i / 4.0) if i < 4 else 1.0
        # Ultra-clear near hood, progressively richer & deeper into distance, silky tip feather
        tip_soft = 1.0 if tm < 0.90 else float(1.0 - 0.35 * ((tm - 0.90) / 0.10))
        depth_a = near_alpha + (far_alpha - near_alpha) * (pt**0.85)
        seg_alpha = float(depth_a * fade_near * tip_soft)
        cv2.fillConvexPoly(color_buf, quad, col, cv2.LINE_AA)
        cv2.fillConvexPoly(alpha_buf, quad, seg_alpha, cv2.LINE_AA)

    alpha_buf = cv2.GaussianBlur(alpha_buf, (0, 0), sigmaX=3.2, sigmaY=2.4)
    color_buf = cv2.GaussianBlur(color_buf, (0, 0), sigmaX=1.6, sigmaY=1.2)
    return color_buf, alpha_buf


def render_modern_trajectory(
    frame_bgr: np.ndarray,
    plan_road_xyz: np.ndarray,
    t_idxs: np.ndarray,
    project_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    horizon_s: float = 4.5,
    waypoint_times: Sequence[float] = DEFAULT_WAYPOINT_TIMES,
    half_width_near: float = 1.08,
    half_width_far: float = 0.68,
    supersample: int = 2,
    min_forward_m: float = 2.6,
) -> np.ndarray:
    """Render a crystal-clear, wide, depth-intensifying 2x supersampled 3D drivable corridor."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    ss = max(1, int(supersample))
    sw, sh = w * ss, h * ss

    t_dense, dense_xyz = _interpolate_dense_plan(
        t_idxs, plan_road_xyz, horizon_s=horizon_s, num_samples=120
    )
    t_norm = np.clip(t_dense / max(horizon_s, 1e-3), 0.0, 1.0)

    fwd_ok = dense_xyz[:, 0] >= min_forward_m
    if int(fwd_ok.sum()) < 4:
        return out

    tangents = np.gradient(dense_xyz[:, :2], axis=0)
    tnorm = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    tangents = tangents / tnorm
    normals_right = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    widths = half_width_near * (1.0 - t_norm) + half_width_far * t_norm

    left_xyz = dense_xyz.copy()
    right_xyz = dense_xyz.copy()
    left_xyz[:, :2] -= normals_right * widths[:, None]
    right_xyz[:, :2] += normals_right * widths[:, None]

    uv_c, ok_c = project_fn(dense_xyz)
    uv_l, ok_l = project_fn(left_xyz)
    uv_r, ok_r = project_fn(right_xyz)
    valid = fwd_ok & ok_c & ok_l & ok_r

    if int(valid.sum()) < 3:
        return out

    # 1) Crystal-clear near -> deep rich sapphire/violet far volumetric 3D corridor
    ribbon_col, ribbon_alpha = _draw_ribbon_layer(
        w, h, uv_l, uv_r, valid, t_norm, near_alpha=0.16, far_alpha=0.56
    )
    _alpha_composite(out, ribbon_col, ribbon_alpha)

    # 2) Crisp 2x supersampled corridor borders, continuous smooth centerline, and glossy nodes
    fg_col = np.zeros((sh, sw, 3), dtype=np.uint8)
    fg_alpha = np.zeros((sh, sw), dtype=np.float32)
    shadow_alpha = np.zeros((sh, sw), dtype=np.float32)

    uv_c_ss = np.round(np.nan_to_num(uv_c, nan=0.0) * ss).astype(np.int32)
    uv_l_ss = np.round(np.nan_to_num(uv_l, nan=0.0) * ss).astype(np.int32)
    uv_r_ss = np.round(np.nan_to_num(uv_r, nan=0.0) * ss).astype(np.int32)

    # Luminous depth-intensifying corridor edge rails (icy-cyan near -> rich cobalt/violet far)
    for i in range(len(t_norm) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        tm = 0.5 * (t_norm[i] + t_norm[i + 1])
        pt = _perceptual_t(tm)
        col = sample_gradient_bgr(pt)
        rail_a = float(0.42 + 0.44 * pt)
        rail_th = max(1, int(round((2.2 - 0.5 * tm) * ss)))
        cv2.line(fg_col, tuple(uv_l_ss[i]), tuple(uv_l_ss[i + 1]), col, rail_th, cv2.LINE_AA)
        cv2.line(fg_alpha, tuple(uv_l_ss[i]), tuple(uv_l_ss[i + 1]), rail_a, rail_th, cv2.LINE_AA)
        cv2.line(fg_col, tuple(uv_r_ss[i]), tuple(uv_r_ss[i + 1]), col, rail_th, cv2.LINE_AA)
        cv2.line(fg_alpha, tuple(uv_r_ss[i]), tuple(uv_r_ss[i + 1]), rail_a, rail_th, cv2.LINE_AA)

    # Silky continuous hero centerline (zero white dotted stitching, deepening along horizon)
    for i in range(len(t_norm) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        tm = 0.5 * (t_norm[i] + t_norm[i + 1])
        pt = _perceptual_t(tm)
        col = sample_gradient_bgr(pt)
        line_th = max(2, int(round((4.8 - 2.0 * tm) * ss)))
        halo_th = line_th + int(round(3.0 * ss))
        sh_a = float(0.28 + 0.14 * pt)
        fg_a = float(0.86 + 0.12 * pt)
        cv2.line(shadow_alpha, tuple(uv_c_ss[i]), tuple(uv_c_ss[i + 1]), sh_a, halo_th, cv2.LINE_AA)
        cv2.line(fg_col, tuple(uv_c_ss[i]), tuple(uv_c_ss[i + 1]), col, line_th, cv2.LINE_AA)
        cv2.line(fg_alpha, tuple(uv_c_ss[i]), tuple(uv_c_ss[i + 1]), fg_a, line_th, cv2.LINE_AA)

    # Perspective-scaled glossy waypoint nodes matching the exact depth-intensifying palette
    wp_t_arr = np.array([t for t in waypoint_times if t <= horizon_s], dtype=np.float64)
    if len(wp_t_arr) > 0:
        wp_xyz = np.stack(
            [np.interp(wp_t_arr, t_dense, dense_xyz[:, k]) for k in range(3)],
            axis=1,
        )
        wp_uv, wp_ok = project_fn(wp_xyz)
        placed: list[Tuple[float, float, float, float]] = []
        for idx in range(len(wp_t_arr)):
            depth = float(wp_xyz[idx, 0])
            if not wp_ok[idx] or depth < min_forward_m:
                continue
            ux, uy = float(wp_uv[idx, 0]), float(wp_uv[idx, 1])
            if not (0.0 <= ux < w and 0.0 <= uy < h):
                continue
            rad_px = float(np.clip(64.0 / max(depth, 4.0), 3.4, 7.8))
            if placed:
                px_prev, py_prev, r_prev, _ = placed[-1]
                if np.hypot(ux - px_prev, uy - py_prev) < (rad_px + r_prev + 6.5):
                    continue
            tn = float(np.clip(wp_t_arr[idx] / max(horizon_s, 1e-3), 0.0, 1.0))
            placed.append((ux, uy, rad_px, tn))

        for ux, uy, rad_px, tn in reversed(placed):
            cx_px = int(round(ux * ss))
            cy_px = int(round(uy * ss))
            rad = max(2, int(round(rad_px * ss)))
            rim_th = max(1, int(round(1.6 * ss)))
            col = sample_gradient_bgr(_perceptual_t(tn))
            cv2.circle(
                shadow_alpha, (cx_px, cy_px), rad + int(round(2.8 * ss)), 0.38, -1, cv2.LINE_AA
            )
            cv2.circle(fg_col, (cx_px, cy_px), rad, col, -1, cv2.LINE_AA)
            cv2.circle(fg_alpha, (cx_px, cy_px), rad, 0.96, -1, cv2.LINE_AA)
            cv2.circle(fg_col, (cx_px, cy_px), rad, (245, 250, 255), rim_th, cv2.LINE_AA)
            cv2.circle(fg_alpha, (cx_px, cy_px), rad, 0.92, rim_th, cv2.LINE_AA)

    if ss > 1:
        fg_col_ds = cv2.resize(fg_col, (w, h), interpolation=cv2.INTER_AREA)
        fg_alpha_ds = cv2.resize(fg_alpha, (w, h), interpolation=cv2.INTER_AREA)
        shadow_ds = cv2.resize(shadow_alpha, (w, h), interpolation=cv2.INTER_AREA)
    else:
        fg_col_ds, fg_alpha_ds, shadow_ds = fg_col, fg_alpha, shadow_alpha

    shadow_ds = cv2.GaussianBlur(shadow_ds, (0, 0), sigmaX=2.0, sigmaY=2.0)
    dark_layer = np.full((h, w, 3), (10, 12, 18), dtype=np.uint8)
    _alpha_composite(out, dark_layer, shadow_ds)
    _alpha_composite(out, fg_col_ds, fg_alpha_ds)
    return out


def compose_side_by_side(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    divider_px: int = 4,
    divider_bgr: Tuple[int, int, int] = (14, 16, 22),
) -> np.ndarray:
    """Concatenate two equal-height panels with a minimal dark seam (zero text)."""
    h = left_bgr.shape[0]
    if right_bgr.shape[0] != h:
        scale = h / max(right_bgr.shape[0], 1)
        right_bgr = cv2.resize(right_bgr, (int(round(right_bgr.shape[1] * scale)), h))
    seam = np.full((h, divider_px, 3), divider_bgr, dtype=np.uint8)
    return np.hstack([left_bgr, seam, right_bgr])
