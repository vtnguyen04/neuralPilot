"""Self-driving camera calibration, virtual MEDMODEL warp, and 3D->2D road projection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

IDX_N = 33
T_IDXS = np.array([10.0 * ((i / 32.0) ** 2) for i in range(IDX_N)], dtype=np.float32)
X_IDXS = np.array([192.0 * ((i / 32.0) ** 2) for i in range(IDX_N)], dtype=np.float32)

# Openpilot calib (+X fwd, +Y right, +Z down) <-> OpenCV view (+X right, +Y down, +Z fwd)
DEVICE_FROM_VIEW = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
VIEW_FROM_DEVICE = DEVICE_FROM_VIEW.T

COMMA2_W, COMMA2_H = 1164, 874
K_COMMA2 = np.array(
    [[910.0, 0.0, COMMA2_W / 2.0], [0.0, 910.0, COMMA2_H / 2.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

COMMA3_W, COMMA3_H = 1928, 1208
K_COMMA3_NARROW = np.array(
    [[2648.0, 0.0, COMMA3_W / 2.0], [0.0, 2648.0, COMMA3_H / 2.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
K_COMMA3_WIDE = np.array(
    [[567.0, 0.0, COMMA3_W / 2.0], [0.0, 567.0, COMMA3_H / 2.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

MEDMODEL_W, MEDMODEL_H = 512, 256
MEDMODEL_CY = 47.6
K_MED = np.array(
    [[910.0, 0.0, 256.0], [0.0, 910.0, MEDMODEL_CY], [0.0, 0.0, 1.0]], dtype=np.float64
)
K_SBIG = np.array(
    [
        [455.0, 0.0, 256.0],
        [0.0, 455.0, 0.5 * (256.0 + MEDMODEL_CY)],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CALIB_FROM_MED = np.linalg.inv(K_MED @ VIEW_FROM_DEVICE)
CALIB_FROM_SBIG = np.linalg.inv(K_SBIG @ VIEW_FROM_DEVICE)


def rot_from_euler(rpy: Sequence[float]) -> np.ndarray:
    """Compute 3x3 rotation matrix from openpilot XYZ intrinsic Euler angles (radians)."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()


def camera_intrinsics_from_fov(
    width: int,
    height: int,
    hfov_deg: float = 70.0,
) -> np.ndarray:
    """Construct pinhole intrinsics `K` for any camera given resolution and horizontal FOV."""
    fx = 0.5 * float(width) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def get_virtual_warp_matrix(
    device_from_calib_euler: Sequence[float],
    intrinsics: np.ndarray,
    wide_frame: bool = False,
) -> np.ndarray:
    """Compute homography mapping canonical `MEDMODEL`/`SBIGMODEL` pixels back to raw camera."""
    calib_from_model = CALIB_FROM_SBIG if wide_frame else CALIB_FROM_MED
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ VIEW_FROM_DEVICE @ device_from_calib
    return (camera_from_calib @ calib_from_model).astype(np.float32)


def warp_to_model_view(bgr: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    """Warp a raw camera BGR frame into canonical `512x256` openpilot virtual camera view."""
    return cv2.warpPerspective(
        bgr,
        warp_matrix,
        (MEDMODEL_W, MEDMODEL_H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )


def project_calib_to_image(
    pts_calib_xyz: np.ndarray,
    rpy: Sequence[float],
    intrinsics: np.ndarray,
    scale_xy: Tuple[float, float] = (1.0, 1.0),
    offset_y: float = 0.0,
    min_depth_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points in openpilot calib frame `(+X fwd, +Y right, +Z down)` onto image px."""
    pts_view = (VIEW_FROM_DEVICE @ rot_from_euler(rpy) @ pts_calib_xyz.T).T
    valid = pts_view[:, 2] > min_depth_m
    uv = np.full((len(pts_calib_xyz), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        proj = (intrinsics @ pts_view[valid].T).T
        u_raw = proj[:, 0] / proj[:, 2]
        v_raw = proj[:, 1] / proj[:, 2]
        uv[valid, 0] = u_raw * scale_xy[0]
        uv[valid, 1] = (v_raw - offset_y) * scale_xy[1]
    return uv, valid


def load_comma_localizer(seg_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load `localizer.safetensors` from a comma5k/openpilot segment directory."""
    from safetensors import safe_open

    with safe_open(str(Path(seg_dir) / "localizer.safetensors"), framework="numpy") as handle:
        fs = handle.get_tensor("frame_states")
        ft = handle.get_tensor("frame_t")
        rpy = handle.get_tensor("rpy")
        wfe = handle.get_tensor("wide_from_device_euler")
    return fs, ft, rpy, wfe


def compute_gt_ego_trajectory(
    frame_states: np.ndarray,
    frame_t: np.ndarray,
    rpy: Sequence[float],
    frame_idx: int,
    horizons_s: Sequence[float],
) -> Tuple[np.ndarray, float]:
    """Compute ground-truth future ego trajectory `(H, 3)` (+X fwd, +Y left, +Z up) & speed."""
    p0 = frame_states[frame_idx, 0:3]
    q0 = frame_states[frame_idx, 3:7]
    r_ecef_from_dev = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    r_ecef_from_calib = r_ecef_from_dev @ rot_from_euler(rpy)

    def _to_ego(p_ecef: np.ndarray) -> np.ndarray:
        d_calib = r_ecef_from_calib.T @ (p_ecef - p0)
        return np.array([d_calib[0], -d_calib[1], -d_calib[2]], dtype=np.float64)

    t0 = float(frame_t[frame_idx])
    traj = np.array(
        [
            _to_ego(frame_states[int(np.argmin(np.abs(frame_t - (t0 + h)))), 0:3])
            for h in horizons_s
        ],
        dtype=np.float32,
    )
    i_prev = max(0, frame_idx - 2)
    dt = max(float(frame_t[frame_idx] - frame_t[i_prev]), 1e-3)
    v_ego = float((-_to_ego(frame_states[i_prev, 0:3])[0]) / dt)
    return traj, v_ego


def resolve_camera_intrinsics(
    width: int,
    height: int,
    intrinsics: Optional[np.ndarray] = None,
    hfov_deg: float = 70.0,
    wide: bool = False,
) -> np.ndarray:
    """Return explicit intrinsics, comma3X exact matrix for 1928x1208, or FOV-derived pinhole K."""
    if intrinsics is not None:
        return np.asarray(intrinsics, dtype=np.float64)
    if width == COMMA3_W and height == COMMA3_H:
        return K_COMMA3_WIDE if wide else K_COMMA3_NARROW
    if width == COMMA2_W and height == COMMA2_H:
        return K_COMMA2
    return camera_intrinsics_from_fov(width, height, hfov_deg=hfov_deg)
