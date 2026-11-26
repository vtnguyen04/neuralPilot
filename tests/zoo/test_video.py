"""Tests verifying 100% standalone NeuroPilot camera geometry, trajectory viz, and video runner."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.camera import (
    COMMA3_H,
    COMMA3_W,
    K_COMMA3_NARROW,
    T_IDXS,
    camera_intrinsics_from_fov,
    get_virtual_warp_matrix,
    project_calib_to_image,
    resolve_camera_intrinsics,
    warp_to_model_view,
)
from neuro_pilot.utils.viz import (
    compose_side_by_side,
    render_modern_trajectory,
    sample_gradient_bgr,
)

PILOT_S_PT = Path(
    "/home/quynhthu/Documents/AI-project/Vinfast/onnx_reverse_engineering/"
    "converted_pt/planner/pilot_s.pt"
)


def test_camera_geom_and_zero_text_trajectory_viz():
    k_mat = camera_intrinsics_from_fov(640, 400, hfov_deg=70.0)
    assert k_mat.shape == (3, 3) and k_mat[0, 2] == 320.0
    assert np.allclose(resolve_camera_intrinsics(COMMA3_W, COMMA3_H), K_COMMA3_NARROW)

    warp = get_virtual_warp_matrix([0.0, 0.0, 0.0], k_mat, wide_frame=False)
    warped = warp_to_model_view(np.full((400, 640, 3), 80, dtype=np.uint8), warp)
    assert warped.shape == (256, 512, 3)

    # Smooth forward right-curving 3D plan on road plane (+Z down = 1.22m)
    x_pts = 10.0 * T_IDXS
    y_pts = 0.015 * (x_pts**2)
    z_pts = np.full_like(x_pts, 1.22)
    plan_road = np.stack([x_pts, y_pts, z_pts], axis=1)

    def _proj(pts: np.ndarray):
        return project_calib_to_image(pts, [0.0, 0.0, 0.0], k_mat)

    base = np.full((400, 640, 3), 40, dtype=np.uint8)
    rendered = render_modern_trajectory(base, plan_road, T_IDXS, _proj, horizon_s=4.5)
    assert rendered.shape == base.shape
    assert not np.array_equal(rendered, base)
    # Near color brighter peach-orange than far deep-rust color
    assert sum(sample_gradient_bgr(0.0)) > sum(sample_gradient_bgr(1.0))

    dual = compose_side_by_side(rendered, rendered, divider_px=4)
    assert dual.shape == (400, 640 * 2 + 4, 3)


@pytest.mark.skipif(not PILOT_S_PT.is_file(), reason="pilot_s.pt not present")
def test_standalone_predict_video_pure_pt(tmp_path: Path):
    """End-to-end pure-.pt `NeuroPilot.predict_video` with zero external repo imports."""
    vid_path = tmp_path / "synthetic_road.avi"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (320, 200))
    for i in range(6):
        frame = np.full((200, 320, 3), 60 + i * 5, dtype=np.uint8)
        cv2.line(frame, (110, 199), (150, 90), (230, 230, 230), 2)
        cv2.line(frame, (210, 199), (170, 90), (230, 230, 230), 2)
        writer.write(frame)
    writer.release()

    out_mp4 = tmp_path / "rendered.mp4"
    model = NeuroPilot(PILOT_S_PT, device="cpu")
    res = model.predict_video(
        vid_path,
        out_mp4=out_mp4,
        warmup_frames=1,
        max_frames=6,
        panel_size=(320, 200),
        use_fp16=False,
    )
    assert res["plans"].shape == (6, 33, 3)
    assert np.isfinite(res["plans"]).all()
    assert out_mp4.is_file() and out_mp4.stat().st_size > 1000
    assert res["render"]["rendered_frames"] == 5
