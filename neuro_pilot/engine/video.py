"""Streaming video trajectory inference & zero-text cinema rendering engine for NeuroPilot."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from neuro_pilot.utils.camera import (
    T_IDXS,
    compute_gt_ego_trajectory,
    get_virtual_warp_matrix,
    load_comma_localizer,
    project_calib_to_image,
    resolve_camera_intrinsics,
    warp_to_model_view,
)
from neuro_pilot.utils.viz import compose_side_by_side, render_modern_trajectory

PathLike = Union[str, Path]


def _resolve_stream_sources(
    source: PathLike,
    wide_video: Optional[PathLike] = None,
    calib_rpy: Optional[Sequence[float]] = None,
    wide_from_device_euler: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Resolve either a single video file or a comma5k dual-camera segment directory."""
    src_path = Path(source)
    if src_path.is_dir():
        fcam = src_path / "fcamera.hevc"
        ecam = src_path / "ecamera.hevc"
        loc_path = src_path / "localizer.safetensors"
        fs = ft = None
        rpy = np.zeros(3, dtype=np.float64)
        wfe = np.zeros(3, dtype=np.float64)
        if loc_path.is_file():
            fs, ft, rpy, wfe = load_comma_localizer(src_path)
        return {
            "name": src_path.name,
            "narrow_path": fcam,
            "wide_path": ecam if ecam.is_file() else None,
            "rpy": np.asarray(calib_rpy, dtype=np.float64) if calib_rpy is not None else rpy,
            "wfe": (
                np.asarray(wide_from_device_euler, dtype=np.float64)
                if wide_from_device_euler is not None
                else wfe
            ),
            "fs": fs,
            "ft": ft,
        }

    return {
        "name": src_path.stem,
        "narrow_path": src_path,
        "wide_path": Path(wide_video) if wide_video is not None else None,
        "rpy": np.asarray(
            calib_rpy if calib_rpy is not None else [0.0, 0.0, 0.0], dtype=np.float64
        ),
        "wfe": np.asarray(
            wide_from_device_euler if wide_from_device_euler is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "fs": None,
        "ft": None,
    }


def infer_driving_video(
    runner: Any,
    source: PathLike,
    *,
    wide_video: Optional[PathLike] = None,
    calib_rpy: Optional[Sequence[float]] = None,
    wide_from_device_euler: Optional[Sequence[float]] = None,
    intrinsics: Optional[np.ndarray] = None,
    wide_intrinsics: Optional[np.ndarray] = None,
    hfov_deg: float = 70.0,
    warmup_frames: int = 30,
    max_frames: Optional[int] = None,
    use_fp16: bool = True,
    return_extras: bool = False,
) -> Any:
    """Stream any video or dual-camera segment through a NeuroPilot policy and collect 3D plans."""
    meta = _resolve_stream_sources(source, wide_video, calib_rpy, wide_from_device_euler)
    cap_f = cv2.VideoCapture(str(meta["narrow_path"]))
    cap_e = cv2.VideoCapture(str(meta["wide_path"])) if meta["wide_path"] is not None else None
    if not cap_f.isOpened():
        raise FileNotFoundError(f"Unable to open video: {meta['narrow_path']}")

    if runner.predictor is not None and hasattr(runner.predictor, "reset_state"):
        runner.predictor.reset_state()

    action_t = np.zeros((1, 2), dtype=np.float32)
    m_narrow = m_wide = None
    k_nar_used: Optional[np.ndarray] = None
    src_wh: Tuple[int, int] = (0, 0)
    if meta["ft"] is not None and len(meta["ft"]) >= 2:
        dt_med = float(np.median(np.diff(meta["ft"])))
        fps_in = (1.0 / dt_med) if dt_med > 1e-4 else 20.0
    elif Path(meta["narrow_path"]).suffix.lower() == ".hevc":
        fps_in = 20.0
    else:
        fps_in = float(cap_f.get(cv2.CAP_PROP_FPS) or 20.0)
        if fps_in <= 0.0 or fps_in > 240.0:
            fps_in = 20.0

    plans: List[np.ndarray] = []
    lane_lines_list: List[np.ndarray] = []
    lane_stds_list: List[np.ndarray] = []
    lane_probs_list: List[np.ndarray] = []
    road_edges_list: List[np.ndarray] = []
    ades: List[float] = []
    fdes: List[float] = []
    latencies_ms: List[float] = []
    op_4s_mask = T_IDXS <= 4.1
    op_4s_t = T_IDXS[op_4s_mask]
    amp_cuda = bool(
        use_fp16 and torch.cuda.is_available() and str(runner.device).startswith("cuda")
    )

    idx = 0
    while max_frames is None or idx < max_frames:
        ok_f, fr_f = cap_f.read()
        if not ok_f:
            break
        if cap_e is not None:
            ok_e, fr_e = cap_e.read()
            if not ok_e:
                break
        else:
            fr_e = fr_f

        if m_narrow is None:
            nh, nw = fr_f.shape[:2]
            wh, ww = fr_e.shape[:2]
            src_wh = (nw, nh)
            k_nar_used = resolve_camera_intrinsics(
                nw, nh, intrinsics, hfov_deg=hfov_deg, wide=False
            )
            k_wid = resolve_camera_intrinsics(
                ww, wh, wide_intrinsics, hfov_deg=hfov_deg, wide=True
            )
            m_narrow = get_virtual_warp_matrix(meta["rpy"], k_nar_used, wide_frame=False)
            m_wide = get_virtual_warp_matrix(meta["rpy"] + meta["wfe"], k_wid, wide_frame=True)

        w_nar = warp_to_model_view(fr_f, m_narrow)
        w_wid = warp_to_model_view(fr_e, m_wide)

        t0 = time.perf_counter()
        with torch.inference_mode():
            if amp_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = runner.predict(w_nar, wide_img=w_wid, action_t=action_t)
            else:
                out = runner.predict(w_nar, wide_img=w_wid, action_t=action_t)
        if amp_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        plan_xyz = out["plan"][:, 0:3].astype(np.float32)
        action_t = out["action"][:2][None, :].astype(np.float32)
        plans.append(plan_xyz)
        if return_extras and "lane_lines" in out:
            lane_lines_list.append(out["lane_lines"].astype(np.float32))
            lane_stds_list.append(out["lane_lines_std"].astype(np.float32))
            prob_vis = np.maximum(out["lane_lines_prob"], out.get("lane_lines_prob_near", 0.0))
            lane_probs_list.append(prob_vis.astype(np.float32))
            road_edges_list.append(out["road_edges"].astype(np.float32))

        if meta["fs"] is not None and idx >= warmup_frames and idx + 82 < len(meta["ft"]):
            gt_4s, _ = compute_gt_ego_trajectory(meta["fs"], meta["ft"], meta["rpy"], idx, op_4s_t)
            pred_ego = np.stack([plan_xyz[:, 0], -plan_xyz[:, 1], -plan_xyz[:, 2]], axis=1)
            err = np.linalg.norm(pred_ego[op_4s_mask, :2] - gt_4s[:, :2], axis=1)
            ades.append(float(np.mean(err)))
            fdes.append(float(err[-1]))

        idx += 1
        if idx % 200 == 0:
            avg_ms = float(np.mean(latencies_ms[-200:]))
            print(f"  [{meta['name'][:12]}] inferred {idx} frames ({avg_ms:.1f} ms/f)")

    cap_f.release()
    if cap_e is not None:
        cap_e.release()

    if not plans:
        raise RuntimeError(f"No frames decoded from {meta['narrow_path']}")

    mean_ms = float(np.mean(latencies_ms))
    info = {
        "name": meta["name"],
        "narrow_path": str(meta["narrow_path"]),
        "frames": len(plans),
        "width": src_wh[0],
        "height": src_wh[1],
        "fps": fps_in,
        "rpy": meta["rpy"].tolist(),
        "intrinsics": k_nar_used.tolist() if k_nar_used is not None else None,
        "mean_latency_ms": round(mean_ms, 2),
        "infer_fps": round(1000.0 / max(mean_ms, 1e-3), 1),
        "ade_4s_m": round(float(np.mean(ades)), 3) if ades else None,
        "fde_4s_m": round(float(np.mean(fdes)), 3) if fdes else None,
    }
    plans_arr = np.stack(plans, axis=0)
    if return_extras:
        extras = {
            "lane_lines": np.stack(lane_lines_list, axis=0) if lane_lines_list else None,
            "lane_stds": np.stack(lane_stds_list, axis=0) if lane_stds_list else None,
            "lane_probs": np.stack(lane_probs_list, axis=0) if lane_probs_list else None,
            "road_edges": np.stack(road_edges_list, axis=0) if road_edges_list else None,
        }
        return plans_arr, info, extras
    return plans_arr, info


def _open_ffmpeg_writer(out_mp4: Path, width: int, height: int, fps: float, crf: int = 18):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _crop_and_projector(info: Dict[str, Any], pw: int, ph: int):
    w, h = int(info["width"]), int(info["height"])
    if w == 1164 and h == 874:
        # Remove comma2k19 lower dashboard bezel cleanly while preserving road + sky (16:9)
        y0, crop_h = 44, 656
    elif h >= 1200:
        y0, crop_h = 4, h - 8
    else:
        y0, crop_h = 0, h
    k_mat = np.asarray(info["intrinsics"], dtype=np.float64)
    rpy = np.asarray(info["rpy"], dtype=np.float64)
    scale_xy = (pw / float(w), ph / float(crop_h))

    def _project(pts_calib: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return project_calib_to_image(pts_calib, rpy, k_mat, scale_xy=scale_xy, offset_y=float(y0))

    return y0, crop_h, _project


def render_trajectory_video(
    plans_left: np.ndarray,
    info_left: Dict[str, Any],
    out_mp4: PathLike,
    *,
    plans_right: Optional[np.ndarray] = None,
    info_right: Optional[Dict[str, Any]] = None,
    warmup_frames: int = 30,
    panel_size: Tuple[int, int] = (960, 600),
    horizon_s: float = 4.5,
    cam_height_m: float = 1.22,
    divider_px: int = 4,
    kf_dir: Optional[PathLike] = None,
    crf: int = 18,
) -> Dict[str, Any]:
    """Stream single or dual-panel videos with zero-text supersampled 3D waypoints to H.264 MP4."""
    pw, ph = panel_size
    dual = plans_right is not None and info_right is not None
    out_w = (pw * 2 + divider_px) if dual else pw
    out_h = ph
    fps = float(info_left.get("fps") or 20.0)

    y0_l, ch_l, proj_l = _crop_and_projector(info_left, pw, ph)
    cap_l = cv2.VideoCapture(info_left["narrow_path"])
    cap_r = cv2.VideoCapture(info_right["narrow_path"]) if dual else None
    y0_r, ch_r, proj_r = _crop_and_projector(info_right, pw, ph) if dual else (0, 0, None)

    total = min(len(plans_left), len(plans_right)) if dual else len(plans_left)
    eff_warmup = min(warmup_frames, max(0, total - 1))
    n_render = max(0, total - eff_warmup)
    kf_rel = {int(n_render * f) for f in (0.08, 0.25, 0.45, 0.65, 0.80, 0.94)} if kf_dir else set()
    kf_root = Path(kf_dir) if kf_dir else None
    if kf_root is not None:
        kf_root.mkdir(parents=True, exist_ok=True)

    out_path = Path(out_mp4)
    proc = _open_ffmpeg_writer(out_path, out_w, out_h, fps=fps, crf=crf)
    assert proc.stdin is not None

    saved_kfs: List[str] = []
    t_start = time.perf_counter()
    for i in range(total):
        ok_l, fr_l = cap_l.read()
        if not ok_l:
            break
        fr_r = None
        if cap_r is not None:
            ok_r, fr_r = cap_r.read()
            if not ok_r:
                break
        if i < eff_warmup:
            continue

        panel_l = cv2.resize(fr_l[y0_l : y0_l + ch_l, :], (pw, ph), interpolation=cv2.INTER_AREA)
        road_l = plans_left[i].copy()
        road_l[:, 2] += cam_height_m
        vis_l = render_modern_trajectory(panel_l, road_l, T_IDXS, proj_l, horizon_s=horizon_s)

        if dual and fr_r is not None and proj_r is not None and plans_right is not None:
            crop_r = fr_r[y0_r : y0_r + ch_r, :]
            panel_r = cv2.resize(crop_r, (pw, ph), interpolation=cv2.INTER_AREA)
            road_r = plans_right[i].copy()
            road_r[:, 2] += cam_height_m
            vis_r = render_modern_trajectory(panel_r, road_r, T_IDXS, proj_r, horizon_s=horizon_s)
            frame_out = compose_side_by_side(vis_l, vis_r, divider_px=divider_px)
        else:
            frame_out = vis_l

        proc.stdin.write(frame_out.tobytes())
        rel = i - eff_warmup
        if kf_root is not None and rel in kf_rel:
            kf_path = kf_root / f"t{rel / max(fps, 1.0):05.1f}s.jpg"
            cv2.imwrite(str(kf_path), frame_out, [cv2.IMWRITE_JPEG_QUALITY, 94])
            saved_kfs.append(str(kf_path))
        if (rel + 1) % 250 == 0:
            print(f"  rendered {rel + 1}/{n_render} frames ({time.perf_counter() - t_start:.1f}s)")

    proc.stdin.close()
    ret = proc.wait()
    cap_l.release()
    if cap_r is not None:
        cap_r.release()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")

    return {
        "video_path": str(out_path),
        "video_size_mb": round(out_path.stat().st_size / 1e6, 2),
        "rendered_frames": n_render,
        "duration_s": round(n_render / max(fps, 1.0), 2),
        "resolution": f"{out_w}x{out_h}",
        "keyframes": saved_kfs,
    }


def run_predict_video(
    runner: Any,
    source: PathLike,
    out_mp4: Optional[PathLike] = None,
    right_source: Optional[PathLike] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """High-level single/dual-panel video inference + optional zero-text MP4 render."""
    infer_keys = {
        "wide_video", "calib_rpy", "wide_from_device_euler", "intrinsics",
        "wide_intrinsics", "hfov_deg", "warmup_frames", "max_frames", "use_fp16",
    }
    render_keys = {
        "warmup_frames", "panel_size", "horizon_s", "cam_height_m",
        "divider_px", "kf_dir", "crf",
    }
    infer_kw = {k: v for k, v in kwargs.items() if k in infer_keys}
    plans_l, info_l = infer_driving_video(runner, source, **infer_kw)
    plans_r, info_r = (
        infer_driving_video(runner, right_source, **infer_kw) if right_source else (None, None)
    )
    render_meta = None
    if out_mp4 is not None:
        render_kw = {k: v for k, v in kwargs.items() if k in render_keys}
        render_meta = render_trajectory_video(
            plans_l, info_l, out_mp4, plans_right=plans_r, info_right=info_r, **render_kw
        )
    return {
        "plans": plans_l,
        "info": info_l,
        "right_plans": plans_r,
        "right_info": info_r,
        "render": render_meta,
    }
