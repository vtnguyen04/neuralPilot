"""Unified CLI for single/dual trajectory demos and full multi-task README showcases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from neuro_pilot import NeuroPilot
from neuro_pilot.engine.video import (
    _crop_and_projector,
    _open_ffmpeg_writer,
    infer_driving_video,
    render_trajectory_video,
)
from neuro_pilot.utils.camera import T_IDXS, X_IDXS
from neuro_pilot.utils.overlay import (
    compose_grid_2x2,
    render_depth_panel,
    render_multilane_3d,
    render_perception_boxes,
    render_proximity_gauge,
    render_seg_overlay,
)
from neuro_pilot.utils.viz import compose_side_by_side, render_modern_trajectory


def _crop_road(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[44:700, :] if (w == 1164 and h == 874) else frame


def _load_perception(pt_root: Path, device: str) -> Dict[str, NeuroPilot]:
    seg_l = pt_root / "perception/seg_l.pt"
    seg_pt = seg_l if seg_l.is_file() else (pt_root / "perception/seg.pt")
    return {
        "cipo": NeuroPilot(pt_root / "perception/cipo.pt", device=device),
        "drive": NeuroPilot(pt_root / "perception/drive.pt", device=device),
        "depth": NeuroPilot(pt_root / "perception/depth.pt", device=device),
        "seg": NeuroPilot(seg_pt, device=device),
    }


def render_perception_and_hero(
    day_video: Path,
    night_video: Path,
    plans_day: np.ndarray,
    info_day: Dict,
    extras_day: Dict,
    plans_night: np.ndarray,
    info_night: Dict,
    extras_night: Dict,
    pt_root: Path,
    out_dir: Path,
    device: str,
    warmup: int = 30,
    perc_frames: int = 400,
    hero_frames: int = 500,
    cam_h: float = 1.22,
) -> Dict[str, str]:
    """Render 2x2 perception+seg (`perception.mp4`) and flagship (`hero.mp4`)."""
    models = _load_perception(pt_root, device)
    pw, ph, div = 640, 360, 4
    grid_w, grid_h = pw * 2 + div, ph * 2 + div

    perc_mp4, hero_mp4 = out_dir / "perception.mp4", out_dir / "hero.mp4"
    perc_jpg, hero_jpg = out_dir / "perception.jpg", out_dir / "hero.jpg"
    proc_perc = _open_ffmpeg_writer(perc_mp4, grid_w, grid_h, fps=20.0, crf=19)
    proc_hero = _open_ffmpeg_writer(hero_mp4, grid_w, grid_h, fps=20.0, crf=19)
    assert proc_perc.stdin is not None and proc_hero.stdin is not None

    y0_d, ch_d, proj_d = _crop_and_projector(info_day, pw, ph)
    y0_n, ch_n, proj_n = _crop_and_projector(info_night, pw, ph)
    cap_d, cap_n = cv2.VideoCapture(str(day_video)), cv2.VideoCapture(str(night_video))
    limit = min(len(plans_day), len(plans_night), warmup + max(perc_frames, hero_frames))

    with torch.inference_mode():
        for i in range(limit):
            ok_d, fr_d = cap_d.read()
            ok_n, fr_n = cap_n.read()
            if not (ok_d and ok_n):
                break
            if i < warmup:
                continue
            rel = i - warmup
            crop_d, crop_n = _crop_road(fr_d), _crop_road(fr_n)
            seg_crop_d = fr_d[y0_d : y0_d + ch_d, :]
            pan_d = cv2.resize(seg_crop_d, (pw, ph), interpolation=cv2.INTER_AREA)
            pan_n = cv2.resize(fr_n[y0_n : y0_n + ch_n, :], (pw, ph), interpolation=cv2.INTER_AREA)

            boxes_d = models["cipo"].predict(crop_d, conf=0.45)["boxes"]
            dist_d = float(models["drive"].predict(crop_d)["distance_m"])
            depth_d = models["depth"].predict(crop_d)["depth"]
            seg_d = models["seg"].predict(seg_crop_d)["class_map"]

            p_seg_d = render_seg_overlay(pan_d, seg_d)
            p_lanes_d = render_multilane_3d(
                pan_d, extras_day["lane_lines"][i], extras_day["lane_probs"][i],
                extras_day["road_edges"][i], X_IDXS, proj_d,
                lane_stds=extras_day["lane_stds"][i], fill_ego_lane=True,
            )
            p_fuse_d = render_proximity_gauge(render_perception_boxes(p_lanes_d, boxes_d), dist_d)
            p_depth_d = render_depth_panel(depth_d, (pw, ph))

            if rel < perc_frames:
                boxes_n = models["cipo"].predict(crop_n, conf=0.40)["boxes"]
                dist_n = float(models["drive"].predict(crop_n)["distance_m"])
                p_lanes_n = render_multilane_3d(
                    pan_n, extras_night["lane_lines"][i], extras_night["lane_probs"][i],
                    extras_night["road_edges"][i], X_IDXS, proj_n,
                    lane_stds=extras_night["lane_stds"][i], fill_ego_lane=True,
                )
                p_fuse_n = render_proximity_gauge(
                    render_perception_boxes(p_lanes_n, boxes_n), dist_n
                )
                grid_p = compose_grid_2x2(p_seg_d, p_fuse_d, p_depth_d, p_fuse_n, div)
                proc_perc.stdin.write(grid_p.tobytes())
                if rel == perc_frames // 2:
                    cv2.imwrite(str(perc_jpg), grid_p, [cv2.IMWRITE_JPEG_QUALITY, 94])

            if rel < hero_frames:
                rd, rn = plans_day[i].copy(), plans_night[i].copy()
                rd[:, 2] += cam_h
                rn[:, 2] += cam_h
                vis_d = render_modern_trajectory(pan_d, rd, T_IDXS, proj_d, horizon_s=4.5)
                vis_n = render_modern_trajectory(pan_n, rn, T_IDXS, proj_n, horizon_s=4.5)
                grid_h_img = compose_grid_2x2(vis_d, vis_n, p_seg_d, p_fuse_d, div)
                proc_hero.stdin.write(grid_h_img.tobytes())
                if rel == hero_frames // 2:
                    cv2.imwrite(str(hero_jpg), grid_h_img, [cv2.IMWRITE_JPEG_QUALITY, 94])

    proc_perc.stdin.close()
    proc_hero.stdin.close()
    if proc_perc.wait() != 0 or proc_hero.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding perception/hero videos")
    cap_d.release()
    cap_n.release()
    return {
        "perception_mp4": str(perc_mp4),
        "perception_jpg": str(perc_jpg),
        "hero_mp4": str(hero_mp4),
        "hero_jpg": str(hero_jpg),
    }


def render_world_vq(
    video: Path,
    pt_root: Path,
    out_dir: Path,
    device: str,
    warmup: int = 30,
    num_frames: int = 300,
) -> Dict[str, object]:
    """Render side-by-side Original | VQ 128-token reconstruction (`world_vq.mp4`)."""
    enc = NeuroPilot(pt_root / "world/vq_enc.pt", device=device)
    dec = NeuroPilot(pt_root / "world/vq_dec.pt", device=device)
    out_mp4, out_jpg = out_dir / "world_vq.mp4", out_dir / "world_vq.jpg"
    pw, ph, div = 512, 256, 4
    proc = _open_ffmpeg_writer(out_mp4, pw * 2 + div, ph, fps=20.0, crf=19)
    assert proc.stdin is not None

    cap = cv2.VideoCapture(str(video))
    psnrs: List[float] = []
    with torch.inference_mode():
        for i in range(warmup + num_frames):
            ok, fr = cap.read()
            if not ok:
                break
            if i < warmup:
                continue
            orig = cv2.resize(_crop_road(fr), (256, 128), interpolation=cv2.INTER_AREA)
            rec = dec.predict(enc.predict(orig)["tokens"])["image_bgr"]
            mse = float(np.mean((orig.astype(np.float32) - rec.astype(np.float32)) ** 2))
            psnrs.append(10.0 * np.log10((255.0**2) / max(mse, 1e-6)))
            frame = compose_side_by_side(
                cv2.resize(orig, (pw, ph), interpolation=cv2.INTER_CUBIC),
                cv2.resize(rec, (pw, ph), interpolation=cv2.INTER_CUBIC),
                divider_px=div,
            )
            proc.stdin.write(frame.tobytes())
            if i - warmup == num_frames // 2:
                cv2.imwrite(str(out_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])

    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding world_vq.mp4")
    cap.release()
    return {
        "world_vq_mp4": str(out_mp4),
        "world_vq_jpg": str(out_jpg),
        "mean_psnr_db": round(float(np.mean(psnrs)), 2) if psnrs else 0.0,
    }


def _run_video(args: argparse.Namespace) -> None:
    runner = NeuroPilot(args.pt, device=args.device)
    plans_l, info_l = infer_driving_video(
        runner, args.video, hfov_deg=args.hfov_deg,
        warmup_frames=args.warmup, max_frames=args.max_frames,
    )
    plans_r = info_r = None
    if args.right_video:
        plans_r, info_r = infer_driving_video(
            runner, args.right_video, hfov_deg=args.hfov_deg,
            warmup_frames=args.warmup, max_frames=args.max_frames,
        )
    meta = render_trajectory_video(
        plans_l, info_l, args.out, plans_right=plans_r, info_right=info_r,
        warmup_frames=args.warmup, panel_size=(args.width, args.height),
        horizon_s=args.horizon_s, cam_height_m=args.cam_height_m, kf_dir=args.kf_dir,
    )
    summary = {"model": Path(args.pt).name, "left": info_l, "right": info_r, **meta}
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_suite(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_root, day_v, night_v = Path(args.pt_root), Path(args.day_video), Path(args.night_video)

    pilot = NeuroPilot(pt_root / "planner/pilot_l.pt", device=args.device)
    plans_d, info_d, ext_d = infer_driving_video(
        pilot, day_v, calib_rpy=[0.0, args.pitch, 0.0],
        warmup_frames=args.warmup, max_frames=args.max_frames, return_extras=True,
    )
    plans_n, info_n, ext_n = infer_driving_video(
        pilot, night_v, calib_rpy=[0.0, args.pitch, 0.0],
        warmup_frames=args.warmup, max_frames=args.max_frames, return_extras=True,
    )
    del pilot
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    e2e_meta = render_trajectory_video(
        plans_d, info_d, out_dir / "day_night.mp4", plans_right=plans_n, info_right=info_n,
        warmup_frames=args.warmup, panel_size=(960, 540), horizon_s=4.5,
        kf_dir=out_dir / "frames", crf=19,
    )
    perc_hero = render_perception_and_hero(
        day_v, night_v, plans_d, info_d, ext_d, plans_n, info_n, ext_n,
        pt_root, out_dir, args.device,
        warmup=args.warmup, perc_frames=args.perc_frames, hero_frames=args.hero_frames,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vq_meta = render_world_vq(
        day_v, pt_root, out_dir, args.device, warmup=args.warmup, num_frames=args.vq_frames
    )
    summary: Dict[str, Optional[object]] = {
        "dataset": "comma2k19 (MIT License)",
        "day": info_d,
        "night": info_n,
        "planner": e2e_meta,
        "perception_hero": perc_hero,
        "world_vq": vq_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroPilot zero-text video demo renderer.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dev_default = "cuda" if torch.cuda.is_available() else "cpu"

    p_vid = sub.add_parser("video", help="Render single or dual-panel 3D trajectory video.")
    p_vid.add_argument("--pt", required=True)
    p_vid.add_argument("--video", required=True)
    p_vid.add_argument("--right_video", default=None)
    p_vid.add_argument("--out", required=True)
    p_vid.add_argument("--kf_dir", default=None)
    p_vid.add_argument("--warmup", type=int, default=30)
    p_vid.add_argument("--max_frames", type=int, default=None)
    p_vid.add_argument("--width", type=int, default=960)
    p_vid.add_argument("--height", type=int, default=600)
    p_vid.add_argument("--horizon_s", type=float, default=4.5)
    p_vid.add_argument("--cam_height_m", type=float, default=1.22)
    p_vid.add_argument("--hfov_deg", type=float, default=70.0)
    p_vid.add_argument("--device", default=dev_default)

    p_suite = sub.add_parser("suite", help="Render all 4 MIT multi-task README demos.")
    p_suite.add_argument("--day_video", required=True)
    p_suite.add_argument("--night_video", required=True)
    p_suite.add_argument("--pt_root", required=True)
    p_suite.add_argument("--out_dir", required=True)
    p_suite.add_argument("--pitch", type=float, default=0.045)
    p_suite.add_argument("--warmup", type=int, default=30)
    p_suite.add_argument("--max_frames", type=int, default=1200)
    p_suite.add_argument("--perc_frames", type=int, default=400)
    p_suite.add_argument("--hero_frames", type=int, default=500)
    p_suite.add_argument("--vq_frames", type=int, default=300)
    p_suite.add_argument("--device", default=dev_default)

    args = parser.parse_args()
    _run_video(args) if args.cmd == "video" else _run_suite(args)


if __name__ == "__main__":
    main()
