"""BFMC validation, video demo, and ONNX export example using `NeuroPilot`."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.logger import logger


def render_video(
    model: NeuroPilot,
    video_path: str | Path,
    out_mp4: str | Path,
    command: int = 0,
    max_frames: int = 0,
    stride: int = 1,
    conf: float = 0.30,
) -> float:
    """Run `model.predict` + `model.visualize` over a video and encode H.264 MP4."""
    cap = cv2.VideoCapture(str(video_path))
    fps = max(15.0, float(cap.get(cv2.CAP_PROP_FPS) or 30.0) / max(1, stride))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 384)
    out_w, out_h = src_w - (src_w % 2), src_h - (src_h % 2)
    tmp_mp4 = str(Path(out_mp4).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    lat_ms, idx, written = [], 0, 0
    while max_frames <= 0 or written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.predict(frame, command=command, conf=conf)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            canvas = (
                frame
                if (frame.shape[1], frame.shape[0]) == (out_w, out_h)
                else cv2.resize(frame, (out_w, out_h))
            )
            vis = model.visualize(
                canvas, out, command=command, conf_thres=conf, tag=f"#{idx:04d}"
            )
            writer.write(vis)
            written += 1
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "22", str(out_mp4)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    Path(tmp_mp4).unlink(missing_ok=True)
    return float(np.mean(lat_ms[5:])) if len(lat_ms) > 5 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate, visualize, and export BFMC via NeuroPilot.")
    p.add_argument("--ckpt", default="pilot_mhp_v2.pt")
    p.add_argument("--val-json", default="")
    p.add_argument("--out-grid", default="")
    p.add_argument("--video", default="")
    p.add_argument("--out-mp4", default="")
    p.add_argument("--export-onnx", default="")
    p.add_argument("--export-engine", default="")
    p.add_argument("--bf16", action="store_true", help="Build TensorRT engine in BF16")
    p.add_argument("--deploy-host", default="", help="SSH target (e.g. orin@192.168.115.78)")
    p.add_argument("--deploy-dir", default="", help="Remote Jetson directory for ONNX/engine")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32", "int8"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--command", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = full video")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--conf", type=float, default=0.30)
    args = p.parse_args()

    model = NeuroPilot(args.ckpt)
    if args.val_json:
        metrics = model.val(val_json=args.val_json, save_grid=args.out_grid or None, conf=args.conf)
        logger.info(
            f"Val Score={metrics['score']:.2f} | MSE={metrics['mse']:.5f} | "
            f"ADE={metrics['ade_px640']:.1f}px | mAP50={metrics['map50']:.4f}"
        )
    if args.video and args.out_mp4:
        ms = render_video(
            model, args.video, args.out_mp4,
            command=args.command, max_frames=args.max_frames,
            stride=args.stride, conf=args.conf,
        )
        fps_val = 1000.0 / max(ms, 0.1)
        logger.info(f"Saved {args.out_mp4} | mean latency={ms:.2f} ms ({fps_val:.1f} FPS)")
    if args.export_onnx:
        out = model.export(format="onnx", file=args.export_onnx, opset=17)
        logger.info(f"Exported ONNX: {out}")
    if args.export_engine:
        out = model.export(
            format="engine", file=args.export_engine, opset=17, bf16=args.bf16, half=not args.bf16
        )
        logger.info(f"Exported TensorRT engine: {out}")
    if args.deploy_host and args.deploy_dir:
        res = model.deploy(
            host=args.deploy_host,
            remote_dir=args.deploy_dir,
            precision=args.precision,
            dry_run=args.dry_run,
        )
        logger.info(f"Jetson deploy target: {args.deploy_host}:{res['remote_engine']}")


if __name__ == "__main__":
    main()
