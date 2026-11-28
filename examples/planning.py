"""End-to-end 3D trajectory planning example (`pilot_l.pt` / `pilot_s.pt`)."""

from __future__ import annotations

import argparse

import torch

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.logger import logger


def main() -> None:
    p = argparse.ArgumentParser(description="Run 3D E2E trajectory planning on a driving video.")
    p.add_argument("--model", default="pilot_l.pt", help="Checkpoint path or hub key (pilot_l.pt)")
    p.add_argument("--video", required=True, help="Primary driving video (.mp4/.hevc)")
    p.add_argument("--right-video", default=None, help="Optional second video for dual day/night")
    p.add_argument("--out", default="runs/planning/trajectory.mp4", help="Output MP4 path")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    planner = NeuroPilot(args.model, device=args.device)
    res = planner.predict_video(
        args.video,
        out_mp4=args.out,
        right_source=args.right_video,
        warmup_frames=args.warmup,
        max_frames=args.max_frames,
        hfov_deg=args.hfov_deg,
    )
    logger.info(f"Saved trajectory video -> {args.out} ({res})")


if __name__ == "__main__":
    main()
