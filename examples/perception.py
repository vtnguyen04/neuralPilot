"""Multi-task perception example: segmentation, depth, ego-lane, and CIPO via `NeuroPilot`."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.visualize import render_image_grid


def run_perception_grid(
    image_path: str | Path,
    out_path: str | Path,
    seg_model: str = "seg_l.pt",
    depth_model: str = "depth.pt",
    cipo_model: str = "cipo.pt",
    lane_model: str = "lane.pt",
    scale: str = "n",
    device: str = "cpu",
) -> Path:
    """Run seg, depth, CIPO, and lane on one BGR image and write a 2x2 visual grid."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    tile_sz = (640, 360)

    seg = NeuroPilot(seg_model, scale=scale, device=device)
    depth = NeuroPilot(depth_model, device=device)
    cipo = NeuroPilot(cipo_model, device=device)
    lane = NeuroPilot(lane_model, device=device)

    vis_seg = cv2.resize(seg.visualize(bgr, width=256, height=128), tile_sz)
    vis_dep = cv2.resize(depth.visualize(bgr), tile_sz)
    vis_cipo = cv2.resize(cipo.visualize(bgr, conf=0.40), tile_sz)
    vis_lane = cv2.resize(lane.visualize(bgr), tile_sz)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), render_image_grid([vis_seg, vis_cipo, vis_dep, vis_lane], cols=2))
    logger.info(f"Saved 2x2 perception grid -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run seg, depth, CIPO, and lane on an image.")
    p.add_argument("--image", required=True, help="Input road image (.jpg/.png)")
    p.add_argument("--out", default="runs/perception/grid.jpg", help="Output 2x2 grid image")
    p.add_argument("--seg", default="seg_l.pt", help="Segmentation model (seg_l.pt or seg.pt)")
    p.add_argument("--depth", default="depth.pt")
    p.add_argument("--cipo", default="cipo.pt")
    p.add_argument("--lane", default="lane.pt")
    p.add_argument("--scale", default="n")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_perception_grid(
        args.image,
        args.out,
        seg_model=args.seg,
        depth_model=args.depth,
        cipo_model=args.cipo,
        lane_model=args.lane,
        scale=args.scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
