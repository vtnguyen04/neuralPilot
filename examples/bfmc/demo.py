"""BFMC visual evaluator, held-out video renderer, and ONNX export CLI (reuses GraphModel.export_onnx)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from examples.bfmc.metrics import CLS_NAMES, CMD_NAMES, decode_nms
from neuro_pilot.engine.predictors.utils import bgr_to_yuv6
from neuro_pilot.nn.zoo import load_zoo_pt

_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (60, 70, 255), (255, 80, 220), (80, 230, 120), (0, 210, 255), (255, 220, 80),
    (255, 140, 40), (255, 255, 110), (255, 150, 90), (120, 255, 180),
    (255, 170, 0), (120, 180, 255), (90, 255, 255), (40, 40, 240),
)


def draw_bfmc_overlay(
    bgr: np.ndarray,
    waypoints: np.ndarray,
    dets: np.ndarray,
    command: int = 0,
    gt_wps: np.ndarray | None = None,
    conf_thres: float = 0.30,
    tag: str = "",
) -> np.ndarray:
    """Draw trajectory corridor, 13-class boxes, optional gold GT path, and compact HUD."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if gt_wps is not None and len(gt_wps) >= 2:
        gpts = np.round(gt_wps * np.array([w, h])).astype(np.int32)
        for i in range(len(gpts) - 1):
            cv2.line(vis, tuple(gpts[i]), tuple(gpts[i + 1]), (0, 215, 255), 2, cv2.LINE_AA)

    pts = np.round(waypoints * np.array([w, h])).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    overlay, left_poly, right_poly = vis.copy(), [], []
    for i, (px, py) in enumerate(pts):
        half_w = int(max(6, (1.0 - 0.65 * (i / max(1, len(pts) - 1))) * (w * 0.065)))
        left_poly.append([max(0, px - half_w), py])
        right_poly.append([min(w - 1, px + half_w), py])
    cv2.fillPoly(overlay, [np.array(left_poly + right_poly[::-1], dtype=np.int32)], (255, 200, 40), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, vis, 0.72, 0.0, vis)
    for i in range(len(pts) - 1):
        cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (255, 235, 90), 3, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(vis, (int(px), int(py)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    for x1, y1, x2, y2, conf, cid in dets:
        if conf < conf_thres:
            continue
        ci = int(cid)
        col = _PALETTE[ci % len(_PALETTE)]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, p1, p2, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.rectangle(vis, p1, p2, col, 2, cv2.LINE_AA)
        label = f"{CLS_NAMES[ci]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 6, p1[1] - 4)
        cv2.rectangle(vis, (p1[0], ty - th - 5), (p1[0] + tw + 6, ty + 2), (18, 22, 28), -1)
        cv2.putText(vis, label, (p1[0] + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    hud = f"{tag} | CMD: {CMD_NAMES[command]}" if tag else f"CMD: {CMD_NAMES[command]}"
    cv2.rectangle(vis, (0, 0), (w, 26), (16, 20, 26), -1)
    cv2.putText(vis, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 245, 255), 1, cv2.LINE_AA)
    return vis


@torch.no_grad()
def render_val_grid(
    model: torch.nn.Module, val_json: str | Path, out_jpg: str | Path,
    device: torch.device, conf_thres: float = 0.30, img_w: int = 640, img_h: int = 512,
) -> None:
    recs = json.loads(Path(val_json).read_text())
    picks, seen = [], set()
    for target_cmd in (0, 1, 2, 3):
        cands = [r for r in recs if r["command"] == target_cmd and r["bboxes"]] or [r for r in recs if r["command"] == target_cmd]
        for idx in np.linspace(0, max(0, len(cands) - 1), min(3, len(cands)), dtype=int):
            r = cands[int(idx)]
            if r["id"] not in seen:
                picks.append(r)
                seen.add(r["id"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    tiles: List[np.ndarray] = []
    for rec in picks[:12]:
        bgr = cv2.imread(rec["img_path"], cv2.IMREAD_COLOR)
        yuv = torch.from_numpy(bgr_to_yuv6(bgr, img_w, img_h)).unsqueeze(0).to(device)
        cmd = torch.tensor([int(rec["command"])], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(img=yuv, command=cmd)
        wp = out["waypoints"][0].float().cpu().numpy()
        dets = decode_nms(out["det"].float(), img_w=img_w, img_h=img_h, conf_thres=conf_thres)[0]
        tiles.append(draw_bfmc_overlay(
            cv2.resize(bgr, (512, 320)), wp, dets, int(rec["command"]),
            gt_wps=np.asarray(rec["waypoints"], dtype=np.float32), conf_thres=conf_thres, tag=rec["session"][:16],
        ))
    grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(out_jpg), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


@torch.no_grad()
def render_video_demo(
    model: torch.nn.Module, video_path: str | Path, out_mp4: str | Path,
    device: torch.device, max_frames: int = 450, stride: int = 2,
    conf_thres: float = 0.30, img_w: int = 640, img_h: int = 512,
) -> float:
    cap = cv2.VideoCapture(str(video_path))
    out_fps = max(15.0, (cap.get(cv2.CAP_PROP_FPS) or 30.0) / stride)
    tmp_mp4 = str(Path(out_mp4).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (640, 384))
    cmd = torch.zeros((1,), dtype=torch.long, device=device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    lat_ms, idx, written = [], 0, 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yuv = torch.from_numpy(bgr_to_yuv6(frame, img_w, img_h)).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(img=yuv, command=cmd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            wp = out["waypoints"][0].float().cpu().numpy()
            dets = decode_nms(out["det"].float(), img_w=img_w, img_h=img_h, conf_thres=conf_thres)[0]
            writer.write(draw_bfmc_overlay(cv2.resize(frame, (640, 384)), wp, dets, 0, conf_thres=conf_thres))
            written += 1
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    Path(tmp_mp4).unlink(missing_ok=True)
    return float(np.mean(lat_ms[5:])) if len(lat_ms) > 5 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-json", default="")
    p.add_argument("--out-grid", default="")
    p.add_argument("--video", default="")
    p.add_argument("--out-mp4", default="")
    p.add_argument("--export-onnx", default="")
    p.add_argument("--conf", type=float, default=0.30)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img_w, img_h = int(raw.get("img_w", 640)), int(raw.get("img_h", 512))
    model = load_zoo_pt(args.ckpt, device=str(device))
    if args.val_json and args.out_grid:
        render_val_grid(model, args.val_json, args.out_grid, device, args.conf, img_w, img_h)
    if args.video and args.out_mp4:
        ms = render_video_demo(model, args.video, args.out_mp4, device, conf_thres=args.conf, img_w=img_w, img_h=img_h)
        print(f"Saved video: {args.out_mp4} | mean GPU latency={ms:.2f} ms ({1000.0/max(ms,0.1):.1f} FPS)")
    if args.export_onnx:
        model.cpu().export_onnx(args.export_onnx, opset=17)
        print(f"Exported ONNX: {args.export_onnx}")


if __name__ == "__main__":
    main()
