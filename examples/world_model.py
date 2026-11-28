"""Discrete VQGAN world tokenizer example (`vq_enc.pt` + `vq_dec.pt`) via `NeuroPilot`."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.logger import logger


def run_vq_roundtrip(
    image_path: str | Path,
    out_path: str | Path,
    enc_model: str = "vq_enc.pt",
    dec_model: str = "vq_dec.pt",
    scale: str = "n",
    device: str = "cpu",
) -> float:
    """Encode an image into discrete world tokens, reconstruct it, and save side-by-side."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    enc = NeuroPilot(enc_model, scale=scale, device=device)
    dec = NeuroPilot(dec_model, scale=scale, device=device)
    tokens = enc.predict(bgr)["tokens"]
    rec_out = dec.predict(tokens)
    rec = rec_out["image_bgr"]

    ref = cv2.resize(bgr, (rec.shape[1], rec.shape[0]), interpolation=cv2.INTER_AREA)
    mse = float(np.mean((ref.astype(np.float32) - rec.astype(np.float32)) ** 2))
    psnr = float(10.0 * np.log10((255.0**2) / max(mse, 1e-6)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), dec.visualize(ref, rec_out))
    logger.info(f"Saved VQ roundtrip -> {out} | tokens={tokens.size} | PSNR={psnr:.2f} dB")
    return psnr


def main() -> None:
    p = argparse.ArgumentParser(description="Encode & decode a frame with VQGAN world tokenizer.")
    p.add_argument("--image", required=True, help="Input image (.jpg/.png)")
    p.add_argument("--out", default="runs/world/vq_roundtrip.jpg")
    p.add_argument("--enc", default="vq_enc.pt")
    p.add_argument("--dec", default="vq_dec.pt")
    p.add_argument("--scale", default="n")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_vq_roundtrip(
        args.image,
        args.out,
        enc_model=args.enc,
        dec_model=args.dec,
        scale=args.scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
