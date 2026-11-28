"""ONNX / TensorRT BF16 export and one-command Jetson SSH deployment example."""

from __future__ import annotations

import argparse

from neuro_pilot import NeuroPilot
from neuro_pilot.utils.logger import logger


def main() -> None:
    p = argparse.ArgumentParser(description="Export TensorRT BF16 locally or deploy to Jetson.")
    p.add_argument("--model", default="pilot_mhp_v2.pt", help="Checkpoint path, .onnx, or hub key")
    p.add_argument("--scale", default="n", help="Model scale when loading YAML/key")
    p.add_argument(
        "--host", default="", help="SSH target (e.g. user@<jetson-ip>) for remote deploy"
    )
    p.add_argument("--remote-dir", default="", help="Target directory on Jetson")
    p.add_argument("--out-engine", default="", help="Local .engine output (run on target GPU)")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32", "int8"])
    p.add_argument("--workspace-mb", type=int, default=1024)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--work-dir", default="runs/deploy")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    model = NeuroPilot(args.model, scale=args.scale, device="cpu")
    if args.host and args.remote_dir:
        res = model.deploy(
            host=args.host,
            remote_dir=args.remote_dir,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            opset=args.opset,
            work_dir=args.work_dir,
            skip_build=args.skip_build,
            dry_run=args.dry_run,
        )
        logger.info(f"Jetson deploy -> {args.host}:{res['remote_engine'] or res['remote_onnx']}")
        return

    out_file = args.out_engine or f"{args.work_dir}/{model.model.model_key or 'model'}.engine"
    built = model.export(
        format="engine",
        file=out_file,
        opset=args.opset,
        bf16=(args.precision == "bf16"),
        half=(args.precision == "fp16"),
        int8=(args.precision == "int8"),
        workspace_mb=args.workspace_mb,
    )
    logger.info(f"Built local TensorRT engine -> {built}")


if __name__ == "__main__":
    main()
