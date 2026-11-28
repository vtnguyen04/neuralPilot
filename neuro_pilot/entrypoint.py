import sys
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.utils.logger import logger, log_banner, log_table, console
import argparse

def print_banner():
    log_banner("NeuroPilot 🚀", "End-to-End Autonomous Driving Framework")


def infer_type(val: str):
    val_lower = str(val).lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    try:
        return float(val) if "." in val else int(val)
    except ValueError:
        return val


def parse_dynamic_kwargs(unknown_args: list) -> dict:
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]

        if "=" in arg:
            key, val = arg.split("=", 1)
            kwargs[key.lstrip("-")] = infer_type(val)
            i += 1
        elif arg.startswith("-") and i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
            key = arg.lstrip("-")
            val = unknown_args[i + 1]
            kwargs[key] = infer_type(val)
            i += 2
        else:
            kwargs[arg.lstrip("-")] = True
            i += 1
    return kwargs


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="NeuroPilot CLI", usage="neuropilot [MODE] [ARGS]")
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("model", type=str, help="Model configuration (yaml)")
    train.add_argument("--data", type=str, help="Dataset configuration")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--device", type=str, default="0")

    predict = subparsers.add_parser("predict", help="Run inference")
    predict.add_argument("source", type=str, help="Input source (image/dir/video/stream)")
    predict.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--stream", action="store_true")
    predict.add_argument("--save", action="store_true")

    val = subparsers.add_parser("val", help="Validate a model")
    val.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    val.add_argument("--data", type=str, help="Dataset configuration")
    val.add_argument("--imgsz", type=int, default=640)

    export = subparsers.add_parser("export", help="Export a model to ONNX, TensorRT, or TorchScript")
    export.add_argument("--model", type=str, required=True, help="Path to weights (.pt) or zoo key")
    export.add_argument("--format", type=str, default="onnx", help="Export format (onnx, engine, ts)")
    export.add_argument("--file", type=str, default=None, help="Output file path")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--bf16", action="store_true", help="Build TensorRT engine in BF16 (Orin)")
    export.add_argument("--fp32", action="store_true", help="Disable FP16/BF16 half precision")
    export.add_argument("--int8", action="store_true", help="Enable TensorRT INT8 calibration flag")
    export.add_argument("--workspace-mb", type=int, default=None, help="TensorRT workspace (MiB)")
    export.add_argument("--dynamic", action="store_true")

    deploy = subparsers.add_parser("deploy", help="Export ONNX, upload via SCP, and build TRT on Jetson")
    deploy.add_argument("--model", type=str, required=True, help="Path to .pt/.onnx or zoo model key")
    deploy.add_argument("--host", type=str, required=True, help="SSH target (e.g. user@192.168.1.50)")
    deploy.add_argument("--remote-dir", type=str, required=True, help="Target directory on Jetson")
    deploy.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32", "int8"])
    deploy.add_argument("--workspace-mb", type=int, default=1024)
    deploy.add_argument("--port", type=int, default=22)
    deploy.add_argument("--ssh-key", type=str, default=None)
    deploy.add_argument("--trtexec", type=str, default="/usr/src/tensorrt/bin/trtexec")
    deploy.add_argument("--opset", type=int, default=17)
    deploy.add_argument("--name", type=str, default=None, help="Override output stem on Jetson")
    deploy.add_argument("--work-dir", type=str, default="runs/deploy")
    deploy.add_argument("--skip-build", action="store_true", help="Only upload ONNX, skip trtexec")
    deploy.add_argument("--dry-run", action="store_true", help="Print deploy commands without SSH")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark performance")
    benchmark.add_argument("--model", type=str, required=True)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch", type=int, default=1)

    download = subparsers.add_parser("download", help="Download pretrained zoo checkpoints (.pt)")
    download.add_argument("models", nargs="*", help="Model keys (e.g. pilot_l seg cipo depth)")
    download.add_argument("--all", action="store_true", help="Download all supported checkpoints")
    download.add_argument("--dir", type=str, default=None, help="Output cache directory")
    download.add_argument("--list", action="store_true", help="List downloadable checkpoints")
    download.add_argument("--force", action="store_true", help="Re-download even if cached")

    # Default to 'train' if first argument is a model file
    if len(sys.argv) > 1 and sys.argv[1] not in subparsers.choices and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "train")

    args, unknown = parser.parse_known_args()

    # Parse extra kwargs dynamically (e.g. --lambda_jepa 1.0 or lambda_jepa=1.0)
    kwargs = parse_dynamic_kwargs(unknown)
    if not args.mode:
        parser.print_help()
        return

    try:
        if args.mode == "download":
            from neuro_pilot.nn.zoo.hub import WEIGHT_SPECS, download_weights, list_downloadable

            if args.list or (not args.all and not args.models):
                rows = {k: f"{WEIGHT_SPECS[k].license} ({WEIGHT_SPECS[k].kind})" for k in list_downloadable()}
                log_table("Downloadable Checkpoints", rows)
                return
            targets = list_downloadable() if args.all else list(args.models)
            for key in targets:
                path = download_weights(key, dest_dir=args.dir, force=args.force)
                console.print(f"[green]Ready[/green] {key} -> {path}")
            return

        if args.mode == "deploy":
            from neuro_pilot.deploy import JetsonTarget, deploy_to_jetson

            target = JetsonTarget(
                host=args.host,
                remote_dir=args.remote_dir,
                port=args.port,
                trtexec=args.trtexec,
                precision=args.precision,
                workspace_mb=args.workspace_mb,
                ssh_key=args.ssh_key,
            )
            res = deploy_to_jetson(
                args.model,
                target,
                opset=args.opset,
                scale=kwargs.pop("model_scale", "n"),
                work_dir=args.work_dir,
                name=args.name,
                skip_build=args.skip_build,
                dry_run=args.dry_run,
            )
            log_table(
                "Jetson Deploy Plan" if args.dry_run else "Jetson Deploy Complete",
                {
                    "Local ONNX": res["local_onnx"],
                    "Remote ONNX": res["remote_onnx"],
                    "Remote Engine": res["remote_engine"] or "(skipped)",
                    "Precision": res["precision"],
                },
            )
            return

        model_scale = kwargs.pop("model_scale", "n")
        model = NeuroPilot(args.model, scale=model_scale)

        if args.mode == "train":
            model.train(
                data=args.data,
                max_epochs=args.epochs,
                batch_size=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                **kwargs,
            )
        elif args.mode == "predict":
            results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, stream=args.stream)

            if args.stream:
                for r in results:
                    for res in r:
                        console.print(f"[green]✔[/green] Processed {res.path}")
                return

            import cv2
            from pathlib import Path

            vid_writer = None
            for res in results:
                logger.info(f"✔ Processed {res.path}")
                if not args.save:
                    continue

                p = Path(res.path)
                if p.suffix.lower() not in [".mp4", ".avi", ".mkv", ".mov", ".ts"]:
                    res.save()
                    continue

                if vid_writer is None:
                    save_path = Path("runs/predict") / p.name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = res.plot(**kwargs).shape[:2]
                    vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

                plot_img = cv2.cvtColor(res.plot(**kwargs), cv2.COLOR_RGB2BGR)
                vid_writer.write(plot_img)

            if vid_writer:
                vid_writer.release()
                console.print(f"[bold green]Video saved to runs/predict/{Path(args.source).name}[/bold green]")
        elif args.mode == "val":
            metrics = model.val(imgsz=args.imgsz)
            log_table("Validation Metrics", metrics)
        elif args.mode == "export":
            exp_kwargs = {
                "format": args.format,
                "imgsz": args.imgsz,
                "opset": args.opset,
                "half": (not args.fp32 and not args.bf16),
                "bf16": args.bf16,
                "int8": args.int8,
                "dynamic": args.dynamic,
                **kwargs,
            }
            if args.file:
                exp_kwargs["file"] = args.file
            if args.workspace_mb is not None:
                exp_kwargs["workspace_mb"] = args.workspace_mb
            path = model.export(**exp_kwargs)
            logger.info(f"Exported to {path}")
        elif args.mode == "benchmark":
            res = model.benchmark(imgsz=args.imgsz, batch=args.batch)
            log_table("Benchmark Results", {"Latency": f"{res['latency_ms']:.2f} ms", "Throughput": f"{res['fps']:.2f} FPS"})

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
