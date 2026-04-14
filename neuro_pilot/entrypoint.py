import sys
import neuro_pilot.engine.task  # triggers @Registry.register_task
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.utils.logger import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import argparse

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]NeuroPilot 🚀[/bold cyan]\n"
        "[dim]End-to-End Autonomous Driving Framework[/dim]",
        border_style="cyan"
    ))

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

    export = subparsers.add_parser("export", help="Export a model")
    export.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    export.add_argument("--format", type=str, default="onnx", help="Export format (onnx, engine)")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--dynamic", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark performance")
    benchmark.add_argument("--model", type=str, required=True)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch", type=int, default=1)

    args, unknown = parser.parse_known_args()

    # Parse extra kwargs dynamically (e.g. --lambda_jepa 1.0 or lambda_jepa=1.0)
    kwargs = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        key, val = None, None
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
        elif arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith("--") and "=" not in unknown[i+1]:
                val = unknown[i+1]
                i += 1
            else:
                val = "True"
        if key and val is not None:
            # Type inference
            if val.lower() == 'true': val = True
            elif val.lower() == 'false': val = False
            else:
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
            kwargs[key] = val
        i += 1

    if not args.mode:
        parser.print_help()
        return

    try:
        model_scale = kwargs.pop('model_scale', 'n')
        model = NeuroPilot(args.model, scale=model_scale)

        if args.mode == "train":
            model.train(
                data=args.data,
                max_epochs=args.epochs,
                batch_size=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                **kwargs
            )
        elif args.mode == "predict":
            results = model.predict(
                args.source,
                conf=args.conf,
                imgsz=args.imgsz,
                stream=args.stream
            )
            if args.stream:
                for r in results:
                    for res in r:
                        console.print(f"[green]✔[/green] Processed {res.path}")
            else:
                import cv2
                from pathlib import Path
                vid_writer = None
                for res in results:
                    console.print(f"[green]✔[/green] Processed {res.path}")
                    if args.save:
                        p = Path(res.path)
                        is_video = p.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.ts']
                        if is_video:
                            if vid_writer is None:
                                save_path = Path("runs/predict") / p.name
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                plot_img = res.plot()
                                h, w = plot_img.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                vid_writer = cv2.VideoWriter(str(save_path), fourcc, 30.0, (w, h))

                            plot_img = res.plot()
                            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
                            vid_writer.write(plot_img)
                        else:
                            res.save()
                if vid_writer:
                    vid_writer.release()
                    console.print(f"[bold green]Video saved to runs/predict/{Path(args.source).name}[/bold green]")
        elif args.mode == "val":
            metrics = model.val(imgsz=args.imgsz)
            table = Table(title="Validation Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                     table.add_row(k, f"{v:.4f}")
            console.print(table)
        elif args.mode == "export":
            path = model.export(format=args.format, imgsz=args.imgsz, dynamic=args.dynamic)
            console.print(f"[bold green]Exported to {path}[/bold green]")
        elif args.mode == "benchmark":
            res = model.benchmark(imgsz=args.imgsz, batch=args.batch)
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Latency", f"{res['latency_ms']:.2f} ms")
            table.add_row("Throughput", f"{res['fps']:.2f} FPS")
            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
