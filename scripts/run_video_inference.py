import argparse
import cv2
import torch
from pathlib import Path
import time
from tqdm import tqdm
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.utils.torch_utils import select_device


def run_video_inference(
    weights,
    source,
    imgsz=None,
    conf=0.25,
    iou=0.45,
    device="",
    save=True,
    save_dir="runs/predict",
    show=False,
    command=1,
    scale="s",
    frames=0,
    half_opt=False,
    skip_heatmap=False,
    timeline=None,
    model_cfg=None,
    **kwargs,
):
    # 0. Setup Device
    device = select_device(device)
    cv2.setNumThreads(1)

    # 1. Load Model
    print(f"Loading model from {weights} (scale={scale})...")

    overrides = {"device": device, "skip_heatmap_inference": skip_heatmap}
    if model_cfg:
        overrides["model_cfg"] = model_cfg
        print(f"Overriding model config with: {model_cfg}")

    model = NeuroPilot(weights, scale=scale, **overrides)
    model.to(device)

    # Half precision optimization
    half = half_opt and (device.type != "cpu")
    if half:
        print("Using half precision (FP16).")
        model.half()
    model.eval()
    print(f"Model loaded successfully on {device}.")

    # 2. Setup Timeline
    timeline_data = None
    if timeline:
        import json

        with open(timeline, "r") as f:
            timeline_data = json.load(f)
        print(f"Loaded timeline with {len(timeline_data)} segments.")

    # 3. Setup Command Logic
    # Nếu có timeline, predictor sẽ tự xử lý frame_index bên trong
    cmd_fixed = None
    if not timeline:
        cmd_fixed = torch.tensor([[command]], device=device).long()

    # 4. Inference Loop
    results_gen = model.predict(
        source, stream=True, conf=conf, iou=iou, cmd=cmd_fixed, timeline=timeline, half=half, imgsz=imgsz
    )

    count = 0
    save_path = None
    out = None
    try:
        # Determine total frames and FPS
        total_frames = None
        fps = 30
        if Path(source).exists() and Path(source).suffix[1:].lower() in ["mp4", "avi", "mov", "mkv", "webm"]:
            cap_temp = cv2.VideoCapture(source)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30
            cap_temp.release()

        if frames > 0:
            total_frames = min(total_frames, frames) if total_frames else frames

        pbar = tqdm(total=total_frames, desc="Inference")
        t0 = time.time()

        for results in results_gen:
            if not results:
                continue

            count += 1
            if frames > 0 and count > frames:
                break

            res = results[0]
            if show or save:
                # Plot results
                plot_rgb = res.plot(heatmap=not skip_heatmap, max_dim=imgsz * 2 if imgsz else 1280)
                plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)

                # Determine current command for display
                display_cmd = command
                if timeline_data:
                    for seg in timeline_data:
                        if seg["start"] <= count <= seg["end"]:
                            display_cmd = seg["command"]
                            break

                # Draw Command on frame
                cmd_names = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
                cv2.putText(
                    plot_bgr,
                    f"CMD: {display_cmd} ({cmd_names.get(display_cmd, '??')})",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )

                plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)

            if show:
                try:
                    cv2.imshow("NeuroPilot Video Inference", cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception as e:
                    print(f"Warning: Could not show window: {e}")
                    show = False

            if save and plot_rgb is not None:
                if out is None:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    filename = Path(source).stem + "_out.avi"
                    save_path = str(Path(save_dir) / filename)
                    oh, ow = plot_rgb.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    out = cv2.VideoWriter(save_path, fourcc, fps, (ow, oh))

                if out is not None and out.isOpened():
                    out.write(cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR))

            pbar.update(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        if out:
            out.release()
        cv2.destroyAllWindows()
        if "pbar" in locals():
            pbar.close()

    if save_path:
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="model.pt path")
    parser.add_argument("--source", type=str, required=True, help="source")
    parser.add_argument("--imgsz", type=int, default=320, help="inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--nosave", action="store_false", dest="save", help="do not save results")
    parser.set_defaults(save=True)
    parser.add_argument("--save-dir", type=str, default="runs/predict", help="directory to save results")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--command", type=int, default=1, help="default command index")
    parser.add_argument("--timeline", type=str, help="path to command timeline JSON")
    parser.add_argument("--scale", type=str, default="s", help="model scale (n, s, m, l, x)")
    parser.add_argument("--model-cfg", type=str, help="Override model config path (YAML)")
    parser.add_argument("--frames", type=int, default=0, help="limit frames to process")
    parser.add_argument("--skip-heatmap-inference", action="store_true", default=True, help="Skip heatmap head")
    parser.add_argument(
        "--with-heatmap", action="store_false", dest="skip_heatmap_inference", help="Enable heatmap head"
    )
    args = parser.parse_args()

    run_video_inference(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_dir=args.save_dir,
        show=args.show,
        command=args.command,
        timeline=args.timeline,
        scale=args.scale,
        model_cfg=args.model_cfg,
        frames=args.frames,
        half_opt=args.half,
        skip_heatmap=args.skip_heatmap_inference,
    )
