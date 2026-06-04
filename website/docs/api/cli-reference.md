---
sidebar_position: 1
---
# CLI Reference Manual

NeuroPilot provides a unified Command Line Interface (CLI) entrypoint via the `neuropilot` executable. This document serves as the exhaustive command reference, documenting every subcommand, option flags, defaults, constraints, hyperparameter overrides, and realistic execution stdout.

---

## 1. Global CLI Syntax

All operations are executed through the `neuropilot` command. When developing locally or using a virtual environment manager, prefix the execution with `uv run`:

```bash
uv run neuropilot <SUBCOMMAND> [OPTIONS] [KEY=VALUE ...]
```

You can append arbitrary key-value pairs (`KEY=VALUE`) at the end of any command to override model hyperparameters, loss weights, learning rates, or optimizer settings dynamically at runtime.

---

## 2. Subcommands Directory

### A. `train` — Model Training Pipeline
The `train` subcommand instantiates the training loop. It parses the model architecture YAML and dataset configurations, configures loaders, establishes CUDA bindings, and checkpoints weights.

#### Usage
```bash
uv run neuropilot train <MODEL_CONFIG> --data <DATASET_CONFIG> [OPTIONS] [KEY=VALUE ...]
```

#### Positional Arguments
* **`MODEL_CONFIG`** *(string, required)*: Path to the model structure definition file (e.g., `neuro_pilot/cfg/models/neuralPilot.yaml`).

#### Options
* **`--data`** *(string, required)*: Path to the dataset mapping configuration (e.g., `data/yolo_dataset.yaml`).
* **`--epochs`** *(integer, default: `100`)*: The maximum number of training epochs to execute. Must be $\ge 1$.
* **`--batch`** *(integer, default: `16`)*: Mini-batch size per GPU. Adjust lower (e.g. `8`, `4`) on low-VRAM edge devices.
* **`--imgsz`** *(integer, default: `640`)*: Input resolution. Input images are resized to a square of `[imgsz x imgsz]`.
* **`--device`** *(string, default: `"0"`)*: CUDA device allocation. Supports `"0"`, `"1"` for multi-GPU setups, `"cpu"` to force CPU training, or `"0,1"` for DataParallel training.
* **`--patience`** *(integer, default: `20`)*: Number of epochs to wait without validation improvement before triggering early stopping.

#### Dynamic Hyperparameter Override Keys
* **`model_scale`** *(string, default: `"s"`)*: Swaps model capacity. Options: `n`, `s`, `m`, `l`, `x`.
* **`learning_rate`** *(float, default: `1e-4`)*: Overrides the optimizer's initial learning rate.
* **`lambda_traj`** *(float, default: `5.0`)*: Multiplier for future trajectory waypoint regression loss.
* **`lambda_det`** *(float, default: `1.0`)*: Multiplier for YOLO11 bounding box detection loss. Set to `0.0` to disable detection.
* **`lambda_heatmap`** *(float, default: `2.0`)*: Multiplier for saliency attention heatmap loss.
* **`lambda_cls`** *(float, default: `1.0`)*: Multiplier for navigation command classification loss.
* **`use_fdat`** *(boolean, default: `True`)*: Switch to enable Frenet-Decomposed Anisotropic Trajectory Loss.

#### Example Command
```bash
uv run neuropilot train neuro_pilot/cfg/models/neuralPilot.yaml \
    --data data/yolo_dataset.yaml \
    --epochs 150 --batch 32 --imgsz 320 --device 0 \
    model_scale=s learning_rate=5e-4 lambda_det=0.0 use_fdat=True
```

#### Realistic Console Output
```text
[INFO] Loading model config from: neuro_pilot/cfg/models/neuralPilot.yaml
[INFO] Scaling model depth and width using scale='s' (depth_multiple=0.33, width_multiple=0.50)
[INFO] Overriding: learning_rate=0.0005, lambda_det=0.0, use_fdat=True
[INFO] Building neural network graph:
  idx        from  n    params  module                                  arguments
    0          -1  1      3520  neuro_pilot.nn.modules.backbone.TimmBackbone  ['mobilenetv4_conv_medium', True, True]
    1          [0]  1         0  neuro_pilot.nn.modules.routing.FeatureRouter  [4]
   ...
   20    [14, 17]  1    120484  neuro_pilot.nn.modules.head.TrajectoryHead  [4, 10]
[INFO] Dataset parsed: 12,400 training samples | 2,100 validation samples.
[INFO] Detection task disabled (lambda_det = 0.0). Freezing object detection heads.
[INFO] Starting training for 150 epochs...
Epoch    GPU_mem   loss_traj  loss_heatmap  loss_cls   loss_total  waypoints_ADE  waypoints_FDE  time
1/150      3.42G      0.0845        0.1420    0.2012       0.4277         0.84m          1.52m  1m 24s
2/150      3.42G      0.0712        0.1189    0.1843       0.3744         0.71m          1.24m  1m 22s
```

---

### B. `predict` — Real-Time Inference
Executes inference on video files, image folders, or live hardware webcam feeds.

```bash
uv run neuropilot predict <SOURCE> --model <WEIGHTS_FILE> [OPTIONS] [KEY=VALUE ...]
```

#### Positional Arguments
* **`SOURCE`** *(string, required)*: Path to input. Can be an image path (`image.jpg`), directory of images (`images/`), video file (`video.mp4`), or video capture index (`0` for device `/dev/video0`).

#### Options
* **`--model`** *(string, required)*: Serialized weights file path. Supports PyTorch checkpoints (`best.pt`), static ONNX models (`neuro_pilot.onnx`), or TensorRT engines (`neuro_pilot.engine`).
* **`--conf`** *(float, default: `0.25`)*: Confidence threshold for plotting bounding boxes.
* **`--imgsz`** *(integer, default: `640`)*: Input resizing dimension.
* **`--save`** *(flag, default: `False`)*: Saves plotted frames and compiled `.mp4` outputs directly to `runs/predict/`.
* **`--stream`** *(flag, default: `False`)*: Displays predictions on-screen in real time using OpenCV window bindings.

#### Example Command
```bash
uv run neuropilot predict data/test_clip.mp4 \
    --model experiments/run1/weights/best.pt \
    --conf 0.35 --imgsz 320 --save
```

---

### C. `export` — Compile Graph to ONNX / TensorRT
Prepares PyTorch weights for low-latency edge deployment.

```bash
uv run neuropilot export --model <WEIGHTS_FILE> [OPTIONS]
```

#### Options
* **`--model`** *(string, required)*: Path to the target PyTorch `.pt` file.
* **`--format`** *(string, default: `"onnx"`)*: Desired edge runtime target. Options: `onnx` (ONNX Runtime) or `engine` (TensorRT engine binary).
* **`--imgsz`** *(integer, default: `640`)*: Static resolution to compile into the runtime graph.
* **`--dynamic`** *(flag, default: `False`)*: Configures the ONNX graph with dynamic axes support for batch dimension sizing.

#### Example Command
```bash
uv run neuropilot export --model best.pt --format engine --imgsz 320
```

---

### D. `val` — Quantitative Validation Sweep
Evaluates the model over the dataset's validation split to report accuracy and planning distance error profiles.

```bash
uv run neuropilot val --model <WEIGHTS_FILE> --data <DATASET_CONFIG> [OPTIONS]
```

#### Options
* **`--model`** *(string, required)*: Weights to evaluate (`.pt`, `.onnx`, or `.engine`).
* **`--data`** *(string, required)*: Path to the dataset configurations.
* **`--imgsz`** *(integer, default: `640`)*: Preprocessing resolution.

#### Example Command
```bash
uv run neuropilot val --model best.pt --data data/covla.yaml --imgsz 320
```

---

### E. `benchmark` — Hardware Speed Profile
Measures throughput speed metrics and memory footings under local hardware constraints.

```bash
uv run neuropilot benchmark --model <WEIGHTS_FILE> [OPTIONS]
```

#### Options
* **`--model`** *(string, required)*: Weights path.
* **`--imgsz`** *(integer, default: `640`)*: Resolution.
* **`--batch`** *(integer, default: `1`)*: Inference batch size.

#### Example Command
```bash
uv run neuropilot benchmark --model best.pt --imgsz 320 --batch 1
```

#### Realistic Console Output
```text
[INFO] Benchmarking: model=best.pt | imgsz=320 | batch_size=1
[INFO] Performing 100 warm-up runs...
[INFO] Running 1000 iteration benchmarks...
---------------------------------------------
Target Device: NVIDIA Jetson Orin Nano (8GB)
Model Scale:   S (Small)
Batch Size:    1
Resolution:    320 x 320
---------------------------------------------
Average Preprocess Latency:  1.24 ms
Average Inference Latency:   24.58 ms
Average Postprocess Latency: 1.08 ms
Total Latency (End-to-End):  26.90 ms
Throughput (Throughput):     37.17 FPS
Peak GPU Memory Allocated:   248.50 MB
---------------------------------------------
```
