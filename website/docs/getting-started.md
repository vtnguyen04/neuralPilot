---
sidebar_position: 2
---
# Getting Started

This guide provides a comprehensive setup walk-through for **NeuroPilot**. We cover system prerequisites, multi-environment installation (using `uv`, native `pip` virtual environments, or Docker), CLI operations, validation procedures, and basic programmatic integration.

---

## 1. Prerequisites & System Requirements

Before installing, verify that your host system or target edge device meets the following compatibility requirements:

### Operating System & Hardware Compatibility
* **Host PC / Server**: Linux (Ubuntu 20.04/22.04 LTS recommended), Windows 10/11, or macOS.
* **Edge Platform**: NVIDIA Jetson Orin Series (Orin Nano, Orin NX, AGX Orin) running JetPack 6.x (recommended) or JetPack 5.x.
* **GPU**: NVIDIA GPU with CUDA Compute Capability 6.1+ (Pascal architecture or newer). Required for accelerated training and real-time TensorRT inference.

### Core Software Stack
* **Python**: `python >= 3.10` and `python < 3.12` (Python 3.10 is the heavily tested baseline).
* **CUDA Toolkit**: Version `11.8` or `12.1+`.
* **CUDNN**: Compatible version mapping your CUDA setup.
* **Package Manager**: [uv](https://github.com/astral-sh/uv) (strongly recommended for fast resolution) or `pip` / `conda`.

---

## 2. Installation Methods

Choose the installation method that matches your development workflow.

### Method A: Ultra-Fast Setup with `uv` (Recommended)
`uv` is a blazing-fast Python package resolver. It manages virtual environments and lockfiles in milliseconds.

1. **Install `uv` globally**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone the repository**:
   ```bash
   git clone https://github.com/vtnguyen04/neuralPilot.git
   cd neuralPilot
   ```
3. **Sync the workspace**:
   This automatically creates a `.venv` directory and installs all dependencies specified in `uv.lock`:
   ```bash
   uv sync
   ```
4. **Developer Sync** (Installs testing, linting, and quality tools):
   ```bash
   uv sync --group dev
   ```

### Method B: Classic Virtual Environment with `pip`
If you prefer not to use `uv`, you can install dependencies using standard Python tools:

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Upgrade base packaging tools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```
3. **Install dependencies**:
   Install the package in editable mode with development extras:
   ```bash
   pip install -e .[dev]
   ```

---

## 3. Verifying the Installation

After installing, verify that the environment loads correctly and the CLI commands are accessible.

### Verify CLI Tool
Run the help command using the `neuropilot` entrypoint:
```bash
uv run neuropilot --help
```
Expected output:
```text
Usage: neuropilot [OPTIONS] COMMAND [ARGS]...

  NeuroPilot: Unified End-to-End Autonomous Driving Framework CLI.

Options:
  --help  Show this message and exit.

Commands:
  train      Train a model from config.
  predict    Run inference on images, video, or webcam.
  export     Convert PyTorch weights (.pt) to ONNX or TensorRT.
  val        Validate model against dataset metrics.
  benchmark  Measure latency, throughput (FPS), and memory utilization.
```

### Verify PyTorch and CUDA bindings
Run a quick Python check to ensure PyTorch is compiled with CUDA support:
```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
```

---

## 4. End-to-End Quick Start

Here is a step-by-step guide to run training, validation, export, and inference.

### Step 1: Prepare a Dataset
Create a dataset folder matching the YOLO-perception / Waypoints format (see the Custom Dataset Guide for details). For quick testing, you can write a simple YAML file:
```yaml
# data/toy_dataset.yaml
path: ./data/toy
train: images/train
val: images/val
names:
  0: obstacle
  1: vehicle
```

### Step 2: Model Training
Train the model using the standard MobileNetV4 backbone config:
```bash
uv run neuropilot train neuralPilot.yaml --data data/toy_dataset.yaml --epochs 10 --batch 8 --imgsz 320
```

* **Dynamic Overrides**: You can pass hyperparameters as `key=value` CLI overrides:
  ```bash
  # Train a Trajectory-only model (disable object detection and gaze heatmaps)
  uv run neuropilot train neuralPilot.yaml \
      --data data/toy_dataset.yaml \
      lambda_det=0.0 lambda_heatmap=0.0 use_fdat=True
  ```

### Step 3: Model Validation
Calculate accuracy metrics (Precision, Recall, mAP50, Trajectory ADE/FDE, Heatmap MSE) on the validation set:
```bash
uv run neuropilot val --model best.pt --data data/toy_dataset.yaml --imgsz 320
```

### Step 4: Model Exporting
To deploy on edge hardware, export your weights to ONNX format:
```bash
uv run neuropilot export --model best.pt --format onnx --imgsz 320 model_scale=s
```

### Step 5: Run Inference
Run inference on a video or webcam feed:
```bash
uv run neuropilot predict path/to/video.mp4 --model best.pt --save --imgsz 320
```
Your processed outputs, showing visual boxes, planned curves, and spatial saliency heatmaps, will be saved in `runs/predict/`.

---

## 5. Troubleshooting Installation Issues

### Issue A: Timm or PyTorch dependency conflict
If `uv sync` fails due to locked dependencies:
* Ensure you are running Python 3.10, which matches the exact dependencies in `uv.lock`.
* If you must use a different Python version, clear the lockfile and regenerate:
  ```bash
  rm uv.lock
  uv pip compile pyproject.toml -o uv.lock
  uv sync
  ```

### Issue B: CUDA Out-Of-Memory (OOM) during training
If you experience OOM on GPUs with < 8GB VRAM:
* Reduce `--batch` size to `8` or `4`.
* Set `--imgsz` to `320`.
* Decrease the model scale to `n` (nano): `model_scale=n` in CLI overrides.
* Disable memory-heavy heads like JEPA by setting `lambda_jepa=0.0`.
