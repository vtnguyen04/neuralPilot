---
sidebar_position: 1
---
# Model Export (ONNX & TensorRT)

Deploying multi-task neural networks on edge robotics platforms requires converting PyTorch graphs (`.pt`) to optimized formats compatible with low-latency runtimes.

---

## 1. Exporting to ONNX

ONNX (Open Neural Network Exchange) acts as the universal bridge. Exporting parses the PyTorch computational graph into a static node representation.

Run the export command:
```bash
uv run neuropilot export --model best.pt --format onnx --imgsz 320 model_scale=s
```

### Key Parameters
* **`--model`**: Path to the PyTorch `.pt` file containing state dictionary weights.
* **`--imgsz`**: Spatial resolution of the input tensors. Resolving features at lower resolutions (e.g. `320` instead of `640`) dramatically improves latency at the cost of tiny obstacle detection precision.
* **`--dynamic`**: Flag to export with dynamic batch sizes (e.g. `[Batch, 3, H, W]`). For edge deployment on NVIDIA Jetson, it is recommended to keep batch size fixed at `1` to avoid the overhead of dynamic shape allocation.

---

## 2. Compiling to TensorRT Engine

For maximum speed on Nvidia GPUs and Jetson modules, compile the ONNX representation into a native TensorRT engine (`.engine`).

Run the command:
```bash
uv run neuropilot export --model best.pt --format engine --imgsz 320 model_scale=s
```

> [!IMPORTANT]
> **Compilation Location**: TensorRT engines are compiled targeting specific GPU microarchitectures and CUDA cores. You **must** run the compilation step directly on the target Jetson hardware (e.g., Jetson Orin Nano).

### Optimization Techniques Applied Under the Hood
* **FP16 Quantization**: Automatically converts all FP32 weights to FP16 half-precision. This reduces model footprint in VRAM by 50% and leverages Tensor Cores for accelerated computation.
* **Kernel Fusion**: Combines adjacent operations (such as Convolution + Batch Normalization + ReLU) into single execution kernels, reducing memory read/write cycles.
* **Optimal Workspace Selection**: Benchmark algorithms to choose the fastest CUDA kernel layouts for your specific hardware platform.
