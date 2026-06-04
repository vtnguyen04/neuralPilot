---
sidebar_position: 2
---
# Jetson Inference & Runtime

After compiling your TensorRT engine, you can run inference using our optimized edge runtimes. This guide covers preprocessing, execution contexts, and troubleshooting edge memory issues.

---

## 1. Under the Hood: GPU Memory Bindings

Executing inference on TensorRT requires mapping inputs and outputs directly to GPU memory spaces.

The runtime script `examples/jetson_inference_trt.py` handles this mapping:
1. **Engine Deserialization**: Loads the binary `.engine` file into memory.
2. **Execution Context**: Allocates an asynchronous execution space.
3. **GPU Allocation**: Reserves host (CPU) and device (GPU) memory buffers for both input images (`[1, 3, H, W]`) and task outputs (bounding boxes, planned waypoints, attention heatmaps).
4. **Execution**: Binds pointers to the CUDA context and launches GPU kernels:
   ```python
   self.context.execute_v2(bindings=self.bindings)
   ```

---

## 2. Launching Inference on Jetson

### Method A: ONNX Runtime (CUDA Execution Provider)
To run ONNX models directly using ONNX Runtime:

```bash
PYTHONPATH=. uv run python examples/jetson_inference_onnx.py \
    --model neuro_pilot_model.onnx \
    --source path/to/video.mp4 \
    --imgsz 320 \
    --save
```

### Method B: Native TensorRT Engine
For the fastest possible execution, use the native TensorRT bindings:

```bash
PYTHONPATH=. uv run python examples/jetson_inference_trt.py \
    --engine neuro_pilot_model.engine \
    --source path/to/video.mp4 \
    --imgsz 320 \
    --save
```

---

## 3. Troubleshooting Edge Failures

### Issue A: Out of Memory (OOM) during Compilation
TensorRT compilation is highly memory-intensive and can crash on 4GB/8GB Orin Nano boards.
* **Mitigation**:
  * Allocate a minimum of **8GB swap space** on your Jetson eMMC/NVMe.
  * Stop the desktop display GUI (Desktop Manager) before compiling to free VRAM:
    ```bash
    sudo systemctl stop gdm3
    ```
  * Compile using `trtexec` with a restricted workspace size to force lower memory usage:
    ```bash
    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --workspace=2048
    ```

### Issue B: Conversion Failure due to Unsupported Layers
If you attempt to convert models using custom `timm` attention layers, the compiler may crash with an `UNSUPPORTED_NODE` error.
* **Mitigation**: Switch to the native YOLO11 Conv configuration: `neuralPilot_yolo11.yaml`. This model uses standard convolutional blocks that are 100% supported by TensorRT.
