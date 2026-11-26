# BFMC 1:10 Multi-Task Autonomous Driving Example (`pilot_mhp`)

End-to-end training, evaluation, video rendering, and ONNX export example built on top of `neuro_pilot`'s generic YAML Zoo (`pilot_mhp.yaml` / `pilot_det.yaml`), reusing core `DFLDetectHead`, `CSPBlock`, `TrajMHPHead`, `DetectionLoss`, `ModelEMA`, `FDATLoss`, `HeatmapLoss`, `MHPLaplaceLoss`, and `neuro_pilot.nn.zoo.distill`.

Full experimental methodology, multi-stage training protocol, ablations (`EXP-MAIN-016` → `EXP-MAIN-024`), plots, and ONNXRuntime parity benchmarks are documented in **[REPORT.md](REPORT.md)**.

## 1. Train (`pilot_mhp` from OpenPilot `pilot_s.pt` + DINOv3 REPA)

```bash
python3 -m examples.bfmc.train \
  --arch pilot_mhp \
  --train-json datasets/bfmc_clean/train.json \
  --val-json datasets/bfmc_clean/val.json \
  --unlabeled-json datasets/bfmc_clean/unlabeled.json \
  --pilot-s-pt ~/.cache/neuro_pilot/weights/pilot_s.pt \
  --dinov3-path ~/.cache/neuro_pilot/weights/dinov3_vitb16.safetensors \
  --save-dir runs/bfmc_mhp \
  --img-w 640 --img-h 512 --epochs 60 --distill
```

## 2. Evaluate, Render Demo Video & Export ONNX

```bash
python3 -m examples.bfmc.demo \
  --ckpt runs/bfmc_v6_smooth_pure_mhp/best.pt \
  --val-json datasets/bfmc_clean/val.json \
  --out-grid val_grid.jpg \
  --video bfmc.mp4 --out-mp4 demo.mp4 \
  --export-onnx runs/bfmc_v6_smooth_pure_mhp/bfmc_v6_best.onnx
```

## 3. Unified Inference (`NeuroPilot` PyTorch or ONNX auto-detect)

```python
from neuro_pilot import NeuroPilot

# Works identically for .pt and .onnx checkpoints
model = NeuroPilot("runs/bfmc_v6_smooth_pure_mhp/bfmc_v6_best.onnx", device="cpu")
out = model.predict(frame_bgr, command=0)  # 0=FOLLOW_LANE, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STRAIGHT
waypoints, detections = out["waypoints"], out["detections"]
```
