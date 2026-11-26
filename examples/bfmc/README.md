# BFMC 1:10 Driving Example (`pilot_mhp`)

Single-camera multi-task model (`FastViT-T12` + PANet `DFLDetectHead` + 5-mode `TrajMHPHead`).
Full dataset breakdown, training stages, experiment tables, and plots are in [REPORT.md](REPORT.md).

## Quick Inference (Auto-Downloads Weights)

```python
from neuro_pilot import NeuroPilot

model = NeuroPilot("pilot_mhp.pt")  # downloads automatically on first run
out = model.predict(frame_bgr, command=0)  # 0=FOLLOW_LANE, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STRAIGHT
waypoints, detections = out["waypoints"], out["detections"]
```

## Fine-Tune on Your Own Data

Passing a bare checkpoint name (`pilot_s.pt` or `pilot_mhp.pt`) downloads weights automatically.

```bash
# Train from OpenPilot pilot_s backbone
python3 -m examples.bfmc.train \
  --arch pilot_mhp \
  --pilot-s-pt pilot_s.pt \
  --train-json train.json \
  --val-json val.json \
  --save-dir runs/bfmc

# Fine-tune from pretrained BFMC checkpoint
python3 -m examples.bfmc.train \
  --arch pilot_mhp \
  --resume-weights pilot_mhp.pt \
  --train-json train.json \
  --val-json val.json \
  --epochs 25 --freeze-epochs 0 \
  --lr-backbone 5e-5 --lr-head 4e-4 \
  --save-dir runs/bfmc_ft
```

## Video Demo & ONNX Export

```bash
python3 -m examples.bfmc.demo \
  --ckpt pilot_mhp.pt \
  --video bfmc.mp4 --out-mp4 demo.mp4 \
  --export-onnx pilot_mhp.onnx
```
