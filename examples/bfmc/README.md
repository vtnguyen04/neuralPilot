# BFMC 1:10 Driving Example (`pilot_mhp_v2`)

Single-camera multi-task model (`FastViT-T12` + PANet `DFLDetectHead` + 33-waypoint End-to-End MoE `TrajMHPHead`).
Full dataset audit, training stages, experiment log, and plots are in [REPORT.md](REPORT.md).

- **Weights:** [`ThanhNguyenVo/neuropilot-weights`](https://huggingface.co/ThanhNguyenVo/neuropilot-weights) (`pilot_mhp_v2.pt`, `pilot_mhp_v2_traj.pt`, `pilot_mhp_v2.onnx`)
- **Dataset:** [`ThanhNguyenVo/bfmc-33wp`](https://huggingface.co/datasets/ThanhNguyenVo/bfmc-33wp) (`4,225` frames, 3-way zero-leakage split)

## Quick Inference (Auto-Downloads Weights)

```python
from neuro_pilot import NeuroPilot

model = NeuroPilot("pilot_mhp_v2.pt")  # downloads automatically on first run
out = model.predict(frame_bgr, command=0)  # 0=FOLLOW_LANE, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STRAIGHT
waypoints, detections = out["waypoints"], out["detections"]
```

## Fine-Tune on Your Own Data

Passing a bare checkpoint name (`pilot_s.pt` or `pilot_mhp_v2.pt`) downloads weights automatically.

```bash
# Train from OpenPilot pilot_s backbone (33-waypoint v2 MoE)
python3 -m examples.bfmc.train \
  --arch pilot_mhp --scale v2 \
  --pilot-s-pt pilot_s.pt \
  --train-json train.json \
  --val-json val.json \
  --save-dir runs/bfmc_moe

# Anti-overfit polish from pretrained BFMC v2 checkpoint
python3 -m examples.bfmc.train \
  --arch pilot_mhp --scale v2 \
  --resume-weights pilot_mhp_v2.pt \
  --train-json train.json \
  --val-json val.json \
  --epochs 12 --freeze-epochs 0 \
  --lr-backbone 4e-5 --lr-head 3e-4 --lr-traj 1.2e-4 \
  --save-dir runs/bfmc_ft
```

## Video Demo & ONNX Export

```bash
python3 -m examples.bfmc.demo \
  --ckpt pilot_mhp_v2.pt \
  --video bfmc.mp4 --out-mp4 demo.mp4 \
  --export-onnx pilot_mhp_v2.onnx
```
