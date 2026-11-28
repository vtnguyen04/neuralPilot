# BFMC 1:10 Driving Example (`pilot_mhp_v2`)

Single-camera multi-task driving model (`FastViT-T12` + PANet `DFLDetectHead` + 33-waypoint End-to-End MoE `TrajMHPHead`).

- **V2 Report (Latest, 33-pt MoE, `EXP-029` → `EXP-034`):** [REPORT_V2.md](REPORT_V2.md)
- **V1 Report (Legacy 10-pt + Jetson Orin BF16 TRT, `EXP-016` → `EXP-028`):** [REPORT_V1.md](REPORT_V1.md)
- **Side-by-Side Index:** [REPORT.md](REPORT.md)

- **Weights (HuggingFace):** [`ThanhNguyenVo/neuropilot-weights`](https://huggingface.co/ThanhNguyenVo/neuropilot-weights) (`pilot_mhp_v2.pt`, `pilot_mhp_v2_traj.pt`, `pilot_mhp_v2.onnx`; legacy 10-pt `pilot_mhp.pt` kept intact)
- **Dataset (HuggingFace):** [`ThanhNguyenVo/bfmc-33wp`](https://huggingface.co/datasets/ThanhNguyenVo/bfmc-33wp) (`4,225` frames, 3-way zero-leakage split, `dHash >= 7`)
- **Labeler Tool:** [`vtnguyen04/neuro_pilot_labeler`](https://github.com/vtnguyen04/neuro_pilot_labeler) (33-waypoint C² B-spline & 13-class studio used to annotate and export the dataset)

## V1 vs V2 Summary

| Version | Waypoints | Params | Trajectory Head | Val MSE / `mAP@50` | Test MSE / `mAP@50` | RTX 3060 E2E |
|---|---:|---:|---|---:|---:|---:|
| `pilot_mhp.pt` (`v1`) | `10` | `14.07M` | PANet-coupled MHP + discrete `argmax` | `0.00440` / `63.2%` *(313 val)* | — | `59.0 FPS` |
| **`pilot_mhp_v2.pt` (`v2`)** | **`33`** | **`16.79M`** | **Decoupled backbone + `64×80` P3 + Differentiable MoE** | **`0.00223` / `66.3%`** *(512 val)* | **`0.00146` / `81.0%`** *(352 test)* | **`63.5 FPS`** |

## 1. Quick Inference & Visualization (Auto-Downloads Weights)

```python
from neuro_pilot import NeuroPilot

model = NeuroPilot("pilot_mhp_v2.pt")  # downloads automatically on first run
out = model.predict(frame_bgr, command=0)  # 0=FOLLOW_LANE, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STRAIGHT
vis = model.visualize(frame_bgr, out, command=0)  # smooth 2D corridor + 13-class boxes + HUD
```

You can also run training, validation, and export straight from Python without any CLI wrapper:

```python
model = NeuroPilot("pilot_mhp", scale="v2")
model.train(train_json="datasets/bfmc_33wp/train.json", val_json="datasets/bfmc_33wp/val.json", save_dir="runs/bfmc")
metrics = model.val(val_json="datasets/bfmc_33wp/test.json", save_grid="test_grid.jpg")
model.export(format="onnx", file="pilot_mhp_v2.onnx", opset=17)
```

## 2. Export Clean 3-Way Dataset (or Download from HuggingFace)

Download the frozen dataset directly from [`ThanhNguyenVo/bfmc-33wp`](https://huggingface.co/datasets/ThanhNguyenVo/bfmc-33wp), or export from a local [NeuroPilot Labeler](https://github.com/vtnguyen04/neuro_pilot_labeler) database with zero-leakage `dHash` filtering:

```bash
python3 examples/bfmc/export_dataset.py \
  --db data/labeler.db \
  --out-dir datasets/bfmc_33wp \
  --num-waypoints 33 --dhash-gap 6
```

## 3. Reproduce V2 Training (`EXP-033` + `EXP-034`)

Passing a bare checkpoint name (`pilot_s.pt` or `pilot_mhp_v2.pt`) downloads weights automatically.

```bash
# Stage 1: End-to-End MoE from OpenPilot pilot_s.pt (40 epochs, BF16 + DINOv3-B REPA)
python3 examples/bfmc/train.py \
  --arch pilot_mhp --scale v2 \
  --pilot-s-pt pilot_s.pt \
  --train-json datasets/bfmc_33wp/train.json \
  --val-json datasets/bfmc_33wp/val.json \
  --epochs 40 --freeze-epochs 3 --batch-size 8 \
  --lr-backbone 2e-4 --lr-head 1.2e-3 \
  --affine-prob 0.65 --copy-paste 0.55 --polish-epochs 0 \
  --distill --distill-weight 1.5 --save-dir runs/bfmc_moe

# Stage 2: Anti-overfit smoothness + detection polish (12 epochs, strong aug kept)
python3 examples/bfmc/train.py \
  --arch pilot_mhp --scale v2 \
  --resume-weights runs/bfmc_moe/best_traj.pt \
  --train-json datasets/bfmc_33wp/train.json \
  --val-json datasets/bfmc_33wp/val.json \
  --epochs 12 --freeze-epochs 0 --batch-size 8 \
  --lr-backbone 4e-5 --lr-head 3e-4 \
  --affine-prob 0.65 --copy-paste 0.55 --polish-epochs 0 \
  --det-weight 1.65 --smooth-weight 2.2 \
  --distill --distill-weight 1.2 --save-dir runs/bfmc_moe_final
```

## 4. Full-Video Demo, ONNX / TensorRT Export & Jetson Deploy

```bash
python3 examples/bfmc/demo.py \
  --ckpt pilot_mhp_v2.pt \
  --val-json datasets/bfmc_33wp/test.json --out-grid test_grid.jpg \
  --video bfmc.mp4 --out-mp4 bfmc_v2_demo.mp4 --max-frames 0 --stride 1 \
  --export-onnx pilot_mhp_v2.onnx
```

Build a local TensorRT BF16 engine or deploy directly to a Jetson Orin over SSH (exports ONNX, uploads via `scp`, and compiles the native `.engine` on Jetson):

```bash
# Local TensorRT BF16 engine (run on the target GPU)
neuropilot export --model pilot_mhp_v2.pt --format engine --file pilot_mhp_v2.engine --bf16 --workspace-mb 1024

# One-command Jetson Orin deploy over SSH
neuropilot deploy \
  --model pilot_mhp_v2.pt \
  --host user@<jetson-ip> \
  --remote-dir ~/BFMC2026/Code/Brain/models \
  --precision bf16 --workspace-mb 1024
```

```python
from neuro_pilot import NeuroPilot

NeuroPilot("pilot_mhp_v2.pt").deploy(
    host="user@<jetson-ip>",
    remote_dir="~/BFMC2026/Code/Brain/models",
    precision="bf16",
    workspace_mb=1024,
)
```
