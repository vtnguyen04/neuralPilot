# BFMC 1:10 Experiment Reports (`pilot_mhp`)

V1 and V2 are documented separately because they use different waypoint densities (`10` vs `33`), trajectory heads (`argmax` vs differentiable MoE), and validation splits (`313` vs strict 3-way `512 val / 352 test`).

- **[`REPORT_V2.md`](REPORT_V2.md) — Current Release (`pilot_mhp_v2.pt`, 33-pt End-to-End MoE):** Full architecture math, pipeline, loss formulation, `EXP-029` → `EXP-034` ablations, Val (512) + Untouched Test (352) metrics, CDF, and ONNX parity.
- **[`REPORT_V1.md`](REPORT_V1.md) — Legacy Baseline & Edge TRT (`pilot_mhp.pt`, 10-pt MHP):** Baseline through V6 smoothness progression (`EXP-016` → `EXP-024`) plus TensorRT 10 & Jetson Orin Nano BF16 deployment (`EXP-025` → `EXP-028`).

## Side-by-Side Overview

| Item | V1 (`REPORT_V1.md`) | V2 (`REPORT_V2.md`) |
|---|---|---|
| Checkpoint | `pilot_mhp.pt` (`53.8 MB`, `14.07M`) | **`pilot_mhp_v2.pt`** (`64.25 MB`, `16.79M`) |
| YAML Scale | `pilot_mhp.yaml` (`scale: v1`) | `pilot_mhp.yaml` (`scale: v2`) |
| Waypoints | `10` points | **`33` C² B-spline points** |
| Trajectory Head | PANet-coupled MHP + discrete `argmax` | **Decoupled backbone + `64×80` P3 + Differentiable MoE + `refine_out`** |
| Dataset & Split | `1,491` frames (`1,178` train / `313` val) | [`bfmc-33wp`](https://huggingface.co/datasets/ThanhNguyenVo/bfmc-33wp) (`3,361` train / `512` val / `352` untouched test, `dHash >= 7`) |
| Val Metrics | `Score 54.46`, `MSE 0.00440`, `Lat 18.5px`, `mAP@50 63.2%` | **`Score 66.53`, `MSE 0.00223` (`traj: 0.00210`), `Lat 13.8px`, `mAP@50 66.3%`** |
| Untouched Test | — | **`Score 74.42`, `MSE 0.00146`, `Lat 13.2px`, `mAP@50 81.0%`** *(864-frame combined MSE `0.00192`)* |
| ONNX & Speed | `54.0 MB`, `1e-6` diff, `59 FPS` RTX 3060 / `12.4 FPS` CPU | **`64.5 MB`, `2.98e-7` diff, `63.5 FPS` RTX 3060 / `18.8 FPS` CPU** |
| Weights Repo | [`ThanhNguyenVo/neuropilot-weights`](https://huggingface.co/ThanhNguyenVo/neuropilot-weights) | [`ThanhNguyenVo/neuropilot-weights`](https://huggingface.co/ThanhNguyenVo/neuropilot-weights) |
