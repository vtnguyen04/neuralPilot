# Changelog

All notable changes to NeuroPilot are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-11-28

### Added
- Layer-by-layer modular YAML model zoo (`neuro_pilot/cfg/models/zoo/`) with 18 architectures across planning (`pilot_s`, `pilot_l`, `pilot_nav`, `pilot_mhp`, `pilot_det`, `driver`), perception (`cipo`, `lane`, `drive`, `depth`, `seg`, `seg_l`), and world models (`jepa`, `vq_enc`, `vq_dec`, `ae`, `dit`, `gpt`).
- Generic `GraphModel` YAML executor (`neuro_pilot/nn/zoo/graph.py`) and shared atomic block library (`neuro_pilot/nn/zoo/blocks/`), including single-camera trajectory decoders (`YuvSinglePrep`, `TrajHead`, `TrajMHPHead` with `scale: v2` 33-waypoint differentiable MoE and decoupled backbone taps).
- Unified per-task package under `neuro_pilot/tasks/` (`base.py`, `multitask.py`, `planner.py`, `perception.py`, `world.py`) with per-task `get_trainer()`, `get_validator()`, and `visualize()`.
- Core multi-task planner training, evaluation, and dataset modules: `PlannerDataset` (`@register_dataset("planner_json")` in `neuro_pilot/data/datasets/planner.py`), `PlannerTrainer` (`neuro_pilot/engine/planner_trainer.py`), `PlannerValidator` (`neuro_pilot/engine/planner_eval.py`), `planner_metrics.py`, and `planner_loss.py`.
- Unified `RunMonitor` (`neuro_pilot/engine/monitor.py`) logging dynamic CSV, JSONL, GPU VRAM, throughput (`img/s`), ETA, structured console metrics, and auto-plotted `training_curves.png`.
- Task-aware visualizers (`neuro_pilot/utils/visualize.py`) wired into `NeuroPilot.visualize()`, featuring continuous-normal 2D trajectory corridor rendering (zero sharp-turn twist).
- TensorRT 10 `--bf16` precision support, `workspace_mb`, and JetPack `/usr/src/tensorrt/bin/trtexec` auto-detection in `neuro_pilot/engine/trt_builder.py` and `Exporter`.
- One-command Jetson SSH deployment (`neuro_pilot/deploy/jetson.py`, `NeuroPilot.deploy()`, and `neuropilot deploy`) that exports ONNX, uploads via SCP, and compiles the native TensorRT `.engine` directly on the target Jetson GPU.
- Training-only DINOv3 dense feature distillation (`DinoV3Teacher`, `DistillProj`, `dense_distill_loss` in `neuro_pilot/nn/zoo/distill.py`).
- Automatic checkpoint hub (`neuro_pilot/nn/zoo/hub.py`) and `neuropilot download` CLI fetching 16 pretrained `.pt` weights (`pilot_mhp_v2.pt`, `pilot_mhp.pt`, `pilot_det.pt`, and 13 upstream checkpoints) from Hugging Face (`ThanhNguyenVo/neuropilot-weights`) on first use.
- Deterministic ONNX and safetensors weight converters (`neuro_pilot/nn/zoo/converters/`) and numerical parity harness (`neuro_pilot/nn/zoo/parity.py`).
- Multi-backend runtime auto-discovery for PyTorch `.pt`, ONNX Runtime `.onnx`, TorchScript `.torchscript`, and TensorRT 10 `.engine`.
- Standalone user examples (`examples/planning.py`, `examples/perception.py`, `examples/world_model.py`, `examples/deploy_jetson.py`, and thin `examples/bfmc/{train,demo,export_dataset}.py`) plus separate V1 and V2 BFMC reports (`REPORT_V1.md`, `REPORT_V2.md`, `REPORT.md`).
- Standalone camera calibration, virtual view warp, streaming video inference (`NeuroPilot.predict_video`), 3D trajectory corridor, 3D multi-lane, CIPO corner-bracket, segmentation, and depth overlays (`neuro_pilot/utils/{camera,viz,overlay}.py`, `neuro_pilot/engine/video.py`, `neuro_pilot/tools/demo.py`).
- Official Apache License 2.0 (`LICENSE`), third-party license notices (`THIRD_PARTY_LICENSES.md`), and strict `neuro_pilot`-only packaging (`MANIFEST.in`).
- Comprehensive unit, E2E, packaging, and clean-code guardrail tests (`tests/engine/test_tasks_and_monitor.py`, `tests/engine/test_deploy_jetson.py`, `tests/zoo/`).

### Changed
- Refactored `examples/bfmc/` to thin standalone scripts using the public `NeuroPilot` API (`model.train()`, `model.val()`, `model.predict()`, `model.visualize()`, `model.export()`, `model.deploy()`) and removed `examples/bfmc/__init__.py`, `dataset.py`, and `metrics.py`.
- Normalized `MultitaskPredictor` detection outputs to `[0, 1]` coordinates for resolution-independent rendering.
- Standardized zoo configs, modules, tests, and demo assets to short 1-2 word domain names while keeping backward-compatible aliases in `MODEL_YAML_MAP` and `PREDICTOR_REGISTRY`.
- Updated `pyproject.toml` to SPDX `license = "Apache-2.0"` and explicit setuptools package exclusions (`tests*`, `examples*`, `website*`, `scripts*`, `tools*`) so `pip install git+...` installs only `neuro_pilot`.
- Upgraded `.github/workflows/ci.yml` and `.github/workflows/deploy-docs.yml` with `uv` caching, pinned `ruff==0.15.7`, system media libraries (`libgl1`, `ffmpeg`), concurrency groups, manual triggers, and docs typecheck/build verification.
- Updated `neuro_pilot/tools/demo.py` to fall back to Hugging Face auto-download when `--pt_root` is omitted.

### Fixed
- Fixed `is_zoo_yaml()` in `neuro_pilot/nn/zoo/graph.py` misclassifying native `neuralPilot.yaml` via the shared `Concat` block name and skipping `register_all_blocks()` when the registry was partially populated.
- Fixed `GraphModel.scale` persistence so non-default scale checkpoints (such as `scale="v2"`) reload cleanly via `NeuroPilot(path)`.
- Fixed `infer_driving_task()` mapping for `vq_enc` and `vq_dec`, restored `colorstr()` in `neuro_pilot/utils/logger.py`, and added `dict` checkpoint handling in `MultiTask.load_weights()`.
- Fixed bare `<key>.yaml` loading in `maybe_build_zoo_graph` and removed PyTorch ONNX `float64` tracing cast in `TrajMHPHead`.

## [1.3.2] - 2026-06-14

### Added
- Overwrite warnings in `register_head`, `register_loss`, and `register_neck` with unit tests in `tests/core/test_registry.py`.

### Fixed
- Fixed `NameError` in `Predictor._prepare_command` when `cmd` was passed explicitly.
- Fixed repeated TensorRT I/O buffer name appending in `TensorRTBackend._allocate_buffers` during dynamic shape reallocation.
- Removed false-positive PyTorch LR scheduler step warnings on training resume.

### Performance
- Vectorized target grid generation in `HeatmapLoss.generate_heatmap` and added `(H, W, device)` meshgrid caching.

## [1.3.1] - 2026-04-20

### Added
- `WingLoss` for waypoint regression and dynamic `smooth_l1` / `l2` / `wing` switching in `MultiTaskLossManager`.
- `scripts/benchmark_losses.py` for ADE/FDE comparison across trajectory loss variants.

### Fixed
- Fixed FDAT anisotropic normalization (`(alpha + beta) / 2`), gate gradient flow, and default backbone naming.
- Applied `ruff` cleanup and type-hint fixes across `neuro_pilot/utils/losses.py` and `neuro_pilot/cfg/schema.py`.

## [1.3.0] - 2026-04-16

### Added
- Temporal video dataset interfaces (`BaseVideoDataset`, `VideoDrivingDataset`) using Decord with `clip_length`, `frame_stride`, and temporal jitter/dropout.
- Multi-frame temporal aggregators (`ConcatAggregator`, `TemporalAttentionAggregator`, `GRUAggregator`) and `TemporalTrajectoryHead`.
- `neuralPilot_video.yaml`, `examples/train_temporal.py`, and temporal unit tests (`tests/models/test_temporal.py`, `tests/data/test_video_dataset.py`).

### Changed
- Split monolithic `lewm_modules.py` into `transformer.py`, `mlp.py`, `predictor.py`, `regularization.py`, and `jepa.py`.
- Updated `Trainer`, `Validator`, losses, and metrics to handle 5D temporal clips (`[B, T, C, H, W]`).

### Fixed
- Fixed batch-size scaling (`* proj.size(-2)`) inside `SIGReg` in `neuro_pilot/nn/modules/regularization.py`.

### Removed
- Removed deprecated single-frame CoVLA dataset scripts (`covla_hf.py`, `covla_local.py`), `marker_detection.py`, and obsolete CoVLA test scripts.

## [1.2.1] - 2026-04-15

### Fixed
- Cleaned trailing whitespace and PEP 8 lint violations across 22 files for CI stability.

## [1.2.0] - 2026-04-15

### Added
- `@register_dataset` registry pattern in `neuro_pilot/core/registry.py`.
- Causal Feature Routing (`neuralPilot_cfr.yaml`), deformable cross-attention trajectory head (`neuralPilot_deformable.yaml`), and JEPA predictor (`jepa.py`).

### Changed
- Switched default ONNX runtime dependency to `onnxruntime-gpu`.

## [1.1.6] - 2026-04-14

### Fixed
- Fixed waypoint clumping in `neuro_pilot/utils/plotting.py` using Catmull-Rom spline spacing.

### Changed
- Locked dependency versions in `uv.lock`.

## [1.0.0] - 2026-03-01

### Added
- Initial NeuroPilot multi-task framework combining trajectory prediction (cubic Bezier head with FiLM command conditioning), anchor-free object detection, attention heatmaps, and command classification.
- YAML-configurable model scaling (`n`, `s`, `m`, `l`, `x`) with pluggable `timm` and Conv backbones.
- PyTorch training engine (`Trainer` with AMP, EMA, DDP), YOLO-format driving dataset loader, and ONNX / TensorRT export.
