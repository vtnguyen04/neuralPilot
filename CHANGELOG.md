# Changelog

All notable changes to NeuroPilot are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Layer-by-layer modular YAML model zoo (`neuro_pilot/cfg/models/zoo/`) with 16 architectures across planning (`pilot_s`, `pilot_l`, `pilot_nav`, `driver`), perception (`cipo`, `lane`, `drive`, `depth`, `seg`, `seg_l`), and world models (`jepa`, `vq_enc`, `vq_dec`, `ae`, `dit`, `gpt`).
- Generic `GraphModel` YAML executor (`neuro_pilot/nn/zoo/graph.py`) and shared atomic block library (`neuro_pilot/nn/zoo/blocks/`).
- Deterministic ONNX and safetensors weight converters (`neuro_pilot/nn/zoo/converters/`) and numerical parity harness (`neuro_pilot/nn/zoo/parity.py`), verified on 13 converted models including lightweight `seg` (`DDRNet23-Slim`, 5.7M) and `seg_l` (`EoMT-Large` DINOv2 ViT-L, 316M).
- Two-tier automatic checkpoint hub (`neuro_pilot/nn/zoo/hub.py`) and `neuropilot download` CLI caching pretrained `.pt` weights into `~/.cache/neuro_pilot/weights/` on first use (`NeuroPilot("pilot_l.pt")`, `NeuroPilot("seg.pt")`).
- Plug-and-play driving tasks (`planner`, `world`, `lane`, `depth`, `seg`, `driver`, `bfmc`) and stateful domain predictors (`neuro_pilot/engine/predictors/`).
- Multi-backend runtime auto-discovery for PyTorch `.pt`, ONNX Runtime `.onnx`, TorchScript `.torchscript`, and TensorRT 10 `.engine` (`neuro_pilot/engine/backend/`, `neuro_pilot/engine/trt_builder.py`).
- Standalone camera calibration, virtual view warp, streaming video inference (`NeuroPilot.predict_video`), 3D trajectory corridor, 3D multi-lane, CIPO corner-bracket, semantic segmentation, and depth visualizers (`neuro_pilot/utils/camera.py`, `viz.py`, `overlay.py`, `neuro_pilot/engine/video.py`).
- Unified demo rendering CLI (`neuro_pilot/tools/demo.py`) and MIT-licensed `comma2k19` demo clips in `assets/demos/`.
- Official Apache License 2.0 (`LICENSE`) and third-party license notices (`THIRD_PARTY_LICENSES.md`).

### Changed
- Standardized zoo configs, modules, tests, and demo assets to short 1-2 word domain names while keeping backward-compatible aliases in `MODEL_YAML_MAP` and `PREDICTOR_REGISTRY`.
- Updated `pyproject.toml` license metadata to Apache-2.0 and replaced the README and icon assets.

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
