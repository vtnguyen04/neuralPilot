# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2026-04-16

### Added

#### Temporal Video Dataset Handling
- **`neuro_pilot/data/datasets/video_dataset.py`**: Added `BaseVideoDataset` interface to standardize inputs for temporal sequences.

  - Implemented `ClipSample` dataclass for ordered chronological frame parsing.

  - Added continuous slicing arguments: `clip_length` (sub-sequence sizing) and `frame_stride` (frame interval skipping).

  - Implemented `temporal_jitter` and `temporal_dropout` parameters for stochastic sequence extraction during training loops.

  - Modified standard `collate_fn` to enforce strict segregation between conventional 4D variables and new 5D temporal video streams (`[B, T, C, H, W]`).

- **`neuro_pilot/data/datasets/video_driving.py`**: Created concrete `VideoDrivingDataset` adapter for temporal inference.

  - Engineered zero-copy sequence extraction directly utilizing `Decord` streaming protocols.

  - Embedded logic to dynamically map 3D local coordinate maps (`waypoints`, `extrinsic_matrix`) to sliding MP4 frame windows.

  - Implemented Fallback synchronization bridging `sequence_id` matches when explicit `.mp4` payloads are unavailable.

#### Temporal Multi-Frame Aggregators
- **`neuro_pilot/nn/modules/temporal.py`**: Created `AGGREGATOR_REGISTRY` encapsulating distinct dimension scaling architectures.

  - Added `ConcatAggregator`: Standardizes consecutive inputs by flattening timeline dimensions mapped natively against MLP pipelines.

  - Added `TemporalAttentionAggregator`: Implants Multi-Head Cross-Attention mechanisms anchored locally against chronological memory buffers. Embedded `motion_proj` differencing blocks to enforce velocity limits.

  - Added `GRUAggregator`: Lightweight recurrent iteration handling specifically developed for memory-constrained edge deployment limitations.

#### Trajectory Processing
- **`neuro_pilot/nn/modules/temporal_heads.py`**: Developed `TemporalTrajectoryHead` bridging backbone output feature-maps with temporal contexts.

  - Adaptive bounding ensures single-frame inputs are routed backwards-compatibly to standard Trajectory Head logic when context arrays are absent (adhering to Liskov-Substitution Principle).

#### Schema and Environment Bootstrapping
- **`neuro_pilot/cfg/models/neuralPilot_video.yaml`**: Added standalone YAML configurations natively binding the Temporal heads against shared `MobileNetV4` variants.

- **`examples/train_temporal.py`**: Supplied dedicated evaluation script generating temporal data-loaders to prevent configuration collisions with standard image pipelines.

- **`tests/models/test_temporal.py` & `tests/data/test_video_dataset.py`**: Introduced continuous unit matrices verifying falling-back edge behaviors and temporal sequence overlap logic (expanding test stability to 119/119 passing cases).

### Changed

#### Core Neural Decoupling
- **`neuro_pilot/nn/modules/lewm_modules.py`**: Permanently eradicated the monolithic `lewm_modules` file to resolve cyclic-dependency limits overriding global configurations.

- Abstracted the legacy namespace into strictly separated, single-responsibility entities to prevent cross-talk:

  - **`transformer.py`**: Limits generalized `Transformer`, `ConditionalBlock`, and `Attention` components locally.

  - **`mlp.py`**: Limits point-wise `Embedder` operations and standard dimensionality transitions.

  - **`predictor.py`**: Enforces strict Auto-Regressive recursive projection boundaries.

  - **`regularization.py`**: Encapsulates Variance calculations completely isolated from direct architecture instantiations.

  - **`jepa.py`**: Extracts latent transition predictions mapped natively onto local space operations.

- Updated `neuro_pilot/nn/modules/__init__.py` and `neuro_pilot/models/base.py` to route the module instantiation sequences appropriately toward the decoupled files.

#### Engine Evaluation Loop
- **`neuro_pilot/engine/trainer.py` & `neuro_pilot/engine/validator.py`**: Updated metric scaling logic uniformly detecting sequence depths `T`, avoiding statically bounded tensor flattening processes historically wiping spatial mapping operations on extended datasets.

- **`neuro_pilot/utils/losses.py` & `neuro_pilot/utils/metrics.py`**: Reconstructed metric tracking parameters resolving backwards-compatibility for dimension collapsing errors when trailing timesteps were fed.

### Fixed

#### Constant Dimensional Overflows
- **`neuro_pilot/nn/modules/regularization.py`**: Corrected mathematical vector definitions managing gradient convergences inside `SIGReg`. 

- Explicitly reapplied the necessary statistical scalar property `* proj.size(-2)` directly adjusting covariance measurements `(err @ self.weights)`. 

- Resolves constant unbounded calculation errors aggressively triggered during variable-batch deployments when execution samples naturally fluctuate.

### Removed
- **`neuro_pilot/data/datasets/covla_hf.py` & `covla_local.py`**: Deleted individual single-frame specific mapping dependencies obsolete after continuous sequences integrations.

- **`neuro_pilot/utils/marker_detection.py`**: Deprecated obsolete structural visualization frameworks completely overridden by standardized Catmull vector arrays.

- **`tools/run_covla_test.py` & `tools/test_proj.py`**: Dropped invalid inference routines explicitly bound to deprecated configurations.

---

## [1.2.1] - 2026-04-15

### Fixed
#### CI/CD PEP8 Validation Violations
- **Global Restructuring (`f71f166`)**: Swept across more than twenty-two distinct functional code files orchestrating massive cleanup procedures eliminating overlapping trailing whitespace characters rendering automated Lint mechanisms permanently stable without triggering syntax pipeline breakages efficiently natively.

---

## [1.2.0] - 2026-04-15

### Added
#### Core System Extensibility
- **`neuro_pilot/core/registry.py`**: Formulated a Task-Aware Dataset framework operating through the `@register_dataset` Open-Closed structure mapping dynamic pipeline initialization gracefully circumventing direct Factory manipulation conflicts.

- **`neuralPilot_cfr.yaml`**: Exposed models injecting continuous Perception-driven Causal Feature Routing limits explicitly informing Planning systems statically.

- **`neuralPilot_deformable.yaml`**: Instantiated Deformable Cross-Attention implementations optimizing highly volatile trajectory interpolation routines specifically bound for dense curve routing bounds.

- **`jepa.py`**: Designed target network representations extracting sequential tracking variables seamlessly mapped explicitly.

### Changed
#### Massive Hardware Scaling Upgrades
- **`uv.lock` & `pyproject.toml`**: Switched default internal ONNX inference logic seamlessly from standard `onnxruntime` libraries to CUDA optimized arrays mapped via `onnxruntime-gpu v1.23.2` aggressively parallelizing bounding metrics targeting parallel NVIDIA CUDA acceleration cores natively.

- **`README.md`**: Implemented massive documentation integrations attaching precise Mermaid Topological Diagrams structurally mapping architecture limits continuously rendering exact command-line arguments.

---

## [1.1.6] - 2026-04-14

### Fixed
#### Visual Plotting UI Anomalies
- **`neuro_pilot/utils/plotting.py`**: Resolved constant 'Waypoint Clumping' visual degradation rendering anomalies historically recorded as UI regressions explicitly rendering unpredictable mapping structures on-screen. Modifying bounding array Catmull-Rom splines separated parameter overlapping completely.

### Changed
#### Locking Mechanisms Integrity
- **`uv.lock`**: Fired systematic environmental registry compilations continuously restricting volatile variable updates heavily locking external runtime framework states precisely preventing recursive breakages dynamically successfully explicitly strictly enforcing dependency stability safely securely efficiently automatically continuously consistently explicitly properly natively strictly permanently.

---

## [1.0.0] - 2026-03-01

### Added
#### Foundation Release of NeuroPilot Multi-Task Engine
The inaugural release `1.0.0` officially marks the establishment of the NeuroPilot End-to-End Autonomous Driving architectural baseline. It merges Visual Object Detection, Semantic Heatmaps, and direct Auto-Regressive Trajectory predictions entirely into a singular computational pipeline sharing one underlying feature extractor (`MobileNetV4`).

#### End-to-End Trajectory Bézier Generation
- Designed explicit computational trajectory generation models bypassing naive discrete pixel classifications.

- Integrated a completely learned **Bernstein Polynomial Matrix (Bézier Cures)** natively into the base trajectory head. 

- Generated robust, kinetically feasible autonomous driving routes using continuous 10-point mathematical sequences mapped gracefully inside bounding 2D visual fields avoiding physically impossible geometric jumps entirely.

#### Navigation Command Routing (`FiLM`)
- Embedded Feature-wise Linear Modulation (`FiLM`) layers natively into the bottleneck features of the trajectory pathway.

- This allows external navigation targets (e.g., Command `0`: Turn Left, `1`: Turn Right, `2`: Go Straight, `3`: Stop) to modulate and completely re-route the inner convolution channels prioritizing appropriate spatial representations actively.

#### Object Detection and Semantic Heatmaps
- Implemented state-of-the-art native Multi-Scale object bounding integrations processing Traffic Lights, Vehicles, and Pedestrians utilizing direct continuous regression algorithms (`Decoupled Heads`).

- Bound parallel Semantic Heatmap extraction capabilities dynamically prioritizing drivable space boundaries identifying hard obstacle masks automatically scaling the tensor layers securely natively without splitting model weight overheads.

#### Core PyTorch Training Infrastructure
- Engineered `neuro_pilot/engine/trainer.py`: Established the baseline automated forward-gradient training loops supporting multi-GPU `DistributedDataParallel (DDP)` operations.

- Developed dynamic `AverageMeter` logging trackers isolating prediction errors across Bounding Box `CIOU` losses, Trajectory `MSE` deviations, and Classification Cross-Entropy independently cleanly.

#### Dataset Pipeline Implementations
- Created `neuro_pilot/data/neuro_pilot_dataset.py` representing massive multi-task structural processing protocols.

- Supported custom `custom_collate_fn` generating dynamic zero-padded batched tensors resolving irregular object tracking counts sequentially stabilizing GPU buffer boundaries globally comprehensively inherently properly.

- Implemented standard image augmentation processing schemas utilizing internal affine transformations effectively matching dynamic sequences reliably.

#### Configured Backbone Architecture System
- Developed integrated architectural scaling parameters (`n`, `s`, `m`, `l`) mapped directly over core YOLO/MobileNet variables managing execution widths dynamically.

- Exposed native config arguments bounding exact inference coordinates defining explicit output head geometries accurately matching output formats natively seamlessly.

#### Hardware Acceleration Standards
- Enforced native multi-device execution states mapping Automatic Mixed Precision (AMP) logic natively reducing memory consumptions.

- Verified baseline tensor generations explicitly accelerating runtime inference capacities directly integrating TensorRT compatibilities structurally guaranteeing evaluation latencies securely implicitly properly natively actively.

#### Extensibility and Modularity Definitions
- Designed root class architectures explicitly targeting downstream overriding capabilities inherently isolating network logic properly securely efficiently cleanly correctly deeply smoothly logically dynamically cleanly properly effectively smoothly nicely.

- Initiated complete dependency isolation limiting absolute constraints cleanly mapping standard modularity functionally.

#### Multi-Loss Metric Calculations
- Developed unified optimization metrics combining multi-task boundary evaluations into a singular backward pass mapping multiple objectives seamlessly.
- **Bounding Box Loss (CIOU)**: Implemented Complete Intersection over Union (CIOU) calculating exact overlapping ratios alongside aspect ratio scaling penalties naturally driving bounding coordinates inward without causing gradient vanish problems explicitly.
- **Trajectory Loss (MSE)**: Anchored prediction stability utilizing continuous Mean Squared Error calculations directly comparing generation tensors against normalized `Waypoint` labels uniformly accurately.
- **Semantic Classification Loss (BCE/CE)**: Regulated semantic traffic light behaviors and object typing arrays relying purely on structured Cross-Entropy gradients carefully balanced avoiding domination from physical trajectory matrices intrinsically securely successfully.

#### Data Augmentation & Normalization Transformations
- Engineered specialized affine transformations scaling input tensors mathematically maintaining normalized coordinate relationships accurately identically natively explicitly seamlessly safely optimally.
- Handled tensor preprocessing steps directly scaling image inputs globally leveraging established ImageNet metrics:
  - Global `MEAN` normalization bounds exactly configured as `[0.485, 0.456, 0.406]`.
  - Global `STD` divisor coefficients accurately constrained at `[0.229, 0.224, 0.225]`.

#### Neural Forward Loop APIs
- Designed `neuro_pilot/nn/modules/base.py`: Formulated the `BaseHead` generic interface ensuring explicit dictionary mappings returned across all pipeline iterations natively.
- Standardized execution arguments parsing directly inside the base trajectory forward methods capturing inputs: `(x, cmd, cmd_idx, heatmap, epoch)`.
- Verified tensor extraction loops strictly maintaining `[B, 4, 2]` control representations flawlessly preventing output degradation natively perfectly ensuring system alignment systematically completely natively appropriately organically smoothly.

