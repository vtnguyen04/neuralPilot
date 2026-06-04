---
sidebar_position: 2
---
# Temporal Video Pipelines

Autonomous driving depends on understanding motion and temporal changes. NeuroPilot provides native support for temporal training pipelines to train models directly on video feeds using the **Decord** video reading library.

---

## 1. Under the Hood: Decord Video Loading

Traditional video frame loading using OpenCV (`cv2.VideoCapture`) relies on sequential frame-by-frame decoding, which is extremely slow when randomly sampling sequences for mini-batch SGD.

NeuroPilot integrates **Decord** to solve this:
* **Hardware-Accelerated Decoding**: Leverages GPU-accelerated video decoding (NVDEC) when available to speed up image data pipelines.
* **Random Frame Seek**: Decord enables direct, index-based access to any video frame without needing to decode preceding frames.
* **Batch Reading**: Reads whole video intervals (e.g. 10 frames) into memory in a single runtime binding operation.

---

## 2. Dynamic Video Dataset Adapter

The `VideoDrivingDataset` class (defined in `neuro_pilot/data/datasets/video_driving.py`) is decorated with `@register_dataset("video_driving")`. It operates by parsing video records paired with JSONL log annotations:

* **Video Directory**: Path containing `.mp4` video segments.
* **JSONL Log Entries**: Coordinate and velocity timestamps corresponding to the video frames.
* **Sequence Alignment**: Maps the frame index to future trajectory coordinate indices to generate ground-truth waypoints.

---

## 3. Configuring Temporal Training

To train models on temporal datasets, use a configuration matching `neuralPilot_video.yaml`. This model layout incorporates a `TemporalTrajectoryHead`:

```yaml
# neuro_pilot/cfg/models/neuralPilot_video.yaml
...
head:
  # Multi-stage feature extraction neck
  - [0, 1, FeatureRouter, [4]]
  
  # Temporal sequence mapping layers
  - [[14, 17, 20], 1, TemporalNeck, [256]]
  
  # Temporal Trajectory head predicts waypoints sequentially
  - [20, 1, TemporalTrajectoryHead, [nm, nw]]
```

### Launching Temporal Video Training
Run the training sequence using the provided example script:

```bash
PYTHONPATH=. uv run python examples/train_temporal.py
```
This runs the temporal encoder pipeline, stacking video sequences of shape `[Batch, Sequence_Length, Channels, Height, Width]` directly into the model graph.
