---
sidebar_position: 2
---
# Python SDK Reference Manual

This document serves as the comprehensive Python API and SDK guide for **NeuroPilot**. We document class signatures, runtime arguments, data schemas, output prediction classes, and custom callbacks with fully compilable code blocks.

---

## 1. Programmatic Entrypoint: `NeuroPilot`

The `NeuroPilot` orchestrator class (located in `neuro_pilot/engine/model.py`) provides high-level hooks matching the CLI subcommands.

```python
from neuro_pilot.engine.model import NeuroPilot

model = NeuroPilot(
    model: str,
    scale: str = "s",
    task: str = "driving"
)
```

### Constructor Parameters
* **`model`** *(str, required)*: Path to a model configuration YAML file (e.g. `neuralPilot.yaml`) to build the PyTorch graph from scratch, or path to serialized weights (`best.pt`, `best.onnx`, or `best.engine`) to load a pre-trained model.
* **`scale`** *(str, default: `"s"`)*: Model size scaling factor control. Affects backbone depth/width multipliers. Choose from: `"n"`, `"s"`, `"m"`, `"l"`, `"x"`.
* **`task`** *(str, default: `"driving"`)*: Execution task registry binding key. Automatically maps corresponding optimizer and loss functions.

---

### Core Methods Directory

#### A. `train` — Run Model Fitting Loop
```python
model.train(
    data: str,
    max_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "0",
    **kwargs
)
```
* **Parameters**:
  * `data` *(str, required)*: Path to the dataset config YAML.
  * `max_epochs` *(int, default: `100`)*: Maximum epochs.
  * `batch_size` *(int, default: `16`)*: Mini-batch size.
  * `learning_rate` *(float, default: `1e-4`)*: Initial learning rate.
  * `device` *(str, default: `"0"`)*: CUDA device ID (e.g., `"0"`, `"cpu"`).
  * `**kwargs`: Pass arbitrary hyperparameter overrides (e.g. `lambda_traj=10.0`, `use_fdat=True`).

---

#### B. `predict` — Real-Time Inference
```python
results = model.predict(
    source: Union[str, int, np.ndarray],
    imgsz: int = 640,
    conf: float = 0.25,
    save: bool = False,
    **kwargs
)
```
* **Parameters**:
  * `source` *(str | int | np.ndarray, required)*: Path to video/image files, camera index (`0`), or pre-loaded numpy arrays.
  * `imgsz` *(int, default: `640`)*: Image resize resolution.
  * `conf` *(float, default: `0.25`)*: Confidence threshold.
  * `save` *(bool, default: `False`)*: Automatically saves annotated outputs to disk.
* **Returns**: `List[PredictResults]` containing frame-by-frame task predictions.

---

#### C. `export` — Compile Graph to ONNX or TensorRT
```python
output_path = model.export(
    format: str = "onnx",
    imgsz: int = 640,
    dynamic: bool = False,
    **kwargs
)
```
* **Parameters**:
  * `format` *(str, default: `"onnx"`)*: Edge target format (`"onnx"` or `"engine"`).
  * `imgsz` *(int, default: `640`)*: Resized resolution baked into the model architecture graph.
  * `dynamic` *(bool, default: `False`)*: Enable dynamic batch sizing inside ONNX.
* **Returns**: `str` representing the file path of the exported binary file.

---

## 2. Output Schema: `PredictResults`

The `predict` method outputs a list of `PredictResults` instances (defined in `neuro_pilot/engine/results.py`). This class contains raw tensors and processing utilities.

### Properties
* **`results.boxes`** *(torch.Tensor | None)*: Object detection bounding boxes of shape `[N, 6]`. Columns represent: `[x_min, y_min, x_max, y_max, confidence, class_id]`.
* **`results.waypoints`** *(torch.Tensor | None)*: Predicted future trajectory path coordinates of shape `[T, 2]`, representing `T` spatial coordinates normalized to `[-1.0, 1.0]`.
* **`results.heatmap`** *(torch.Tensor | None)*: Visual gaze spatial attention heatmap matrix of shape `[H, W]`.
* **`results.command`** *(int | None)*: Index of the predicted navigation directive.

### Methods
* **`results.plot(heatmap: bool = True) -> np.ndarray`**: Draws the overlays (bounding boxes with class names, plan splines, attention heatmaps) and returns a BGR image as a numpy array.
* **`results.save(save_dir: str = "runs/predict")`**: Writes the plotted visual frames to a file on disk.

---

## 3. Training Callbacks & Plugins (`callbacks.py`)

NeuroPilot provides a hook-based callback registry to intercept training states and customize logging or checkpointing behavior.

### Writing and Registering a Custom Callback
```python
import torch
from neuro_pilot.engine.callbacks import add_callback

def log_learning_rate_to_wandb(trainer):
    """
    Called at the end of each training epoch.
    Accesses trainer parameters dynamically.
    """
    current_epoch = trainer.epoch
    optimizer = trainer.optimizer
    current_lr = optimizer.param_groups[0]['lr']
    
    # Example logging to weights and biases (W&B)
    # wandb.log({"epoch": current_epoch, "lr": current_lr})
    print(f"[Callback Logger] Epoch: {current_epoch} | Current Learning Rate: {current_lr:.6f}")

# Bind callback function to the end of a training epoch
add_callback("on_train_epoch_end", log_learning_rate_to_wandb)
```

---

## 4. End-to-End Programmatic Pipeline Example

Below is a complete, compilable python script illustrating custom model orchestration:

```python
import os
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.callbacks import add_callback

# 1. Register a validation checkpointing callback hook
def custom_checkpoint_saver(trainer):
    current_loss = trainer.val_loss
    epoch = trainer.epoch
    if current_loss < getattr(trainer, "best_val_loss", float("inf")):
        trainer.best_val_loss = current_loss
        save_path = os.path.join(trainer.save_dir, "weights", "custom_best.pt")
        trainer.save_model(save_path)
        print(f"[Checkpoint Callback] Best model updated at epoch {epoch} with validation loss: {current_loss:.4f}")

add_callback("on_train_epoch_end", custom_checkpoint_saver)

# 2. Instantiate model from YAML configuration file
model = NeuroPilot(
    model="neuro_pilot/cfg/models/neuralPilot.yaml", 
    scale="s"
)

# 3. Fit model on custom dataset
model.train(
    data="data/yolo_dataset.yaml",
    max_epochs=5,
    batch_size=8,
    learning_rate=2e-4,
    device="0",
    use_fdat=True,
    lambda_det=1.0,
    lambda_heatmap=2.0
)

# 4. Export the resulting PyTorch weights to ONNX format
onnx_path = model.export(
    format="onnx",
    imgsz=320,
    dynamic=False
)
print(f"Model exported successfully to: {onnx_path}")

# 5. Run inference on a test video
results = model.predict(
    source="data/test_clip.mp4",
    imgsz=320,
    conf=0.25,
    save=True
)

# Print metrics summary from the first 5 frames
for idx, res in enumerate(results[:5]):
    num_detections = len(res.boxes) if res.boxes is not None else 0
    print(f"Frame {idx} | Detections: {num_detections} | Target Nav Command: {res.command}")
```
