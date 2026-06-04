---
sidebar_position: 1
---
# Custom Dataset Integration

NeuroPilot utilizes the **Open-Closed Principle (OCP)** for dataset loading. By implementing a central registry, you can add support for any new dataset format (such as nuScenes, CARLA simulator logs, CoVLA, or custom sensor recording pipelines) with **zero modifications** to the core repository engine.

---

## 1. The Dataset Registry Architecture

NeuroPilot matches the `type` field defined in your dataset configuration YAML file with the python classes registered in `DATASET_REGISTRY`. 

```
Dataset Configuration YAML (type: "carla_simulation")
  │
  ▼
Dataset Factory Loader
  │
  ├── Searches DATASET_REGISTRY for "carla_simulation"
  │
  ▼
Instantiates CarlaSimulationDataset (BaseDrivingDataset subclass)
```

Registration is handled dynamically via the `@register_dataset` class decorator:
```python
from neuro_pilot.data.datasets.base import BaseDrivingDataset, register_dataset

@register_dataset("carla_simulation")
class CarlaSimulationDataset(BaseDrivingDataset):
    ...
```

---

## 2. Core API: BaseDrivingDataset

Your custom dataset must inherit from `BaseDrivingDataset` and implement three core methods:

### A. Constructor `__init__`
Instantiate data lists, load JSON/CSV indexes, and setup transformations.

### B. Length `__len__`
Return the total number of frames or samples in the dataset:
```python
def __len__(self) -> int:
    return len(self.samples)
```

### C. Sample Getter `__getitem__`
Retrieve a single sample at index `idx` and return it as a dictionary.

### D. Factory Instantiation `from_config`
This classmethod is called by the trainer to instantiate the dataset using configuration settings:
```python
@classmethod
def from_config(cls, config: Any, split: str, yaml_dict: dict) -> "YourDataset":
    ...
```

---

## 3. Data Dictionary Schema

The dictionary returned by `__getitem__` must contain specific key-value mappings depending on which task heads are enabled in your model configuration:

| Key | Expected Type | Shape / Dimension | Description |
|---|---|---|---|
| **`image`** *(Required)* | `torch.Tensor` | `[3, H, W]` | Preprocessed RGB image float tensor (normalized to `[0.0, 1.0]`). |
| **`waypoints`** | `torch.Tensor` | `[T, 2]` | Planned trajectory points. Normalized to the range `[-1.0, 1.0]`, where `(0, 0)` is the center-bottom of the camera. |
| **`command`** | `int` or `torch.Tensor` | Scalar or `[num_commands]` | Navigation directive class index (e.g., `0: Follow`, `1: Left`, `2: Right`, `3: Straight`). |
| **`bboxes`** | `torch.Tensor` | `[N, 4]` | YOLO format bounding boxes. Coordinates are normalized `(x_center, y_center, width, height)`. |
| **`categories`** | `torch.Tensor` | `[N]` | Class labels representing each bounding box. |
| **`heatmap`** | `torch.Tensor` | `[H, W]` | Saliency or gaze target density maps. |

---

## 4. Bounding Box Collate Logic

A major hurdle in multi-task training is collation. In standard PyTorch, batches are stacked using `torch.stack`. However, if Image A has 3 bounding boxes and Image B has 7 bounding boxes, `torch.stack` will throw a shape mismatch error because the bounding box dimension $N$ varies.

`BaseDrivingDataset` solves this with a custom `collate_fn`:
* Tensors of identical size (like `image`, `waypoints`, `command`) are stacked into a batch tensor.
* Variable-length tensors (like `bboxes` and `categories`) are kept as lists of tensors, which the anchor-free detection head handles natively.

---

## 5. Complete Step-by-Step Implementation Example

Below is a complete implementation of a custom dataset adapter that reads images from a folder and labels from a JSONL file:

```python
# neuro_pilot/data/datasets/custom_sensor_dataset.py
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from neuro_pilot.data.datasets.base import BaseDrivingDataset, register_dataset

@register_dataset("custom_sensor_data")
class CustomSensorDataset(BaseDrivingDataset):
    def __init__(self, data_root: str, split: str, image_size: int = 320):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Load sample index
        self.samples = []
        index_path = os.path.join(data_root, f"{split}_index.json")
        with open(index_path, "r") as f:
            self.samples = json.load(f)

    @classmethod
    def from_config(cls, config, split: str, yaml_dict: dict) -> "CustomSensorDataset":
        """Build dataset instance dynamically using the training configuration."""
        data_root = yaml_dict.get("path", "")
        imgsz = config.get("imgsz", 320)
        return cls(data_root=data_root, split=split, image_size=imgsz)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_meta = self.samples[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.data_root, sample_meta["image_path"])
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)
        
        # 2. Load Trajectory Waypoints (T=10 waypoints)
        raw_wps = sample_meta["waypoints"]  # List of [x, y]
        wps_tensor = torch.tensor(raw_wps, dtype=torch.float32)
        
        # 3. Load Command
        cmd = int(sample_meta.get("command", 0))
        
        # 4. Load Bounding Boxes (YOLO format: normalized cx, cy, w, h)
        raw_boxes = sample_meta.get("bboxes", [])  # List of [cx, cy, w, h]
        boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
        
        # 5. Load BBox Categories
        raw_cats = sample_meta.get("categories", [])
        cats_tensor = torch.tensor(raw_cats, dtype=torch.long)
        
        return {
            "image": img_tensor,
            "waypoints": wps_tensor,
            "command": cmd,
            "bboxes": boxes_tensor,
            "categories": cats_tensor
        }
```

### Registries Binding
To ensure the class is loaded and registered before training starts, import the file in `neuro_pilot/data/datasets/__init__.py`:

```python
# neuro_pilot/data/datasets/__init__.py
...
from .custom_sensor_dataset import CustomSensorDataset
```

---

## 6. Configure and Run

Create a dataset config file:

```yaml
# data/custom_sensor.yaml
path: /data/custom_driving_data
type: custom_sensor_data   # Must match @register_dataset name
train: train/
val: val/
names:
  0: vehicle
  1: pedestrian
  2: traffic_light
```

Execute training using your new dataset:
```bash
uv run neuropilot train neuralPilot.yaml --data data/custom_sensor.yaml
```
