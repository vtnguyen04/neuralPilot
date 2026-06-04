---
sidebar_position: 1
---
# Modular Backbones & Feature Extractor

The backbone acts as the primary visual feature encoder of NeuroPilot. It transforms raw image tensors of shape `[3 x H x W]` into hierarchical multi-scale feature maps. This document details the visual stage hierarchies, channel scaling rules, Timm integrations, and comparison tables.

---

## 1. Feature Level Strides (P2–P5)

NeuroPilot extracts multi-scale feature maps at strides 4, 8, 16, and 32. This multi-level resolution mapping ensures the network captures both local high-resolution details and global contextual information.

```
Input Frame [320x320]
  │
  ├── Stage P2 (Stride 4)  ──► [Batch, C_P2, 80, 80]   ──► Local fine textures (Heatmaps)
  │
  ├── Stage P3 (Stride 8)  ──► [Batch, C_P3, 40, 40]   ──► Small object detection
  │
  ├── Stage P4 (Stride 16) ──► [Batch, C_P4, 20, 20]   ──► Medium object bounding boxes
  │
  └── Stage P5 (Stride 32) ──► [Batch, C_P5, 10, 10]   ──► Contextual planning features
```

### Purpose of Stages
* **P2 Stage**: Focuses on fine-grained visual details (e.g., lane borders, edge boundaries). Essential for generating precise, continuous gaze attention heatmaps.
* **P3 & P4 Stages**: Encodes spatial geometries. Sent to the YOLO11 PAN neck to perform anchor-free 2D object detection across varying scales.
* **P5 Stage**: Captures macro context. Modulated with navigational commands to guide trajectory planning.

---

## 2. Pluggable `timm` Wrapper

NeuroPilot uses a generic wrapper class `TimmBackbone` (defined in `neuro_pilot/nn/modules/backbone.py`) to connect the repository to the `timm` (PyTorch Image Models) zoo. This wrapper automatically extracts intermediate feature maps from arbitrary visual models.

```python
# neuro_pilot/nn/modules/backbone.py (Excerpt)
import timm
import torch.nn as nn

class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, features_only: bool = True):
        super().__init__()
        # Load backbone from timm zoo
        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=(1, 2, 3, 4) # maps to P2, P3, P4, P5 stages
        )
        self.feature_channels = self.net.feature_info.channels()

    def forward(self, x):
        # Returns a list of multi-scale tensors
        return self.net(x)
```

### Setting the Backbone in YAML
To swap the backbone, modify the `backbone` block in the model configuration YAML:

```yaml
backbone:
  # [from, repeats, module, arguments]
  - [-1, 1, TimmBackbone, ['fastvit_t8.apple_dist_in1k', True, True]]
```

---

## 3. Comparison of Pluggable Backbones

The choice of backbone directly dictates VRAM footprint, training times, and execution FPS on edge hardware.

| Backbone Key Name | Parameter Count | VRAM (Batch=8) | Latency (Jetson Orin Nano) | Recommended Use Case |
|---|---|---|---|---|
| **`mobilenetv4_conv_small`** | ~3.8M | 1.8 GB | **12.4 ms** | Default baseline for ultra low-latency edge deployment. |
| **`efficientvit_b1`** | ~4.6M | 2.5 GB | **18.9 ms** | Transformer-based; best suited for dense obstacle configurations. |
| **`fastvit_t8`** | ~3.2M | 2.0 GB | **14.2 ms** | Re-parameterized architecture; optimal balance of speed/accuracy. |
| **`mobilevitv2_050`** | ~4.2M | 2.8 GB | **29.5 ms** | Global attention; suitable for servers or high-end Jetson AGX. |

---

## 4. Native YOLO11 Conv Backbone

If you wish to deploy without the `timm` package, NeuroPilot provides a native Conv-based YOLO11 stem (`neuro_pilot/nn/modules/block.py`):

* **C3k2 Blocks**: CSP bottlenecks with two convolutions. It splits channels, runs parallel bottleneck convolutions, and merges them using `1x1` kernels to enhance gradient flow.
* **C2PSA (Spatial Attention)**: Focuses the feature search space on critical objects.
* **SPPF (Spatial Pyramid Pooling - Fast)**: Fuses features at multiple receptive fields (kernel sizes `5x5`, `9x9`, `13x13`) using parallel max-pooling layers, avoiding information loss on multi-scale objects.
