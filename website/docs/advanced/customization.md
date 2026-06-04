---
sidebar_position: 1
---
# Advanced Customization & Extension

NeuroPilot is designed following the **Open-Closed Principle (OCP)**. You do not need to modify the core codebase to experiment with custom layers, custom routing necks, new task loss formulations, or specialized training loops. Instead, you can extend the framework by writing modular custom code and linking it via config files and decorators.

---

## 1. Dynamic YAML Module Resolution

When loading model architecture configs (such as `neuralPilot.yaml`), the parsing engine (`neuro_pilot/nn/tasks.py`) parses layers using the following structure:

```yaml
# [from, repeats, module_name, arguments]
- [20, 1, neuro_pilot.nn.modules.head.ClassificationHead, [nm]]
```

### How the Loader Resolves Modules
1. **Local Safe Map**: If `module_name` is a simple string (e.g., `Conv`, `Detect`), it resolves to built-in PyTorch modules or NeuroPilot layers.
2. **Dynamic Imports**: If `module_name` contains a dot (e.g., `my_package.layers.CustomHead`), the loader dynamically imports the module at runtime:
   ```python
   # Excerpt from neuro_pilot/nn/tasks.py
   module_name, attr = m_name.rsplit(".", 1)
   mod = importlib.import_module(module_name)
   return getattr(mod, attr)
   ```

This architecture allows you to create custom modules in a separate folder and load them directly via the YAML config without altering core code.

---

## 2. Writing and Referencing a Custom Layer/Head

To add a new prediction task (for example, a head that predicts road surface conditions or lane boundary offsets):

### Step A: Write the PyTorch Module
Create your custom module (e.g., `my_custom_package/heads.py`):

```python
import torch
import torch.nn as nn

class CustomRoadConditionHead(nn.Module):
    def __init__(self, c1: int, num_classes: int = 3):
        """
        c1: input channel dimension (automatically injected by the compiler)
        num_classes: output target dimensions
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.act = nn.SiLU()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is the input feature map drawn from the specified layers index
        out = self.act(self.bn(self.conv(x)))
        out = self.pool(out).flatten(1)
        return self.fc(out)
```

### Step B: Reference in YAML Configuration
In your model YAML configuration, reference your head using its fully qualified import path:

```yaml
# my_cfg.yaml
nc: 14
nm: 4
nw: 10

backbone:
  # ... standard backbone definition ...

head:
  # ... neck and other heads ...
  
  # Connect to index 20 (e.g., the output of the PAN neck)
  - [20, 1, my_custom_package.heads.CustomRoadConditionHead, [3]]
```

---

## 3. Registering Custom Tasks (`Registry`)

A **Task** in NeuroPilot handles the execution flow of training steps, loss calculations, validation epochs, and prediction collation. 

To create a custom task (e.g., adding custom reinforcement learning rewards or custom evaluation hooks):

### Step A: Subclass `BaseTask` and Decorate
Inherit from `BaseTask` (defined in `neuro_pilot/engine/task.py`) and register it using `@Registry.register_task`:

```python
# my_custom_package/tasks.py
from neuro_pilot.core.registry import Registry
from neuro_pilot.engine.task import BaseTask

@Registry.register_task("custom_driving_task")
class CustomDrivingTask(BaseTask):
    def train_step(self, batch, model, optimizer):
        """Define your own custom forward pass, loss calculation, and optimization step."""
        images = batch["image"].to(self.device)
        targets = batch["waypoints"].to(self.device)
        
        # Custom forward pass
        preds = model(images)
        
        # Custom loss evaluation
        loss = self.compute_custom_loss(preds, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {"loss": loss.item()}

    def compute_custom_loss(self, preds, targets):
        return torch.mean((preds - targets) ** 2)
```

### Step B: Execute via the CLI
Invoke the custom task runner directly using the CLI:
```bash
uv run neuropilot train config.yaml --data data.yaml task=custom_driving_task
```

---

## 4. Custom Temporal Aggregators

When dealing with sequential video inputs, the `TemporalNeck` uses a strategy pattern to aggregate frame feature steps over time. You can plug in a custom temporal aggregator using the `AGGREGATOR_REGISTRY` (defined in `neuro_pilot/nn/modules/temporal.py`):

```python
# my_custom_package/aggregators.py
from torch import nn
from neuro_pilot.nn.modules.temporal import AGGREGATOR_REGISTRY

@AGGREGATOR_REGISTRY.register("custom_lstm")
class CustomLSTMAggregator(nn.Module):
    def __init__(self, feature_dim: int, clip_length: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        # x shape: [Batch, Clip_Length, Feature_Dim]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Return last sequence step
```

Reference `"custom_lstm"` inside your temporal configurations:
```yaml
- [20, 1, TemporalNeck, [256, 'custom_lstm']]
```
