"""Universal dataset protocol for self-driving tasks.

Any dataset (YOLO, CoVLA, nuScenes, Waymo, CARLA, custom VLA, Segmentation, ...)
can plug into NeuroPilot by:

1. Subclassing ``BaseDrivingDataset``
2. Implementing ``__getitem__`` to return a ``dict``
3. Decorating with ``@register_dataset("my_type")``

The factory reads the ``type`` key from the dataset YAML and dispatches
to the registered class — **zero** if/elif chains required (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Registry (GoF Registry / Service‑Locator lite)
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    """Class decorator that registers a ``BaseDrivingDataset`` subclass.

    Usage::

        @register_dataset("carla_hf")
        class CarlaHFDataset(BaseDrivingDataset):
            ...

    The name must match the ``type`` field in the dataset YAML file.
    """
    def _decorator(cls: type[BaseDrivingDataset]) -> type[BaseDrivingDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset type '{name}' is already registered by "
                f"{DATASET_REGISTRY[name].__name__}. Cannot register "
                f"{cls.__name__}."
            )
        DATASET_REGISTRY[name] = cls
        return cls
    return _decorator


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------
class BaseDrivingDataset(Dataset, ABC):
    """Universal protocol for NeuroPilot datasets.

    Subclasses must return a ``dict`` from ``__getitem__``.

    Mandatory keys:
        image (Tensor[C, H, W]): Pre-processed image tensor.

    Common optional keys (task-dependent):
        command (int | Tensor):          Navigation command index or one-hot.
        waypoints (Tensor[T, 2]):        Future trajectory waypoints.
        bboxes (Tensor[N, 4]):           Bounding boxes (x_c, y_c, w, h) normalized.
        categories (Tensor[N]):          Class labels for each bbox.
        heatmap (Tensor[H, W]):          Ground-truth attention heatmap.
        segmentation_mask (Tensor[H, W]):Semantic segmentation mask.
        depth_map (Tensor[H, W]):        Depth estimation target.
        language_prompt (str):           Text instruction (VLA / Grounding).
        action_target (Tensor):          Raw control action target (VLA).

    The collate function handles mixed tensor/non-tensor values automatically.
    """

    # ---- Factory classmethod (can be overridden per-dataset) ---------------
    @classmethod
    def from_config(cls, config: Any, split: str, yaml_dict: dict) -> "BaseDrivingDataset":
        """Instantiate this dataset from the pipeline config + YAML dict.

        Override in subclasses to map config fields to constructor args.
        The default implementation raises NotImplementedError to force
        each dataset to provide its own logic.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement `from_config(config, split, yaml_dict)`."
        )

    # ---- Abstract interface ------------------------------------------------
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a data sample as a dict."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    # ---- Default collate ---------------------------------------------------
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Default collate: stack tensors, keep lists for variable-length data."""
        keys = batch[0].keys()
        out: dict = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                try:
                    out[k] = torch.stack(vals)
                except RuntimeError:
                    # Variable-length tensors (e.g. bboxes with different N)
                    out[k] = vals
            else:
                out[k] = vals
        return out
