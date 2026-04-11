"""Universal dataset protocol for self-driving tasks.

Any dataset (YOLO, CoVLA, nuScenes, Waymo, custom VLA, Segmentation, Grounding...)
can plug into NeuroPilot by subclassing ``BaseDrivingDataset`` and implementing
``__getitem__`` to return a ``dict``.

The dict protocol is intentionally open-ended: only ``image`` is mandatory.
All other keys are task-dependent and consumed by the corresponding model heads
and loss modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


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

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a data sample as a dict."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

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
