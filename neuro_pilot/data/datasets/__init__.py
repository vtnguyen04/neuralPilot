"""Dataset package — auto-discovers all registered dataset adapters.

Each adapter uses ``@register_dataset("type_name")`` so it is added to
``DATASET_REGISTRY`` at import time.  The factory in
``neuro_pilot_dataset.py`` does a simple dict lookup — no if/elif chains.
"""

from .base import BaseDrivingDataset, DATASET_REGISTRY, register_dataset

# Import each adapter module so its @register_dataset decorator executes.
from . import video_driving  # → "video_driving"

# Base classes for extension
from .video_dataset import BaseVideoDataset

__all__ = [
    "BaseDrivingDataset",
    "BaseVideoDataset",
    "DATASET_REGISTRY",
    "register_dataset",
]
