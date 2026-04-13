"""Dataset package — auto-discovers all registered dataset adapters.

Each adapter uses ``@register_dataset("type_name")`` so it is added to
``DATASET_REGISTRY`` at import time.  The factory in
``neuro_pilot_dataset.py`` does a simple dict lookup — no if/elif chains.
"""

from .base import BaseDrivingDataset, DATASET_REGISTRY, register_dataset

# Import each adapter module so its @register_dataset decorator executes.
from . import covla_local   # → "covla_local"
from . import covla_hf      # → "covla_hf"

__all__ = [
    "BaseDrivingDataset",
    "DATASET_REGISTRY",
    "register_dataset",
]
