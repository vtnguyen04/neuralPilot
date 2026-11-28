"""Backward-compatible re-exports of native multi-task classes from `neuro_pilot.tasks`."""

from __future__ import annotations

from neuro_pilot.tasks import (
    BaseTask,
    DetectionTask,
    HeatmapTask,
    MultiTask,
    SingleLossTask,
    TrajectoryTask,
)

__all__ = [
    "BaseTask",
    "MultiTask",
    "SingleLossTask",
    "DetectionTask",
    "TrajectoryTask",
    "HeatmapTask",
]
