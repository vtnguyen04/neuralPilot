"""Modular task registry package: one clean module per driving task family."""

from __future__ import annotations

from neuro_pilot.tasks.base import (
    BaseTask,
    GraphDrivingTask,
    infer_driving_task,
    load_compiled_zoo_backend,
    maybe_build_zoo_graph,
    save_graph_checkpoint,
)
from neuro_pilot.tasks.multitask import (
    DetectionTask,
    HeatmapTask,
    MultiTask,
    SingleLossTask,
    TrajectoryTask,
)
from neuro_pilot.tasks.perception import CipoTask, DepthTask, LaneTask, SegTask
from neuro_pilot.tasks.planner import BFMCTask, PlannerTask
from neuro_pilot.tasks.world import DriverTask, WorldTask


def register_driving_tasks() -> None:
    """Ensure all driving tasks and Zoo losses are registered in `Registry`."""
    import neuro_pilot.nn.zoo.losses  # noqa: F401


__all__ = [
    "BaseTask",
    "GraphDrivingTask",
    "MultiTask",
    "SingleLossTask",
    "DetectionTask",
    "TrajectoryTask",
    "HeatmapTask",
    "PlannerTask",
    "BFMCTask",
    "SegTask",
    "DepthTask",
    "LaneTask",
    "CipoTask",
    "WorldTask",
    "DriverTask",
    "infer_driving_task",
    "load_compiled_zoo_backend",
    "maybe_build_zoo_graph",
    "register_driving_tasks",
    "save_graph_checkpoint",
]
