"""Backward-compatible re-exports of GraphDriving tasks from `neuro_pilot.tasks`."""

from __future__ import annotations

from neuro_pilot.tasks import (
    BFMCTask,
    CipoTask,
    DepthTask,
    DriverTask,
    GraphDrivingTask,
    LaneTask,
    PlannerTask,
    SegTask,
    WorldTask,
    infer_driving_task,
    load_compiled_zoo_backend,
    maybe_build_zoo_graph,
    register_driving_tasks,
    save_graph_checkpoint,
)

__all__ = [
    "BFMCTask",
    "CipoTask",
    "DepthTask",
    "DriverTask",
    "GraphDrivingTask",
    "LaneTask",
    "PlannerTask",
    "SegTask",
    "WorldTask",
    "infer_driving_task",
    "load_compiled_zoo_backend",
    "maybe_build_zoo_graph",
    "register_driving_tasks",
    "save_graph_checkpoint",
]
