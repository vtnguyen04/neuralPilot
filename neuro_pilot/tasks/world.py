"""Generative driving world models and cabin driver monitoring tasks."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from neuro_pilot.core.registry import Registry
from neuro_pilot.tasks.base import GraphDrivingTask


@Registry.register_task("world")
@Registry.register_task("world_model")
class WorldTask(GraphDrivingTask):
    """Generative driving world models (DiT, GPT, VQGAN tokenizer, V-JEPA)."""

    task_name: str = "world"
    default_cfg: str = "dit"
    default_loss: str = "flow_matching"

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_world_frame

        return render_world_frame(frame_bgr, output, **kwargs)


@Registry.register_task("driver")
@Registry.register_task("dmonitoring")
class DriverTask(GraphDrivingTask):
    """In-cabin driver monitoring task (head pose, gaze, distraction)."""

    task_name: str = "driver"
    default_cfg: str = "driver"
    default_loss: str = "laplace_nll"

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_driver_frame

        return render_driver_frame(frame_bgr, output, **kwargs)
