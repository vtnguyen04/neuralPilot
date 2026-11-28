"""End-to-end driving planner tasks (dual-cam OpenPilot & single-cam multi-task MHP)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from neuro_pilot.core.registry import Registry
from neuro_pilot.tasks.base import GraphDrivingTask


@Registry.register_task("planner")
@Registry.register_task("e2e_planning")
class PlannerTask(GraphDrivingTask):
    """End-to-end trajectory planning and joint traffic-sign/vehicle detection task."""

    task_name: str = "planner"
    default_cfg: str = "pilot_mhp"
    default_loss: str = "mhp_laplace"

    def get_trainer(self) -> Any:
        from neuro_pilot.engine.planner_trainer import PlannerTrainer

        if self.model is None:
            self.build_model()
        return PlannerTrainer(model=self.model, overrides=self.overrides, task_name=self.task_name)

    def get_validator(self) -> Any:
        from neuro_pilot.engine.planner_eval import PlannerValidator

        if self.model is None:
            self.build_model()
        return PlannerValidator(model=self.model, overrides=self.overrides)

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_planner_frame

        return render_planner_frame(frame_bgr, output, **kwargs)


@Registry.register_task("bfmc")
class BFMCTask(PlannerTask):
    """Command-conditioned 1:10 autonomous driving alias of PlannerTask."""

    task_name: str = "bfmc"
    default_cfg: str = "pilot_mhp"
    default_loss: str = "mhp_laplace"
