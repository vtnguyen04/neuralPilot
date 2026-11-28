"""Native multi-task and single-loss tasks (multitask, detection, trajectory, heatmap)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from neuro_pilot.core.registry import Registry
from neuro_pilot.tasks.base import BaseTask
from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.torch_utils import default_names, load_checkpoint


@Registry.register_task("multitask")
class MultiTask(BaseTask):
    """Native joint Detection + Trajectory + Heatmap multi-task pipeline."""

    task_name: str = "multitask"

    def __init__(
        self,
        cfg: Any,
        overrides: Optional[Dict[str, Any]] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(cfg, overrides, backbone)
        self.names = default_names(self.cfg.head.num_classes)
        ds_yaml = getattr(getattr(self.cfg, "data", None), "dataset_yaml", None)
        if ds_yaml:
            self._load_dataset_names(ds_yaml)

    def _load_dataset_names(self, dataset_yaml: str) -> None:
        from neuro_pilot.data.utils import check_dataset

        try:
            data_cfg = check_dataset(dataset_yaml)
            names = data_cfg.get("names")
            if isinstance(names, list):
                self.names = {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                self.names = {int(k): v for k, v in names.items()}
            elif names:
                self.names = names
        except Exception as exc:
            logger.warning(f"Failed to load names from {dataset_yaml}: {exc}")

    def build_model(self) -> nn.Module:
        from neuro_pilot.nn.factory import build_model

        model_cfg = self.overrides.get("model_cfg")
        skip_hm = self.overrides.get(
            "skip_heatmap_inference", self.cfg.head.skip_heatmap_inference
        )
        cfg_path = (
            model_cfg
            if (model_cfg and str(model_cfg).endswith((".yaml", ".yml")))
            else "neuro_pilot/cfg/models/neuralPilot.yaml"
        )
        self.model = build_model(
            cfg_path=cfg_path,
            ch=3,
            nc=self.cfg.head.num_classes,
            names=self.names,
            verbose=bool(model_cfg),
            skip_heatmap_inference=skip_hm,
        )
        return self.model

    def build_criterion(self) -> nn.Module:
        from neuro_pilot.utils.losses import MultiTaskLossManager

        device = next(self.model.parameters()).device if self.model else None
        self.criterion = MultiTaskLossManager(self.cfg, self.model, device=device)
        return self.criterion

    def get_trainer(self) -> Any:
        from neuro_pilot.engine.trainer import Trainer

        if self.criterion is None:
            self.build_criterion()
        trainer = Trainer(self.cfg, overrides=self.overrides)
        trainer.criterion = self.criterion
        if self.model:
            trainer.model = self.model
        return trainer

    def get_validator(self) -> Any:
        from neuro_pilot.engine.validator import Validator

        if self.criterion is None:
            self.build_criterion()
        device = self.overrides.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        validator = Validator(self.cfg, self.model, self.criterion, device=device)
        validator.names = self.names
        return validator

    def load_weights(self, weights_path: Union[str, Path, dict]) -> None:
        if self.model is None:
            self.build_model()
        if isinstance(weights_path, dict):
            state = (
                weights_path.get("model_state_dict")
                or weights_path.get("state_dict")
                or weights_path.get("model")
                or weights_path.get("ema")
                or weights_path
            )
            if isinstance(state, nn.Module):
                state = state.state_dict()
            assert self.model is not None
            self.model.load_state_dict(state, strict=False)
            return
        load_checkpoint(weights_path, self.model)


class SingleLossTask(MultiTask):
    """MultiTask specialization that zeros out non-target task losses."""

    _zeroed_losses: Dict[str, float] = {}
    _task_label: str = ""

    def __init__(
        self,
        cfg: Any,
        overrides: Optional[Dict[str, Any]] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        from neuro_pilot.cfg.schema import deep_update

        merged = deep_update(overrides or {}, {"loss": dict(self._zeroed_losses)})
        super().__init__(cfg, merged, backbone)
        if self._task_label:
            logger.info(f"Task: {self._task_label}")


@Registry.register_task("detection")
@Registry.register_task("detect")
class DetectionTask(SingleLossTask):
    """Detection-only training and validation (trajectory/heatmap/gate disabled)."""

    task_name: str = "detection"
    _zeroed_losses = {
        "lambda_traj": 0,
        "lambda_heatmap": 0,
        "lambda_cls": 0,
        "lambda_smooth": 0,
        "lambda_gate": 0,
    }
    _task_label = "Detection-only (trajectory/heatmap/cls losses disabled)"


@Registry.register_task("trajectory")
class TrajectoryTask(SingleLossTask):
    """Trajectory-only training and validation (detection loss disabled)."""

    task_name: str = "trajectory"
    _zeroed_losses = {"lambda_det": 0}
    _task_label = "Trajectory-only (detection loss disabled)"


@Registry.register_task("heatmap")
class HeatmapTask(SingleLossTask):
    """Path heatmap-only training (detection and direct waypoint losses disabled)."""

    task_name: str = "heatmap"
    _zeroed_losses = {"lambda_det": 0, "lambda_traj": 0, "lambda_cls": 0, "lambda_smooth": 0}
    _task_label = "Heatmap-only (detection/trajectory losses disabled)"
