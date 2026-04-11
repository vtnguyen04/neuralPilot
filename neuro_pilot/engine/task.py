from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.torch_utils import default_names
from neuro_pilot.core.registry import Registry

class BaseTask(ABC):
    """
    Abstract Base Class for NeuroPilot Tasks.
    """

    def __init__(
        self, cfg: Any, overrides: Dict[str, Any] = None, backbone: nn.Module = None
    ):
        self.cfg = cfg
        self.overrides: Dict[str, Any] = overrides or {}
        self.backbone = backbone
        self.model = None
        self.criterion = None
        self.trainer = None
        self.validator = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model architecture."""
        pass

    @abstractmethod
    def build_criterion(self) -> nn.Module:
        """Construct and return the loss function."""
        pass

    @abstractmethod
    def get_trainer(self) -> Any:
        """Return a Trainer instance configured for this task."""
        pass

    @abstractmethod
    def get_validator(self) -> Any:
        """Return a Validator instance configured for this task."""
        pass

    def load_weights(self, weights_path: Union[str, Path]):
        """Load weights into the model."""
        pass


from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.engine.validator import Validator
from neuro_pilot.utils.torch_utils import load_checkpoint

@Registry.register_task("multitask")
class MultiTask(BaseTask):
    """
    E2E Multitask (Detection + Trajectory + Heatmap).
    """

    def __init__(self, cfg, overrides=None, backbone=None):
        super().__init__(cfg, overrides, backbone)
        self.names = default_names(self.cfg.head.num_classes)
        if hasattr(self.cfg.data, "dataset_yaml") and self.cfg.data.dataset_yaml:
            from neuro_pilot.data.utils import check_dataset

            try:
                data_cfg = check_dataset(self.cfg.data.dataset_yaml)
                if "names" in data_cfg:
                    names = data_cfg["names"]
                    if isinstance(names, list):
                        self.names = {i: name for i, name in enumerate(names)}
                    elif isinstance(names, dict):
                        # Ensure keys are integers
                        self.names = {int(k): v for k, v in names.items()}
                    else:
                        self.names = names
                    logger.debug(f"MultiTask loaded {len(self.names)} names from {self.cfg.data.dataset_yaml}: {list(self.names.values())[:3]}...")
                else:
                    logger.warning(f"No 'names' key found in {self.cfg.data.dataset_yaml}")
            except Exception as e:
                logger.warning(
                    f"Failed to load names from {self.cfg.data.dataset_yaml}: {e}"
                )

    def build_model(self) -> nn.Module:
        from neuro_pilot.nn.factory import build_model

        model_cfg = self.overrides.get("model_cfg")
        skip_heatmap = self.overrides.get(
            "skip_heatmap_inference", self.cfg.head.skip_heatmap_inference
        )

        cfg_path = model_cfg if (model_cfg and str(model_cfg).endswith((".yaml", ".yml"))) else "neuro_pilot/cfg/models/neuralPilot.yaml"
        verbose = bool(model_cfg)

        self.model = build_model(
            cfg_path=cfg_path,
            ch=3,
            nc=self.cfg.head.num_classes,
            names=self.names,
            verbose=verbose,
            skip_heatmap_inference=skip_heatmap,
        )
        return self.model

    def build_criterion(self) -> nn.Module:
        device = next(self.model.parameters()).device if self.model else None
        self.criterion = CombinedLoss(self.cfg, self.model, device=device)
        return self.criterion

    def get_trainer(self) -> Trainer:
        if self.criterion is None:
            self.build_criterion()
        trainer = Trainer(self.cfg, overrides=self.overrides)
        trainer.criterion = self.criterion
        if self.model:
            trainer.model = self.model
        return trainer

    def get_validator(self) -> Validator:
        if self.criterion is None:
            self.build_criterion()
        device = self.overrides.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        v = Validator(self.cfg, self.model, self.criterion, device=device)
        v.names = self.names
        return v

    def load_weights(self, weights_path: Union[str, Path]):
        if self.model is None:
            self.build_model()
        load_checkpoint(weights_path, self.model)

class SingleLossTask(MultiTask):
    """Base for tasks that zero out specific losses.

    Subclasses define ``_zeroed_losses`` as a dict of loss keys to disable.
    """
    _zeroed_losses: dict = {}
    _task_label: str = ""

    def __init__(self, cfg, overrides=None, backbone=None):
        from neuro_pilot.cfg.schema import deep_update
        loss_overrides = {"loss": dict(self._zeroed_losses)}
        overrides = deep_update(overrides or {}, loss_overrides)
        super().__init__(cfg, overrides, backbone)
        if self._task_label:
            logger.info(f"Task: {self._task_label}")


@Registry.register_task("detection")
class DetectionTask(SingleLossTask):
    """Detection-only training. Disables trajectory, heatmap, classification, and gate losses."""
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
    """Trajectory-only training. Disables detection loss."""
    _zeroed_losses = {"lambda_det": 0}
    _task_label = "Trajectory-only (detection loss disabled)"
