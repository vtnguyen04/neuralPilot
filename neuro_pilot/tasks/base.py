"""Unified BaseTask and GraphDrivingTask interfaces for all NeuroPilot tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from neuro_pilot.core.registry import Registry
from neuro_pilot.engine.predictors import BaseGraphPredictor, create_zoo_predictor
from neuro_pilot.nn.zoo import MODEL_YAML_MAP, GraphModel, build_zoo_model, is_zoo_yaml


class BaseTask(ABC):
    """Abstract base class unifying native multi-task and modular Zoo driving tasks."""

    task_name: str = "base"

    def __init__(
        self,
        cfg: Any = None,
        overrides: Optional[Dict[str, Any]] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.overrides: Dict[str, Any] = overrides or {}
        self.backbone = backbone
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.trainer: Any = None
        self.validator: Any = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the task model."""

    @abstractmethod
    def build_criterion(self) -> nn.Module:
        """Construct and return the task loss module."""

    @abstractmethod
    def get_trainer(self) -> Any:
        """Return a configured trainer instance for this task."""

    @abstractmethod
    def get_validator(self) -> Any:
        """Return a configured validator instance for this task."""

    def load_weights(self, weights_path: Union[str, Path, dict]) -> None:
        """Load checkpoint weights into `self.model`."""
        return None

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        """Render task-specific visual overlays onto `frame_bgr`."""
        from neuro_pilot.utils.visualize import visualize_prediction

        return visualize_prediction(self.task_name, frame_bgr, output, **kwargs)


def infer_driving_task(model_key: str, family: str = "") -> str:
    """Infer the canonical driving task name from a Zoo model key or family."""
    key = model_key.lower()
    if key == "driver" or "dmonitoring" in key or family == "cabin":
        return "driver"
    if key == "depth" or "scene3d" in key:
        return "depth"
    if key in {"seg", "seg_l"} or "ddrnet" in key or "eomt" in key:
        return "seg"
    if key == "lane" or "autosteer" in key:
        return "lane"
    if key in {"cipo", "drive"} or "autospeed" in key or "autodrive" in key:
        return "cipo"
    world_exact = {"jepa", "vq_enc", "vq_dec", "ae", "dit", "gpt"}
    world_sub = ("vq_", "comma_", "commavq", "latent_drive")
    if (
        family in {"world", "world_models"}
        or key in world_exact
        or any(k in key for k in world_sub)
    ):
        return "world"
    return "planner"


def maybe_build_zoo_graph(
    cfg_path: Union[str, Path], scale: str = "n"
) -> Optional[GraphModel]:
    """Return a GraphModel when `cfg_path` is a Zoo model key or Zoo YAML, else None."""
    zoo_scale = None if scale == "n" else scale
    raw_key = str(cfg_path)
    if raw_key in MODEL_YAML_MAP:
        return build_zoo_model(raw_key, scale=zoo_scale)
    path = Path(raw_key)
    if path.suffix.lower() in {".yaml", ".yml"} and path.stem in MODEL_YAML_MAP:
        return build_zoo_model(path.stem, scale=zoo_scale)
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        if is_zoo_yaml(yaml.safe_load(path.read_text(encoding="utf-8"))):
            return build_zoo_model(path, scale=zoo_scale)
    return None


def save_graph_checkpoint(
    model: GraphModel,
    filename: Union[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a GraphModel `.pt` checkpoint compatible with `NeuroPilot(path)`."""
    cfg_meta = {"scale": model.scale} if getattr(model, "scale", None) else {}
    payload: Dict[str, Any] = {
        "model_key": model.model_key,
        "family": model.family,
        "config": cfg_meta,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, filename)


def load_compiled_zoo_backend(
    model_path: Path,
    device: torch.device,
    overrides: Dict[str, Any],
    task: Optional[str] = None,
) -> tuple:
    """Create a compiled backend (ONNX/TRT/TS) and return `(backend, task_name, task_wrapper)`."""
    from neuro_pilot.engine.backend.factory import AutoBackend

    backend = AutoBackend.create(
        model_path,
        device=device,
        fp16=bool(overrides.get("half", False)),
        model_key=str(overrides.get("model_key", "")),
    )
    if backend.model_key:
        task_name = task or infer_driving_task(backend.model_key, backend.family)
        task_wrapper = Registry.get_task(task_name)(overrides=overrides)
        return backend, task_name, task_wrapper
    return backend, task or "multitask", None


class GraphDrivingTask(BaseTask):
    """Plug-and-play base task for modular YAML GraphModel architectures."""

    task_name: str = "driving"
    default_cfg: str = "pilot_s"
    default_loss: str = "laplace_nll"

    def build_model(self) -> GraphModel:
        model_cfg = self.overrides.get("model_cfg") or self.default_cfg
        scale = self.overrides.get("scale")
        if scale == "n":
            scale = None
        pt_path = self.overrides.get("weights")
        self.model = build_zoo_model(model_cfg, scale=scale, pt_path=pt_path)
        return self.model

    def build_criterion(self) -> nn.Module:
        loss_key = self.overrides.get("loss_name", self.default_loss)
        loss_cls = Registry.get_loss(loss_key)
        if loss_cls is None:
            raise KeyError(f"Unknown loss '{loss_key}'. Register with @Registry.register_loss.")
        self.criterion = loss_cls()
        return self.criterion

    def load_weights(self, weights_path: Union[str, Path, dict]) -> None:
        if self.model is None:
            self.build_model()
        assert isinstance(self.model, GraphModel)
        if isinstance(weights_path, dict):
            state = weights_path.get("state_dict", weights_path)
            self.model.load_state_dict(state, strict=True)
        else:
            self.model.load_pt_checkpoint(weights_path, strict=True)

    def get_predictor(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> BaseGraphPredictor:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        return create_zoo_predictor(self.model, device=device)

    def get_trainer(self) -> Any:
        raise NotImplementedError(
            f"Task '{self.task_name}' uses inference/adapter mode; override get_trainer() "
            "or attach a CommandAdapter via model.attach_adapter()."
        )

    def get_validator(self) -> Any:
        raise NotImplementedError(
            f"Task '{self.task_name}' does not define a standalone validator yet."
        )

    def predict(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.get_predictor()(source, **kwargs)
