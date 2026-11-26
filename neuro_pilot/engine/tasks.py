"""Plug-and-play self-driving tasks registered into NeuroPilot's native Registry.

Adding a new self-driving task requires only:
1. A layer-by-layer YAML config in `cfg/models/zoo/` (composing any blocks).
2. A 4-line subclass of `GraphDrivingTask` decorated with `@Registry.register_task(...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from neuro_pilot.core.registry import Registry
from neuro_pilot.engine.predictors import BaseGraphPredictor, create_zoo_predictor
from neuro_pilot.nn.zoo import MODEL_YAML_MAP, GraphModel, build_zoo_model, is_zoo_yaml


def infer_driving_task(model_key: str, family: str = "") -> str:
    """Infer the default self-driving task name from a Zoo model key or family."""
    key = model_key.lower()
    if key == "driver" or "dmonitoring" in key or family == "cabin":
        return "driver"
    if key == "depth" or "scene3d" in key:
        return "depth"
    if key in {"seg", "seg_l"} or "ddrnet" in key or "eomt" in key:
        return "seg"
    if key == "lane" or "autosteer" in key:
        return "lane"
    world_keys = ("jepa", "vq_", "ae", "dit", "gpt", "comma_", "commavq", "latent_drive")
    if family in {"world", "world_models"} or any(
        key == k or (len(k) > 3 and k in key) for k in world_keys
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
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        if is_zoo_yaml(yaml.safe_load(path.read_text(encoding="utf-8"))):
            return build_zoo_model(path, scale=zoo_scale)
    return None


def save_graph_checkpoint(model: GraphModel, filename: Union[str, Path]) -> None:
    """Save a GraphModel checkpoint compatible with `NeuroPilot(path)` and `ModelZoo.load()`."""
    torch.save(
        {
            "model_key": model.model_key,
            "family": model.family,
            "config": {},
            "state_dict": model.state_dict(),
        },
        filename,
    )


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


class GraphDrivingTask:
    """Base plug-and-play task for YAML GraphModel architectures (Ultralytics-style)."""

    task_name: str = "driving"
    default_cfg: str = "pilot_s"
    default_loss: str = "laplace_nll"

    def __init__(
        self,
        cfg: Any = None,
        overrides: Optional[Dict[str, Any]] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.overrides: Dict[str, Any] = overrides or {}
        self.backbone = backbone
        self.model: Optional[GraphModel] = None
        self.criterion: Optional[nn.Module] = None

    def build_model(self) -> GraphModel:
        model_cfg = self.overrides.get("model_cfg") or self.default_cfg
        scale = self.overrides.get("scale")
        pt_path = self.overrides.get("weights")
        self.model = build_zoo_model(model_cfg, scale=scale, pt_path=pt_path)
        return self.model

    def build_criterion(self) -> nn.Module:
        loss_key = self.overrides.get("loss_name", self.default_loss)
        loss_cls = Registry.get_loss(loss_key)
        if loss_cls is None:
            raise KeyError(f"Unknown loss '{loss_key}'. Register it with @Registry.register_loss.")
        self.criterion = loss_cls()
        return self.criterion

    def load_weights(self, weights_path: Union[str, Path, dict]) -> None:
        if self.model is None:
            self.build_model()
        assert self.model is not None
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

    def predict(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.get_predictor()(source, **kwargs)


@Registry.register_task("planner")
@Registry.register_task("e2e_planning")
class PlannerTask(GraphDrivingTask):
    """End-to-end driving policy and trajectory planning."""

    task_name = "planner"
    default_cfg = "pilot_s"
    default_loss = "mhp_laplace"


@Registry.register_task("world")
@Registry.register_task("world_model")
class WorldTask(GraphDrivingTask):
    """Generative driving world models (DiT, GPT, VQ/ViT-AE, V-JEPA)."""

    task_name = "world"
    default_cfg = "dit"
    default_loss = "flow_matching"


@Registry.register_task("lane")
class LaneTask(GraphDrivingTask):
    """Ego-lane / path geometry prediction."""

    task_name = "lane"
    default_cfg = "lane"
    default_loss = "waypoint"


@Registry.register_task("depth")
class DepthTask(GraphDrivingTask):
    """Monocular depth estimation."""

    task_name = "depth"
    default_cfg = "depth"
    default_loss = "laplace_nll"


@Registry.register_task("seg")
class SegTask(GraphDrivingTask):
    """19-class semantic road and obstacle segmentation."""

    task_name = "seg"
    default_cfg = "seg"
    default_loss = "laplace_nll"


@Registry.register_task("driver")
@Registry.register_task("dmonitoring")
class DriverTask(GraphDrivingTask):
    """In-cabin driver monitoring."""

    task_name = "driver"
    default_cfg = "driver"
    default_loss = "laplace_nll"


@Registry.register_task("bfmc")
class BFMCTask(GraphDrivingTask):
    """Command-conditioned 1:10 RC autonomous driving fine-tuning."""

    task_name = "bfmc"
    default_cfg = "pilot_s"
    default_loss = "waypoint"


def register_driving_tasks() -> None:
    """Ensure all self-driving tasks and losses are registered in `Registry`."""
    import neuro_pilot.nn.zoo.losses  # noqa: F401
