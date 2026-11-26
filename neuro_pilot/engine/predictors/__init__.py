"""Plug-and-play predictors for all NeuroPilot Zoo GraphModel architectures."""

from __future__ import annotations

import importlib
from typing import Any, Optional, Union

import torch

from neuro_pilot.engine.predictors.base import PREDICTOR_REGISTRY, BaseGraphPredictor

_PREDICTOR_MODULES = (
    "neuro_pilot.engine.predictors.planner",
    "neuro_pilot.engine.predictors.perception",
    "neuro_pilot.engine.predictors.world",
)


def register_all_predictors() -> None:
    """Import all built-in predictor modules to populate `PREDICTOR_REGISTRY`."""
    for mod in _PREDICTOR_MODULES:
        importlib.import_module(mod)


def create_zoo_predictor(
    model: Any,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseGraphPredictor:
    """Instantiate the registered predictor for `model.model_key`, or fallback to base."""
    register_all_predictors()
    key = getattr(model, "model_key", "")
    if key in PREDICTOR_REGISTRY:
        return PREDICTOR_REGISTRY.build(key, model=model, device=device)
    return BaseGraphPredictor(model=model, device=device)


__all__ = [
    "PREDICTOR_REGISTRY",
    "BaseGraphPredictor",
    "create_zoo_predictor",
    "register_all_predictors",
]
