"""Deterministic ONNX-to-GraphModel (.pt) weight converters."""

from __future__ import annotations

import importlib

_CONVERTER_MODULES = (
    "neuro_pilot.nn.zoo.converters.planner",
    "neuro_pilot.nn.zoo.converters.perception",
    "neuro_pilot.nn.zoo.converters.seg",
    "neuro_pilot.nn.zoo.converters.eomt",
    "neuro_pilot.nn.zoo.converters.world",
)


def register_all_converters() -> None:
    """Import every available converter module so CONVERTER_REGISTRY is populated."""
    for mod in _CONVERTER_MODULES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            pass
