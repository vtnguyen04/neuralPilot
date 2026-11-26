"""Shared atomic block registry imports for NeuroPilot YAML graphs."""

from __future__ import annotations

import importlib

_BLOCK_MODULES = (
    "neuro_pilot.nn.zoo.blocks.common",
    "neuro_pilot.nn.zoo.blocks.transformer",
    "neuro_pilot.nn.zoo.blocks.cnn",
    "neuro_pilot.nn.zoo.blocks.effnet",
    "neuro_pilot.nn.zoo.blocks.vision",
    "neuro_pilot.nn.zoo.blocks.vq",
    "neuro_pilot.nn.zoo.blocks.latent",
    "neuro_pilot.nn.zoo.blocks.planner",
    "neuro_pilot.nn.zoo.blocks.perception",
    "neuro_pilot.nn.zoo.blocks.seg",
    "neuro_pilot.nn.zoo.blocks.eomt",
    "neuro_pilot.nn.zoo.blocks.vit_ae",
    "neuro_pilot.nn.zoo.blocks.world",
)


def register_all_blocks() -> None:
    """Import every available block module so BLOCK_REGISTRY is populated."""
    for mod in _BLOCK_MODULES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            pass
