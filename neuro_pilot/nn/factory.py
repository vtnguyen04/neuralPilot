"""Centralized model construction factory (Dependency Inversion Principle).

All model instantiation should go through this module.
This avoids scattering ``from neuro_pilot.nn.tasks import DetectionModel``
across engine/model.py, engine/task.py, and engine/trainer.py.
"""

from __future__ import annotations

import torch.nn as nn
from neuro_pilot.utils.logger import logger


def build_model(
    cfg_path: str,
    ch: int = 3,
    nc: int | None = None,
    scale: str = "n",
    names: dict | None = None,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """Build a DetectionModel from a YAML config path.

    This is the single source of truth for model construction.
    All call-sites should use this instead of importing DetectionModel directly.
    """
    from neuro_pilot.nn.tasks import DetectionModel

    if nc is None:
        from neuro_pilot.cfg.schema import HeadConfig
        nc = HeadConfig().num_classes

    logger.info(f"Building model from {cfg_path} (scale={scale}, nc={nc})")
    return DetectionModel(
        cfg=cfg_path,
        ch=ch,
        nc=nc,
        scale=scale,
        names=names,
        verbose=verbose,
        **kwargs,
    )
