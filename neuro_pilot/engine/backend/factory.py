"""Factory routing weights/modules to PyTorch, ONNX, TensorRT, or TorchScript backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from neuro_pilot.engine.backend.base import BaseBackend
from neuro_pilot.engine.backend.pytorch import PyTorchBackend
from neuro_pilot.utils.logger import logger


class AutoBackend:
    """Select and instantiate the appropriate inference backend by file suffix or type."""

    @classmethod
    def create(
        cls,
        weights: Union[str, Path, nn.Module],
        device: Optional[torch.device] = None,
        fp16: bool = False,
        fuse: bool = False,
        model_key: str = "",
    ) -> BaseBackend:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(weights, nn.Module):
            return PyTorchBackend(weights, device, fp16=fp16, fuse=fuse, model_key=model_key)

        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".engine", ".plan"}:
            logger.info(f"Detected TensorRT engine: {path}")
            from neuro_pilot.engine.backend.tensorrt import TensorRTBackend

            return TensorRTBackend(str(path), device, fp16=fp16, model_key=model_key)
        if suffix == ".onnx":
            logger.info(f"Detected ONNX model: {path}")
            from neuro_pilot.engine.backend.onnx import ONNXBackend

            return ONNXBackend(str(path), device, fp16=fp16, model_key=model_key)
        if suffix in {".torchscript", ".ts"}:
            logger.info(f"Detected TorchScript model: {path}")
            from neuro_pilot.engine.backend.torchscript import TorchScriptBackend

            return TorchScriptBackend(str(path), device, fp16=fp16, model_key=model_key)
        if suffix in {".pt", ".pth"}:
            logger.info(f"Detected PyTorch checkpoint: {path}")
            return PyTorchBackend(str(path), device, fp16=fp16, fuse=fuse, model_key=model_key)
        raise NotImplementedError(f"Unsupported model format: {suffix}")
