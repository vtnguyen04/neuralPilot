"""Abstract base class and shared contract for all NeuroPilot inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from neuro_pilot.utils.torch_utils import default_names


def load_sidecar_metadata(weights_path: Union[str, Path]) -> Dict[str, Any]:
    """Load optional `<weights>.meta.json` sidecar metadata when present."""
    meta_path = Path(f"{weights_path}.meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def save_sidecar_metadata(weights_path: Union[str, Path], meta: Dict[str, Any]) -> Path:
    """Persist lightweight metadata sidecar alongside exported ONNX/TRT/TS binaries."""
    meta_path = Path(f"{weights_path}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


class BaseBackend(ABC):
    """Unified inference backend contract for PyTorch, ONNX, TensorRT, and TorchScript."""

    def __init__(
        self,
        weights: Any,
        device: torch.device,
        fp16: bool = False,
        model_key: str = "",
    ) -> None:
        self.weights = weights
        self.device = device
        self.fp16 = fp16
        self.warmup_done = False
        self.model_key: str = model_key
        self.family: str = ""
        self.input_specs: Dict[str, Any] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        from neuro_pilot.cfg.schema import HeadConfig

        self.names = default_names(HeadConfig().num_classes)

    @abstractmethod
    def forward(
        self, im: Union[torch.Tensor, np.ndarray], **kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Execute legacy single-image (optional cmd) inference."""

    def forward_onnx_dict(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute named-dict inference for multi-input GraphModel architectures."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forward_onnx_dict")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self.forward_onnx_dict(args[0])
        return self.forward(*args, **kwargs)

    @abstractmethod
    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """Warm up runtime buffers and kernels."""

    def to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to contiguous CPU numpy array."""
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def to_tensor(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert numpy array to tensor on backend device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return x.to(self.device)

    def _apply_metadata(self, meta: Dict[str, Any], explicit_key: Optional[str] = None) -> None:
        """Populate model_key, family, input_specs, and output_names from metadata."""
        if explicit_key:
            self.model_key = explicit_key
        elif meta.get("model_key"):
            self.model_key = str(meta["model_key"])
        if meta.get("family"):
            self.family = str(meta["family"])
        if isinstance(meta.get("input_specs"), dict) and meta["input_specs"]:
            self.input_specs = meta["input_specs"]
        if isinstance(meta.get("output_names"), list) and meta["output_names"]:
            self.output_names = list(meta["output_names"])
