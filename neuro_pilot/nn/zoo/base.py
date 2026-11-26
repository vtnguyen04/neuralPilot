"""Core registries, base classes, and parity DTOs for NeuroPilot ModelZoo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn

T = TypeVar("T")


class ZooRegistry(Generic[T]):
    """Typed name-to-class registry supporting decorator registration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        def _decorator(obj: Any) -> Any:
            self._items[key] = obj
            return obj

        return _decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            avail = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown '{key}' in {self._name}. Available: [{avail}]")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> T:
        return self.get(key)(*args, **kwargs)

    def keys(self) -> List[str]:
        return sorted(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items


@dataclass
class OutputParityMetric:
    """Numerical parity metrics for a single output tensor (PyTorch .pt vs ONNX Runtime)."""

    name: str
    shape: Tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    rel_l2_err: float
    cosine_sim: float
    passed: bool


@dataclass
class ParityReport:
    """Full ONNX -> PyTorch .pt conversion and numerical equivalence report."""

    model_key: str
    onnx_path: str
    pt_path: str
    num_params: int
    mapped_tensors: int
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    outputs: List[OutputParityMetric] = field(default_factory=list)
    all_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "onnx_path": self.onnx_path,
            "pt_path": self.pt_path,
            "num_params": self.num_params,
            "mapped_tensors": self.mapped_tensors,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "all_passed": self.all_passed,
            "outputs": [
                {
                    "name": o.name,
                    "shape": list(o.shape),
                    "max_abs_diff": round(o.max_abs_diff, 6),
                    "mean_abs_diff": round(o.mean_abs_diff, 6),
                    "rel_l2_err": round(o.rel_l2_err, 6),
                    "cosine_sim": round(o.cosine_sim, 7),
                    "passed": o.passed,
                }
                for o in self.outputs
            ],
        }


class BaseZooModel(nn.Module, ABC):
    """Base nn.Module for every E2E & World Model in NeuroPilot Zoo."""

    model_key: str = ""
    family: str = ""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_pt_checkpoint(self, pt_path: str | Path, strict: bool = True) -> Any:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        return self.load_state_dict(state, strict=strict)

    @abstractmethod
    def make_sample_inputs(self, seed: int = 42) -> Dict[str, torch.Tensor]:
        """Return deterministic sample inputs keyed by exact tensor names."""

    @abstractmethod
    def forward_onnx_dict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward pass accepting named input dict and returning named output dict."""


class BaseOnnxConverter(ABC):
    """Converter mapping an official ONNX checkpoint into a GraphModel state_dict."""

    model_key: str = ""
    yaml_name: str = ""

    def build_model(self, config: Optional[Dict[str, Any]] = None) -> BaseZooModel:
        """Build the target GraphModel from its layer-by-layer YAML config."""
        from neuro_pilot.nn.zoo import build_zoo_model

        return build_zoo_model(self.yaml_name or self.model_key, **(config or {}))

    @abstractmethod
    def extract_state_dict(
        self,
        onnx_tensors: Dict[str, Any],
        model: BaseZooModel,
    ) -> Dict[str, torch.Tensor]:
        """Map ONNX initializer arrays into exact PyTorch state_dict keys & shapes."""


BLOCK_REGISTRY: ZooRegistry[nn.Module] = ZooRegistry("zoo_block")
ZOO_REGISTRY: ZooRegistry[BaseZooModel] = ZooRegistry("zoo_model")
CONVERTER_REGISTRY: ZooRegistry[BaseOnnxConverter] = ZooRegistry("onnx_converter")
