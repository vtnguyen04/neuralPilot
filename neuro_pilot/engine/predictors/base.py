"""Base GraphModel predictor and plug-and-play predictor registry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import ZooRegistry

PREDICTOR_REGISTRY: ZooRegistry["BaseGraphPredictor"] = ZooRegistry("zoo_predictor")


class BaseGraphPredictor:
    """Unified runtime predictor for any GraphModel or compiled backend (ONNX/TRT/TS).

    Subclasses override `preprocess()` and `postprocess()` for domain-specific
    sensor inputs (camera frames, dual-cam queues, video clips, token grids).
    Passing a pre-built `dict` of named tensors always bypasses preprocessing.
    """

    def __init__(
        self,
        model: Any,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if isinstance(model, nn.Module):
            first_param = next(model.parameters(), None)
            default_dev = first_param.device if first_param is not None else torch.device("cpu")
            self.device = torch.device(device) if device is not None else default_dev
            self.model = model.to(self.device).eval()
        else:
            default_dev = getattr(model, "device", torch.device("cpu"))
            self.device = torch.device(device) if device is not None else torch.device(default_dev)
            self.model = model

    def reset_state(self) -> None:
        """Reset any temporal queues or buffers maintained by stateful predictors."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Convert user input into the exact named tensor dict expected by `GraphModel`."""
        if isinstance(source, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in source.items()
            }
        if isinstance(source, torch.Tensor):
            first_key = next(iter(self.model.input_specs.keys()))
            return {first_key: source.to(self.device)}
        raise TypeError(
            f"{type(self).__name__} expects dict or Tensor, got {type(source).__name__}"
        )

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        """Convert raw model output tensors into user-friendly numpy/dict outputs."""
        return {k: v.detach().cpu().numpy() for k, v in raw_outputs.items()}

    def predict(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run end-to-end preprocess -> no-grad forward -> postprocess."""
        feed = self.preprocess(source, **kwargs)
        if isinstance(self.model, nn.Module):
            self.model.eval()
        with torch.no_grad():
            raw = self.model.forward_onnx_dict(feed)
        return self.postprocess(raw, **kwargs)

    def __call__(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.predict(source, **kwargs)
