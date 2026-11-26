"""PyTorch runtime backend supporting nn.Module, GraphModel, and `.pt` checkpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from neuro_pilot.engine.backend.base import BaseBackend
from neuro_pilot.utils.logger import logger


class PyTorchBackend(BaseBackend):
    """PyTorch backend supporting both legacy multitask modules and Zoo GraphModels."""

    def __init__(
        self,
        weights: Union[str, nn.Module],
        device: torch.device,
        fp16: bool = False,
        fuse: bool = False,
        model_key: str = "",
    ) -> None:
        super().__init__(weights, device, fp16=fp16, model_key=model_key)

        if isinstance(weights, nn.Module):
            self.model = weights
        else:
            logger.info(f"Loading PyTorch model from {weights}")
            loaded = torch.load(weights, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "model_key" in loaded:
                from neuro_pilot.nn.zoo import load_zoo_pt

                self.model = load_zoo_pt(weights, device=str(device))
            elif isinstance(loaded, nn.Module):
                self.model = loaded
            else:
                raise ValueError(
                    f"Weights {weights} is a dict without 'model_key'; pass YAML or nn.Module."
                )

        self.model.to(self.device).eval()
        self.model_key = getattr(self.model, "model_key", self.model_key)
        self.family = getattr(self.model, "family", self.family)
        self.input_specs = getattr(self.model, "input_specs", {})
        if hasattr(self.model, "names"):
            self.names = self.model.names
        if hasattr(self.model, "nc"):
            self.nc = self.model.nc
        elif isinstance(self.names, (list, dict)):
            self.nc = len(self.names)

        if fuse and hasattr(self.model, "fuse"):
            logger.info("Fusing model layers for PyTorch inference...")
            self.model.fuse()
        if self.fp16:
            self.model.half()

    def forward_onnx_dict(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feed = {}
        for name, tensor in inputs.items():
            t = tensor.to(self.device)
            if self.fp16 and t.is_floating_point():
                t = t.half()
            feed[name] = t
        with torch.no_grad():
            return self.model.forward_onnx_dict(feed)

    def forward(
        self, im: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.fp16 and im.is_floating_point() and im.dtype != torch.float16:
            im = im.half()
        im = im.to(self.device)
        with torch.no_grad():
            return self.model(im, **kwargs)

    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        if self.warmup_done:
            return
        if hasattr(self.model, "make_sample_inputs"):
            self.forward_onnx_dict(self.model.make_sample_inputs(seed=0))
        else:
            self.forward(torch.zeros(imgsz, dtype=torch.float32, device=self.device))
        self.warmup_done = True
