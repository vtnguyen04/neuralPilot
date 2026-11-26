"""TorchScript runtime backend supporting exported GraphModel `.torchscript` files."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Union

import torch

from neuro_pilot.engine.backend.base import BaseBackend, load_sidecar_metadata


class TorchScriptBackend(BaseBackend):
    """Run traced `.torchscript` / `.ts` GraphModels with embedded I/O metadata."""

    def __init__(
        self,
        weights: str,
        device: torch.device,
        fp16: bool = False,
        model_key: str = "",
    ) -> None:
        super().__init__(weights, device, fp16=fp16, model_key=model_key)
        extra = {"neuropilot_meta.json": ""}
        self.module = torch.jit.load(str(weights), map_location=device, _extra_files=extra).eval()
        if self.fp16:
            self.module.half()

        embedded: Dict[str, Any] = {}
        if extra["neuropilot_meta.json"]:
            raw_json = extra["neuropilot_meta.json"]
            if isinstance(raw_json, bytes):
                raw_json = raw_json.decode("utf-8")
            embedded = json.loads(raw_json)
        sidecar = load_sidecar_metadata(weights)
        meta = {**embedded, **sidecar}
        self._apply_metadata(meta, explicit_key=model_key)
        self.input_names = list(meta.get("input_names") or list(self.input_specs.keys()))

    def forward_onnx_dict(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        ordered = []
        for name in self.input_names:
            tensor = inputs[name].to(self.device)
            if self.fp16 and tensor.is_floating_point():
                tensor = tensor.half()
            ordered.append(tensor)
        with torch.no_grad():
            outs = self.module(*ordered)
        if not isinstance(outs, tuple):
            outs = (outs,)
        names = self.output_names or [f"output_{i}" for i in range(len(outs))]
        return dict(zip(names, outs))

    def forward(
        self, im: Union[torch.Tensor, Any], **kwargs: Any
    ) -> Any:
        tensor = self.to_tensor(im)
        if self.input_names:
            return self.forward_onnx_dict({self.input_names[0]: tensor})
        with torch.no_grad():
            return self.module(tensor)

    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        if self.warmup_done:
            return
        if self.input_specs:
            dummy = {
                k: torch.zeros(tuple(v["shape"]), dtype=getattr(torch, v["dtype"]))
                for k, v in self.input_specs.items()
            }
            self.forward_onnx_dict(dummy)
        else:
            self.forward(torch.zeros(imgsz, dtype=torch.float32, device=self.device))
        self.warmup_done = True
