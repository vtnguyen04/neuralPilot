"""ONNX Runtime backend supporting both multi-input GraphModels and legacy models."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch

from neuro_pilot.engine.backend.base import BaseBackend, load_sidecar_metadata

_ORT_DTYPE_MAP = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(uint8)": "uint8",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(bool)": "bool",
}


class ONNXBackend(BaseBackend):
    """ONNX Runtime backend with automatic metadata discovery and named-dict execution."""

    def __init__(
        self,
        weights: str,
        device: torch.device,
        fp16: bool = False,
        model_key: str = "",
    ) -> None:
        super().__init__(weights, device, fp16=fp16, model_key=model_key)

        avail = set(ort.get_available_providers())
        providers: List[str] = []
        if device.type == "cuda" and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(weights), sess_options=opts, providers=providers)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        self._input_types = {x.name: x.type for x in self.session.get_inputs()}
        self.input_specs = {
            x.name: {
                "shape": [1 if not isinstance(d, int) or d <= 0 else d for d in x.shape],
                "dtype": _ORT_DTYPE_MAP.get(x.type, "float32"),
            }
            for x in self.session.get_inputs()
        }

        meta = self._read_embedded_meta()
        sidecar = load_sidecar_metadata(weights)
        self._apply_metadata({**meta, **sidecar}, explicit_key=model_key)
        self._apply_legacy_meta(meta)

    def _read_embedded_meta(self) -> Dict[str, Any]:
        raw = self.session.get_modelmeta().custom_metadata_map
        parsed: Dict[str, Any] = dict(raw)
        for json_key in ("input_specs", "output_names"):
            if json_key in raw:
                try:
                    parsed[json_key] = json.loads(raw[json_key])
                except (ValueError, TypeError):
                    pass
        return parsed

    def _apply_legacy_meta(self, meta: Dict[str, Any]) -> None:
        if "names" in meta:
            try:
                self.names = ast.literal_eval(str(meta["names"]))
            except (ValueError, SyntaxError):
                pass
        if "stride" in meta:
            try:
                self.stride = [int(meta["stride"])]
            except ValueError:
                pass

    def _cast_input(self, name: str, arr: np.ndarray) -> np.ndarray:
        ort_type = self._input_types.get(name, "")
        if ort_type == "tensor(float16)" and arr.dtype != np.float16:
            return arr.astype(np.float16)
        if ort_type == "tensor(float)" and arr.dtype != np.float32:
            return arr.astype(np.float32)
        if ort_type == "tensor(int64)" and arr.dtype != np.int64:
            return arr.astype(np.int64)
        if ort_type == "tensor(uint8)" and arr.dtype != np.uint8:
            return arr.astype(np.uint8)
        return arr

    def forward_onnx_dict(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feed = {
            name: self._cast_input(name, self.to_numpy(inputs[name]))
            for name in self.input_names
            if name in inputs
        }
        outs = self.session.run(self.output_names, feed)
        return {
            name: torch.from_numpy(arr).to(self.device)
            for name, arr in zip(self.output_names, outs)
        }

    def forward(
        self, im: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        im_np = self._cast_input(self.input_names[0], self.to_numpy(im))
        feed: Dict[str, np.ndarray] = {self.input_names[0]: im_np}
        if len(self.input_names) > 1:
            cmd = kwargs.get("cmd", kwargs.get("command"))
            if cmd is None:
                from neuro_pilot.cfg.schema import HeadConfig

                cmd = torch.zeros((1, HeadConfig().num_commands), dtype=torch.float32)
                cmd[0, 0] = 1.0
            if cmd.ndim == 0:
                cmd = cmd.unsqueeze(0).unsqueeze(0)
            elif cmd.ndim == 1:
                cmd = cmd.unsqueeze(0)
            if cmd.shape[0] != im_np.shape[0]:
                cmd = cmd.repeat(im_np.shape[0], 1)
            feed[self.input_names[1]] = self._cast_input(self.input_names[1], self.to_numpy(cmd))

        outs = self.session.run(self.output_names, feed)
        tensors = [torch.from_numpy(o).to(self.device) for o in outs]
        if len(self.output_names) > 1:
            return dict(zip(self.output_names, tensors))
        return tensors[0]

    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        if self.warmup_done:
            return
        if self.model_key and self.input_specs:
            dummy = {
                k: torch.zeros(tuple(v["shape"]), dtype=getattr(torch, v["dtype"]))
                for k, v in self.input_specs.items()
            }
            self.forward_onnx_dict(dummy)
        else:
            self.forward(torch.zeros(imgsz, dtype=torch.float32, device=self.device))
        self.warmup_done = True
