"""TensorRT 10 zero-copy GPU backend supporting multi-input GraphModels and legacy models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import torch

from neuro_pilot.engine.backend.base import BaseBackend, load_sidecar_metadata
from neuro_pilot.utils.logger import logger


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt

        return trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT runtime requires 'tensorrt>=10.0.0' installed with CUDA."
        ) from exc


class TensorRTBackend(BaseBackend):
    """TensorRT 10 backend supporting named-dict multi-input GraphModels & dynamic shapes."""

    def __init__(
        self,
        weights: str,
        device: torch.device,
        fp16: bool = True,
        model_key: str = "",
    ) -> None:
        super().__init__(weights, device, fp16=fp16, model_key=model_key)
        self.trt = _import_tensorrt()
        self.trt_logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.trt.init_libnvinfer_plugins(self.trt_logger, "")

        logger.info(f"Loading TensorRT engine from {weights}")
        with open(weights, "rb") as handle, self.trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if not self.engine:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {weights}")

        self.context = self.engine.create_execution_context()
        self.bindings: Dict[str, torch.Tensor] = OrderedDict()
        self.stream = torch.cuda.Stream(device=device)
        self._allocate_buffers()

        sidecar = load_sidecar_metadata(weights)
        self._apply_metadata(sidecar, explicit_key=model_key)

    def _trt_to_torch_dtype(self, trt_dtype: Any) -> torch.dtype:
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if hasattr(trt.DataType, "UINT8"):
            mapping[trt.DataType.UINT8] = torch.uint8
        if hasattr(trt.DataType, "INT64"):
            mapping[trt.DataType.INT64] = torch.int64
        if trt_dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
        return mapping[trt_dtype]

    def _allocate_buffers(
        self, dynamic_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    ) -> None:
        trt = self.trt
        self.bindings = OrderedDict()
        self.input_names = []
        self.output_names = []
        specs: Dict[str, Any] = {}

        # First set all input shapes so TensorRT can resolve dependent output shapes
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            if dynamic_shapes and name in dynamic_shapes:
                shape = tuple(int(d) for d in dynamic_shapes[name])
            else:
                raw = tuple(int(d) for d in self.engine.get_tensor_shape(name))
                shape = tuple(1 if d < 0 else d for d in raw)
            self.context.set_input_shape(name, shape)
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            self.bindings[name] = torch.zeros(shape, dtype=dtype, device=self.device).contiguous()
            self.input_names.append(name)
            specs[name] = {"shape": list(shape), "dtype": str(dtype).replace("torch.", "")}

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            raw = tuple(int(d) for d in self.context.get_tensor_shape(name))
            shape = tuple(max(1, d) for d in raw)
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            self.bindings[name] = torch.zeros(shape, dtype=dtype, device=self.device).contiguous()
            self.output_names.append(name)

        if not self.input_specs:
            self.input_specs = specs

    def get_io_info(self) -> Dict[str, Tuple[int, ...]]:
        """Return `{tensor_name: shape}` for all engine IO bindings."""
        return {name: tuple(t.shape) for name, t in self.bindings.items()}

    def forward_onnx_dict(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        need_realloc = any(
            tuple(inputs[name].shape) != tuple(self.bindings[name].shape)
            for name in self.input_names
        )
        if need_realloc:
            self._allocate_buffers({name: tuple(inputs[name].shape) for name in self.input_names})

        for name in self.input_names:
            buf = self.bindings[name]
            src = inputs[name].to(device=self.device, dtype=buf.dtype).contiguous()
            buf.copy_(src)

        for name, tensor in self.bindings.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream))
        self.stream.synchronize()
        return {name: self.bindings[name].clone() for name in self.output_names}

    def forward(
        self,
        im: Union[torch.Tensor, Any],
        command: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        feed: Dict[str, torch.Tensor] = {self.input_names[0]: self.to_tensor(im)}
        cmd = command if command is not None else kwargs.get("cmd")
        if cmd is not None and len(self.input_names) > 1:
            feed[self.input_names[1]] = self.to_tensor(cmd)
        return self.forward_onnx_dict(feed)

    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 320, 320)) -> None:
        if self.warmup_done:
            return
        dummy = {name: self.bindings[name].clone() for name in self.input_names}
        if not self.model_key and self.input_names:
            dummy[self.input_names[0]] = torch.zeros(
                imgsz, dtype=self.bindings[self.input_names[0]].dtype, device=self.device
            )
        for _ in range(3):
            self.forward_onnx_dict(dummy)
        self.warmup_done = True
        logger.info(f"TensorRT warmup complete. IO Buffers: {self.get_io_info()}")
