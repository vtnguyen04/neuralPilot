"""Generic layer-by-layer YAML computation graph executor (GraphModel)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import yaml

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY, BaseZooModel

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _subst(val: Any, scale_vars: Dict[str, Any]) -> Any:
    """Recursively substitute scale/config variable names inside YAML args or shapes."""
    if isinstance(val, str) and val in scale_vars:
        return scale_vars[val]
    if isinstance(val, list):
        return [_subst(v, scale_vars) for v in val]
    if isinstance(val, dict):
        return {k: _subst(v, scale_vars) for k, v in val.items()}
    return val


def _resolve_block(name: str) -> Any:
    if name in BLOCK_REGISTRY:
        return BLOCK_REGISTRY.get(name)
    if name.startswith("nn.") and hasattr(nn, name[3:]):
        return getattr(nn, name[3:])
    raise KeyError(f"Unknown block '{name}'. Registered: {BLOCK_REGISTRY.keys()}")


def is_zoo_yaml(cfg: Any) -> bool:
    """Return True when a parsed YAML dict defines a Zoo layer graph."""
    if not isinstance(cfg, dict):
        return False
    rows = list(cfg.get("backbone") or []) + list(cfg.get("head") or [])
    if not rows:
        return False
    if not BLOCK_REGISTRY.keys():
        from neuro_pilot.nn.zoo.blocks import register_all_blocks

        register_all_blocks()
    return any(isinstance(r, list) and len(r) >= 3 and r[2] in BLOCK_REGISTRY for r in rows)


def _make_tensor(spec: Any, gen: torch.Generator) -> torch.Tensor:
    if isinstance(spec, list):
        return torch.randn(tuple(int(d) for d in spec), generator=gen, dtype=torch.float32)
    shape = tuple(int(d) for d in spec["shape"])
    dtype = _DTYPE_MAP[str(spec.get("dtype", "float32"))]
    dist = str(spec.get("dist", "randn"))
    if dist == "randint":
        lo, hi = int(spec.get("low", 0)), int(spec.get("high", 256))
        return torch.randint(lo, hi, shape, generator=gen, dtype=dtype)
    if dist == "rand":
        lo, hi = float(spec.get("low", 0.0)), float(spec.get("high", 1.0))
        return torch.rand(shape, generator=gen, dtype=dtype) * (hi - lo) + lo
    if dist == "const":
        val = spec.get("value", 0.0)
        if isinstance(val, list):
            return torch.tensor(val, dtype=dtype).reshape(shape)
        return torch.full(shape, val, dtype=dtype)
    if dist == "arange":
        return torch.arange(int(shape[-1]), dtype=dtype).reshape(shape)
    scale = float(spec.get("scale", 1.0))
    return torch.randn(shape, generator=gen, dtype=dtype) * scale


class GraphModel(BaseZooModel):
    """Unified PyTorch model executed directly from a layer-by-layer YAML graph.

    Every layer row follows `[from, repeats, BlockName, args]`:
    - `from`: `-1` (prev layer), `int` (layer index), `"input_name"`, `"idx.key"`,
      or a list combining any of these.
    - `repeats`: integer count or scale variable name (`> 1` builds `nn.Sequential`).
    - `BlockName`: registered atomic block in `BLOCK_REGISTRY` (or `nn.*`).
    - `args`: positional list (or dict of kwargs when last item is a dict? kept simple: list).
    """

    def __init__(
        self,
        cfg: Union[Dict[str, Any], str, Path],
        scale: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        if isinstance(cfg, (str, Path)):
            raw = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        else:
            raw = deepcopy(cfg)

        self.cfg: Dict[str, Any] = raw
        self.model_key = str(raw.get("name", ""))
        self.family = str(raw.get("family", ""))

        scale_vars: Dict[str, Any] = {}
        if isinstance(raw.get("scales"), dict):
            chosen = scale or raw.get("scale") or next(iter(raw["scales"]))
            if chosen not in raw["scales"]:
                raise ValueError(f"Unknown scale '{chosen}'. Available: {list(raw['scales'])}")
            scale_vars.update(raw["scales"][chosen])
        scale_vars.update(overrides)
        self.scale_vars = scale_vars
        self.input_specs: Dict[str, Any] = _subst(raw.get("inputs", {}), scale_vars)
        self.output_map: Dict[str, Any] = dict(raw.get("outputs", {}))

        rows = list(raw.get("backbone") or []) + list(raw.get("head") or [])
        layers: List[nn.Module] = []
        sources: List[Any] = []
        for idx, row in enumerate(rows):
            src_raw, rep_raw, mod_name, args_raw = row
            src = _subst(src_raw, scale_vars)
            rep = int(_subst(rep_raw, scale_vars))
            args = _subst(list(args_raw), scale_vars)
            ctor = _resolve_block(str(mod_name))
            if hasattr(ctor, "build_stage"):
                built = ctor.build_stage(rep, *args)
            elif rep > 1:
                built = nn.Sequential(*(ctor(*args) for _ in range(rep)))
            else:
                built = ctor(*args)
            layers.append(built)
            sources.append(src)

        self.layers = nn.ModuleList(layers)
        self._sources = sources
        self._save_indices = self._compute_saved_indices()

    def _compute_saved_indices(self) -> set:
        saved = set()

        def _mark(ref: Any) -> None:
            if isinstance(ref, list):
                for item in ref:
                    _mark(item)
            elif isinstance(ref, int) and ref >= 0:
                saved.add(ref)
            elif isinstance(ref, str) and "." in ref:
                head = ref.split(".", 1)[0]
                if head.isdigit():
                    saved.add(int(head))

        for src in self._sources:
            _mark(src)
        for ref in self.output_map.values():
            _mark(ref)
        return saved

    @staticmethod
    def _resolve_ref(ref: Any, prev: Any, cache: Dict[int, Any], inputs: Dict[str, Any]) -> Any:
        if isinstance(ref, int):
            return prev if ref == -1 else cache[ref]
        if isinstance(ref, str):
            if "." in ref:
                head, key = ref.split(".", 1)
                obj = cache[int(head)] if head.isdigit() else inputs[head]
                if isinstance(obj, (list, tuple)) and key.isdigit():
                    return obj[int(key)]
                return obj[key]
            return inputs[ref]
        return [GraphModel._resolve_ref(r, prev, cache, inputs) for r in ref]

    def forward_onnx_dict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cache: Dict[int, Any] = {}
        prev: Any = next(iter(inputs.values())) if len(inputs) == 1 else inputs
        for idx, (layer, src) in enumerate(zip(self.layers, self._sources)):
            inp = self._resolve_ref(src, prev, cache, inputs)
            if isinstance(inp, list) and getattr(layer, "unpack_inputs", False):
                out = layer(*inp)
            else:
                out = layer(inp)
            if idx in self._save_indices:
                cache[idx] = out
            prev = out

        if not self.output_map:
            return out if isinstance(out, dict) else {"output": out}
        result: Dict[str, torch.Tensor] = {}
        for out_name, ref in self.output_map.items():
            result[out_name] = self._resolve_ref(ref, prev, cache, inputs)
        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self.forward_onnx_dict(args[0])
        if kwargs and not args:
            return self.forward_onnx_dict(kwargs)
        keys = list(self.input_specs.keys())
        feed = {k: v for k, v in zip(keys, args)}
        feed.update(kwargs)
        out = self.forward_onnx_dict(feed)
        return next(iter(out.values())) if len(out) == 1 else out

    def make_sample_inputs(self, seed: int = 42) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        return {name: _make_tensor(spec, gen) for name, spec in self.input_specs.items()}

    def export_onnx(
        self,
        onnx_path: Union[str, Path],
        opset: int = 17,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        do_constant_folding: bool = True,
    ) -> Path:
        """Export this GraphModel directly to ONNX preserving exact I/O names."""
        from neuro_pilot.nn.zoo.export import export_graph_onnx

        return export_graph_onnx(
            self,
            onnx_path=onnx_path,
            opset=opset,
            sample_inputs=sample_inputs,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
        )

    def export_torchscript(
        self,
        ts_path: Union[str, Path],
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Trace and save this GraphModel to a self-describing `.torchscript` archive."""
        from neuro_pilot.nn.zoo.export import export_graph_torchscript

        return export_graph_torchscript(self, ts_path=ts_path, sample_inputs=sample_inputs)
