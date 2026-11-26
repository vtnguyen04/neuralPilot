"""ONNX and TorchScript export helpers with embedded metadata for GraphModel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class _PositionalExportWrapper(nn.Module):
    """Adapter exposing positional tensor inputs and tuple outputs for tracing/export."""

    def __init__(self, model: Any, in_names: List[str], out_names: List[str]) -> None:
        super().__init__()
        self.model = model
        self.in_names = in_names
        self.out_names = out_names

    def forward(self, *args: torch.Tensor) -> Any:
        out = self.model.forward_onnx_dict(dict(zip(self.in_names, args)))
        return tuple(out[k] for k in self.out_names)


def _prepare_export_feed(
    model: Any, sample_inputs: Optional[Dict[str, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    sample = sample_inputs if sample_inputs is not None else model.make_sample_inputs(seed=42)
    feed = {k: v.to(device) for k, v in sample.items()}
    in_names = list(feed.keys())
    with torch.no_grad():
        out_names = list(model.forward_onnx_dict(feed).keys())
    return feed, in_names, out_names


def _embed_onnx_meta(model: Any, onnx_path: Path, out_names: List[str]) -> None:
    meta = {
        "model_key": model.model_key,
        "family": model.family,
        "input_specs": model.input_specs,
        "output_names": out_names,
    }
    if onnx_path.stat().st_size < 1_800_000_000:
        import onnx

        proto = onnx.load(str(onnx_path))
        for key, val in meta.items():
            prop = proto.metadata_props.add()
            prop.key = key
            prop.value = val if isinstance(val, str) else json.dumps(val)
        onnx.save(proto, str(onnx_path))
    else:
        from neuro_pilot.engine.backend.base import save_sidecar_metadata

        save_sidecar_metadata(onnx_path, meta)


def export_graph_onnx(
    model: Any,
    onnx_path: Union[str, Path],
    opset: int = 17,
    sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    do_constant_folding: bool = True,
) -> Path:
    """Export a GraphModel to self-describing ONNX preserving exact I/O names."""
    out_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    feed, in_names, out_names = _prepare_export_feed(model, sample_inputs)
    wrapper = _PositionalExportWrapper(model, in_names, out_names).eval()
    torch.onnx.export(
        wrapper,
        tuple(feed[k] for k in in_names),
        str(out_path),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=do_constant_folding,
    )
    _embed_onnx_meta(model, out_path, out_names)
    if was_training:
        model.train()
    return out_path


def export_graph_torchscript(
    model: Any,
    ts_path: Union[str, Path],
    sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    """Trace and save a GraphModel to a self-describing `.torchscript` archive."""
    out_path = Path(ts_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    feed, in_names, out_names = _prepare_export_feed(model, sample_inputs)
    with torch.no_grad():
        wrapper = _PositionalExportWrapper(model, in_names, out_names).eval()
        traced = torch.jit.trace(wrapper, tuple(feed[k] for k in in_names), strict=False)
    meta = json.dumps(
        {
            "model_key": model.model_key,
            "family": model.family,
            "input_names": in_names,
            "input_specs": model.input_specs,
            "output_names": out_names,
        }
    )
    torch.jit.save(traced, str(out_path), _extra_files={"neuropilot_meta.json": meta})
    if was_training:
        model.train()
    return out_path
