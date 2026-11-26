"""Deterministic safetensors-to-GraphModel converter for EoMT-Large (`seg_l.yaml`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from neuro_pilot.nn.zoo.base import CONVERTER_REGISTRY, BaseOnnxConverter, BaseZooModel

_BLOCK_MAP: Tuple[Tuple[str, str], ...] = (
    ("norm1.weight", "norm1.weight"),
    ("norm1.bias", "norm1.bias"),
    ("attention.q_proj.weight", "q_proj.weight"),
    ("attention.q_proj.bias", "q_proj.bias"),
    ("attention.k_proj.weight", "k_proj.weight"),
    ("attention.k_proj.bias", "k_proj.bias"),
    ("attention.v_proj.weight", "v_proj.weight"),
    ("attention.v_proj.bias", "v_proj.bias"),
    ("attention.out_proj.weight", "out_proj.weight"),
    ("attention.out_proj.bias", "out_proj.bias"),
    ("layer_scale1.lambda1", "ls1"),
    ("norm2.weight", "norm2.weight"),
    ("norm2.bias", "norm2.bias"),
    ("mlp.fc1.weight", "fc1.weight"),
    ("mlp.fc1.bias", "fc1.bias"),
    ("mlp.fc2.weight", "fc2.weight"),
    ("mlp.fc2.bias", "fc2.bias"),
    ("layer_scale2.lambda1", "ls2"),
)


@CONVERTER_REGISTRY.register("seg_l")
@CONVERTER_REGISTRY.register("eomt_large")
class EomtLargeConverter(BaseOnnxConverter):
    """Map HuggingFace EoMT-Large safetensors into exact `seg_l.yaml` GraphModel tensors."""

    model_key: str = "seg_l"
    yaml_name: str = "seg_l"

    def extract_state_dict(
        self, raw_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        ref = model.state_dict()
        sd: Dict[str, torch.Tensor] = {
            "layers.0.mean": ref["layers.0.mean"].clone(),
            "layers.0.std": ref["layers.0.std"].clone(),
            "layers.0.proj.weight": raw_tensors["embeddings.patch_embeddings.projection.weight"],
            "layers.0.proj.bias": raw_tensors["embeddings.patch_embeddings.projection.bias"],
            "layers.0.cls_token": raw_tensors["embeddings.cls_token"],
            "layers.0.register_tokens": raw_tensors["embeddings.register_tokens"],
            "layers.0.pos_embed": raw_tensors["embeddings.position_embeddings.weight"],
            "layers.2.query": raw_tensors["query.weight"],
            "layers.4.norm.weight": raw_tensors["layernorm.weight"],
            "layers.4.norm.bias": raw_tensors["layernorm.bias"],
            "layers.4.class_predictor.weight": raw_tensors["class_predictor.weight"],
            "layers.4.class_predictor.bias": raw_tensors["class_predictor.bias"],
            "layers.4.fc1.weight": raw_tensors["mask_head.fc1.weight"],
            "layers.4.fc1.bias": raw_tensors["mask_head.fc1.bias"],
            "layers.4.fc2.weight": raw_tensors["mask_head.fc2.weight"],
            "layers.4.fc2.bias": raw_tensors["mask_head.fc2.bias"],
            "layers.4.fc3.weight": raw_tensors["mask_head.fc3.weight"],
            "layers.4.fc3.bias": raw_tensors["mask_head.fc3.bias"],
        }
        for i in range(20):
            for src_s, dst_s in _BLOCK_MAP:
                sd[f"layers.1.{i}.{dst_s}"] = raw_tensors[f"layers.{i}.{src_s}"]
        for i in range(4):
            for src_s, dst_s in _BLOCK_MAP:
                sd[f"layers.3.{i}.{dst_s}"] = raw_tensors[f"layers.{20 + i}.{src_s}"]
        for bi in (0, 1):
            src = f"upscale_block.block.{bi}"
            dst = f"layers.4.upscale.{bi}"
            sd[f"{dst}.conv1.weight"] = raw_tensors[f"{src}.conv1.weight"]
            sd[f"{dst}.conv1.bias"] = raw_tensors[f"{src}.conv1.bias"]
            sd[f"{dst}.conv2.weight"] = raw_tensors[f"{src}.conv2.weight"]
            sd[f"{dst}.norm.weight"] = raw_tensors[f"{src}.layernorm2d.weight"]
            sd[f"{dst}.norm.bias"] = raw_tensors[f"{src}.layernorm2d.bias"]
        return sd


def convert_safetensors_to_pt(
    model_key: str,
    safetensors_path: str | Path,
    pt_output_path: str | Path,
    half: bool = True,
) -> Tuple[BaseZooModel, List[str], List[str], int]:
    """Convert a HuggingFace `.safetensors` checkpoint into a NeuroPilot `.pt` checkpoint."""
    from safetensors.torch import load_file

    converter: BaseOnnxConverter = CONVERTER_REGISTRY.build(model_key)
    model = converter.build_model()
    raw = load_file(str(safetensors_path))
    state_dict = converter.extract_state_dict(raw, model)
    incompatible = model.load_state_dict(state_dict, strict=True)

    saved_sd = {
        k: (v.half() if half and v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    out_path = Path(pt_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_key": model_key,
            "family": model.family,
            "config": {},
            "state_dict": saved_sd,
        },
        str(out_path),
    )
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    return model, missing, unexpected, len(state_dict)
