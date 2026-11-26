"""ONNX-to-GraphModel weight converters for World Models."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import CONVERTER_REGISTRY, BaseOnnxConverter, BaseZooModel
from neuro_pilot.nn.zoo.blocks.common import GroupNormAct
from neuro_pilot.nn.zoo.blocks.vq import VQAttnBlock, VQResBlock
from neuro_pilot.nn.zoo.converters.utils import to_tensor

_SUFFIX_RE = re.compile(r"_(\d+)$")


def _build_vq_enc_map() -> Dict[str, str]:
    mapping = {"encoder.conv_in": "layers.0.conv"}
    idx = 1
    for stage in range(5):
        if stage < 4:
            for blk in range(2):
                mapping[f"encoder.down.{stage}.block.{blk}"] = f"layers.{idx}.{blk}"
            mapping[f"encoder.down.{stage}.downsample"] = f"layers.{idx + 1}"
            idx += 2
        else:
            for blk in range(2):
                mapping[f"encoder.down.{stage}.block.{blk}"] = f"layers.{idx}"
                mapping[f"encoder.down.{stage}.attn.{blk}"] = f"layers.{idx + 1}"
                idx += 2
    mapping["encoder.mid.block_1"] = f"layers.{idx}"
    mapping["encoder.mid.attn_1"] = f"layers.{idx + 1}"
    mapping["encoder.mid.block_2"] = f"layers.{idx + 2}"
    mapping["encoder.conv_out"] = f"layers.{idx + 4}.conv"
    mapping["quant_conv"] = f"layers.{idx + 5}.conv"
    mapping["quantize"] = f"layers.{idx + 6}"
    return mapping


def _build_vq_dec_map() -> Dict[str, str]:
    mapping = {
        "quantize": "layers.0",
        "post_quant_conv": "layers.1.conv",
        "decoder.conv_in": "layers.2.conv",
        "decoder.mid.block_1": "layers.3",
        "decoder.mid.attn_1": "layers.4",
        "decoder.mid.block_2": "layers.5",
    }
    idx = 6
    for stage in range(4, -1, -1):
        if stage == 4:
            for blk in range(3):
                mapping[f"decoder.up.{stage}.block.{blk}"] = f"layers.{idx}"
                mapping[f"decoder.up.{stage}.attn.{blk}"] = f"layers.{idx + 1}"
                idx += 2
            mapping[f"decoder.up.{stage}.upsample"] = f"layers.{idx}"
            idx += 1
        else:
            for blk in range(3):
                mapping[f"decoder.up.{stage}.block.{blk}"] = f"layers.{idx}.{blk}"
            idx += 1
            if stage > 0:
                mapping[f"decoder.up.{stage}.upsample"] = f"layers.{idx}"
                idx += 1
    mapping["decoder.conv_out"] = f"layers.{idx + 1}.conv"
    return mapping


def _sorted_prefix(
    onnx_tensors: Dict[str, np.ndarray], prefix: str
) -> List[Tuple[str, np.ndarray]]:
    matched: List[Tuple[int, str, np.ndarray]] = []
    for name, arr in onnx_tensors.items():
        if name.startswith(prefix):
            match = _SUFFIX_RE.search(name)
            if match is not None:
                matched.append((int(match.group(1)), name, arr))
    matched.sort(key=lambda item: item[0])
    return [(name, arr) for _, name, arr in matched]


def _pop_tensor(
    queue: List[Tuple[str, np.ndarray]],
    shape: Tuple[int, ...],
    label: str,
    transpose: bool = False,
) -> torch.Tensor:
    if not queue:
        raise KeyError(f"ONNX queue exhausted while extracting '{label}'")
    name, arr = queue.pop(0)
    if tuple(arr.shape) != shape:
        raise ValueError(f"Shape mismatch for '{label}' from '{name}': {arr.shape} vs {shape}")
    out = arr.T if transpose else arr
    return to_tensor(out)


def _pop_linear(
    sd: Dict[str, torch.Tensor],
    queue: List[Tuple[str, np.ndarray]],
    key: str,
    rows: int,
    cols: int,
) -> None:
    sd[key] = _pop_tensor(queue, (rows, cols), key, transpose=True)


def _map_ld_named_key(rel: str) -> str:
    if rel.startswith("encoder.patch_embed."):
        return "layers.0." + rel[len("encoder.patch_embed.") :]
    if rel.startswith("encoder.blocks."):
        parts = rel.split(".")
        idx, rest = parts[2], ".".join(parts[3:])
        if rest.startswith("attn."):
            return f"layers.1.{idx}.{rest[5:]}"
        if rest.startswith("norm2."):
            return f"layers.1.{idx}.mlp.norm.{rest[6:]}"
        return f"layers.1.{idx}.{rest}"
    if rel.startswith("encoder.norm."):
        return "layers.2." + rel[len("encoder.") :]
    if rel.startswith("image_fc.") or rel.startswith("status_encoding."):
        return "layers.3." + rel
    if rel.startswith("transformer."):
        return "layers.4." + rel
    if rel.startswith("trajectory_head._mlp.0."):
        return "layers.5.mlp.fc1." + rel.split(".")[-1]
    if rel.startswith("trajectory_head._mlp.2."):
        return "layers.5.mlp.fc2." + rel.split(".")[-1]
    raise KeyError(f"Unmapped LatentDrive ONNX tensor: m.{rel}")


def _extract_ld_transformer(
    sd: Dict[str, torch.Tensor],
    matmuls: List[Tuple[str, np.ndarray]],
    adds: List[Tuple[str, np.ndarray]],
    num_enc: int,
    num_dec: int,
    d_model: int,
    ff_dim: int,
) -> None:
    for idx in range(num_enc):
        pfx = f"layers.4.transformer.encoder.layers.{idx}"
        _pop_linear(sd, matmuls, f"{pfx}.self_attn.in_proj_weight", d_model, 3 * d_model)
        _pop_linear(sd, matmuls, f"{pfx}.linear1.weight", d_model, ff_dim)
        _pop_linear(sd, matmuls, f"{pfx}.linear2.weight", ff_dim, d_model)

    for idx in range(num_dec):
        pfx = f"layers.4.transformer.decoder.layers.{idx}"
        _pop_linear(sd, matmuls, f"{pfx}.self_attn.in_proj_weight", d_model, 3 * d_model)
        q_w = _pop_tensor(matmuls, (d_model, d_model), f"{pfx}.multihead_attn.q_proj", True)
        kv_w = _pop_tensor(matmuls, (d_model, 2 * d_model), f"{pfx}.multihead_attn.kv_proj", True)
        q_b = _pop_tensor(adds, (d_model,), f"{pfx}.multihead_attn.q_bias")
        kv_b = _pop_tensor(adds, (2 * d_model,), f"{pfx}.multihead_attn.kv_bias")
        sd[f"{pfx}.multihead_attn.in_proj_weight"] = torch.cat((q_w, kv_w), dim=0)
        sd[f"{pfx}.multihead_attn.in_proj_bias"] = torch.cat((q_b, kv_b), dim=0)
        _pop_linear(sd, matmuls, f"{pfx}.linear1.weight", d_model, ff_dim)
        _pop_linear(sd, matmuls, f"{pfx}.linear2.weight", ff_dim, d_model)


@CONVERTER_REGISTRY.register("jepa")
@CONVERTER_REGISTRY.register("latent_drive_vjepa")
class JepaConverter(BaseOnnxConverter):
    """Convert V-JEPA planner ONNX weights into GraphModel state_dict."""

    model_key = "jepa"
    yaml_name = "jepa"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        for name, arr in onnx_tensors.items():
            if name.startswith("m."):
                sd[_map_ld_named_key(name[2:])] = to_tensor(arr)

        matmuls = _sorted_prefix(onnx_tensors, "onnx::MatMul_")
        adds = _sorted_prefix(onnx_tensors, "onnx::Add_")
        expands = _sorted_prefix(onnx_tensors, "onnx::Expand_")

        vit_blocks = model.layers[1]
        embed_dim = vit_blocks[0].norm1.normalized_shape[0]
        mlp_dim = vit_blocks[0].mlp.fc1.out_features
        d_model = model.layers[3].image_fc.out_features
        ff_dim = model.layers[5].mlp.fc1.out_features

        for idx in range(len(vit_blocks)):
            pfx = f"layers.1.{idx}"
            _pop_linear(sd, matmuls, f"{pfx}.qkv.weight", embed_dim, 3 * embed_dim)
            _pop_linear(sd, matmuls, f"{pfx}.proj.weight", embed_dim, embed_dim)
            _pop_linear(sd, matmuls, f"{pfx}.mlp.fc1.weight", embed_dim, mlp_dim)
            _pop_linear(sd, matmuls, f"{pfx}.mlp.fc2.weight", mlp_dim, embed_dim)

        _pop_linear(sd, matmuls, "layers.3.image_fc.weight", embed_dim, d_model)
        pos_shape = tuple(model.layers[3].encoder_pos_emb.shape)
        query_shape = tuple(model.layers[4].decoder_query.shape)
        sd["layers.3.encoder_pos_emb"] = _pop_tensor(adds, pos_shape, "encoder_pos_emb")
        sd["layers.4.decoder_query"] = _pop_tensor(expands, query_shape, "decoder_query")

        num_enc = len(model.layers[4].transformer.encoder.layers)
        num_dec = len(model.layers[4].transformer.decoder.layers)
        _extract_ld_transformer(sd, matmuls, adds, num_enc, num_dec, d_model, ff_dim)
        _pop_linear(sd, matmuls, "layers.5.mlp.fc1.weight", d_model, ff_dim)
        _pop_linear(sd, matmuls, "layers.5.mlp.fc2.weight", ff_dim, 3)
        return sd


def _assign_gn(
    sd: Dict[str, torch.Tensor],
    muls: List[Tuple[str, np.ndarray]],
    adds: List[Tuple[str, np.ndarray]],
    prefix: str,
    norm: nn.GroupNorm,
) -> None:
    ch = norm.num_channels
    sd[f"{prefix}.weight"] = _pop_tensor(muls, (ch, 1, 1), f"{prefix}.weight").reshape(ch)
    sd[f"{prefix}.bias"] = _pop_tensor(adds, (ch, 1, 1), f"{prefix}.bias").reshape(ch)


def _extract_vq_state_dict(
    onnx_tensors: Dict[str, Any], model: BaseZooModel, prefix_map: Dict[str, str]
) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    ordered = sorted(prefix_map.items(), key=lambda item: len(item[0]), reverse=True)
    for name, arr in onnx_tensors.items():
        if not name.startswith("model."):
            continue
        rel = name[6:]
        mapped = None
        for src, dst in ordered:
            if rel.startswith(src + "."):
                mapped = dst + rel[len(src) :]
                break
        if mapped is None:
            raise KeyError(f"Unmapped CommaVQ ONNX tensor: {name}")
        sd[mapped] = to_tensor(arr)

    muls = _sorted_prefix(onnx_tensors, "onnx::Mul_")
    adds = _sorted_prefix(onnx_tensors, "onnx::Add_")
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, nn.Sequential):
            for sub_idx, sub in enumerate(layer):
                if isinstance(sub, VQResBlock):
                    _assign_gn(sd, muls, adds, f"layers.{idx}.{sub_idx}.norm1.norm", sub.norm1.norm)
                    _assign_gn(sd, muls, adds, f"layers.{idx}.{sub_idx}.norm2.norm", sub.norm2.norm)
        elif isinstance(layer, VQResBlock):
            _assign_gn(sd, muls, adds, f"layers.{idx}.norm1.norm", layer.norm1.norm)
            _assign_gn(sd, muls, adds, f"layers.{idx}.norm2.norm", layer.norm2.norm)
        elif isinstance(layer, VQAttnBlock):
            _assign_gn(sd, muls, adds, f"layers.{idx}.norm.norm", layer.norm.norm)
        elif isinstance(layer, GroupNormAct):
            _assign_gn(sd, muls, adds, f"layers.{idx}.norm", layer.norm)
    return sd


@CONVERTER_REGISTRY.register("vq_enc")
@CONVERTER_REGISTRY.register("comma_vq_encoder")
class VqEncConverter(BaseOnnxConverter):
    """Convert VQGAN encoder ONNX weights into GraphModel state_dict."""

    model_key = "vq_enc"
    yaml_name = "vq_enc"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        return _extract_vq_state_dict(onnx_tensors, model, _build_vq_enc_map())


@CONVERTER_REGISTRY.register("vq_dec")
@CONVERTER_REGISTRY.register("comma_vq_decoder")
class VqDecConverter(BaseOnnxConverter):
    """Convert VQGAN decoder ONNX weights into GraphModel state_dict."""

    model_key = "vq_dec"
    yaml_name = "vq_dec"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        return _extract_vq_state_dict(onnx_tensors, model, _build_vq_dec_map())
