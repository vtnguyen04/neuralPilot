"""ONNX-to-GraphModel tensor mapping helpers for openpilot converters."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from neuro_pilot.nn.zoo.converters.utils import to_tensor

_MLP_PAIRS = (
    ("norm.weight", "norm.weight"),
    ("norm.bias", "norm.bias"),
    ("c_fc.weight", "fc1.weight"),
    ("c_proj.weight", "fc2.weight"),
)
_POINT_HEADS = (
    "lane_lines",
    "lane_lines_prob",
    "road_edges",
    "meta",
    "desire_pred",
    "road_transform",
    "wide_from_device_euler",
    "pose",
)


def copy_wb(ox: Dict[str, Any], sd: Dict[str, torch.Tensor], src: str, dst: str) -> None:
    """Copy a standard `.weight` and `.bias` tensor pair."""
    sd[f"{dst}.weight"] = to_tensor(ox[f"{src}.weight"])
    sd[f"{dst}.bias"] = to_tensor(ox[f"{src}.bias"])


def copy_mlp(ox: Dict[str, Any], sd: Dict[str, torch.Tensor], src: str, dst: str) -> None:
    """Copy a Pre-LN bias-free residual MLP from ONNX keys to `MLP` keys."""
    for src_sfx, dst_sfx in _MLP_PAIRS:
        sd[f"{dst}.{dst_sfx}"] = to_tensor(ox[f"{src}.{src_sfx}"])


class AnonStream:
    """Ordered consumer for anonymous float ONNX initializers."""

    def __init__(self, onnx_tensors: Dict[str, Any]) -> None:
        self._items: List[Tuple[str, np.ndarray]] = [
            (key, val)
            for key, val in onnx_tensors.items()
            if not key.startswith("model.") and val.dtype.kind == "f" and val.size > 1
        ]
        self._idx = 0

    def pop(self, expected_shape: Tuple[int, ...], transpose: bool = False) -> torch.Tensor:
        if self._idx >= len(self._items):
            raise IndexError(f"Anonymous stream exhausted at index {self._idx}")
        name, arr = self._items[self._idx]
        self._idx += 1
        tensor = to_tensor(arr, transpose=transpose)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch on '{name}': got {tuple(tensor.shape)}, expected {expected_shape}"
            )
        return tensor

    def assert_exhausted(self) -> None:
        if self._idx != len(self._items):
            remaining = [key for key, _ in self._items[self._idx :]]
            raise ValueError(f"Unconsumed anonymous tensors ({len(remaining)}): {remaining[:5]}")


def map_queue_and_pack(
    ox: Dict[str, Any], sd: Dict[str, torch.Tensor], queue_idx: int, pack_idx: int
) -> None:
    """Map camera normalization buffers and output padding buffer."""
    sd[f"layers.{queue_idx}.vision_mean"] = to_tensor(ox["model.vision._mean"])
    sd[f"layers.{queue_idx}.vision_std"] = to_tensor(ox["model.vision._std"])
    sd[f"layers.{pack_idx}.pad"] = to_tensor(ox["model.pad"])


def map_fastvit_layers(
    ox: Dict[str, Any], sd: Dict[str, torch.Tensor], start_idx: int = 1
) -> None:
    """Map FastViT-T12 ONNX weights into GraphModel backbone layers."""
    prefix = "model.vision.encoder"
    for idx in range(3):
        copy_wb(ox, sd, f"{prefix}.stem.{idx}.reparam_conv", f"layers.{start_idx}.stem_{idx}")

    layer_idx = start_idx + 1
    for stage_idx, depth in enumerate((2, 2, 6, 2)):
        if stage_idx > 0:
            ds_src = f"{prefix}.stages.{stage_idx}.downsample.proj"
            copy_wb(ox, sd, f"{ds_src}.0.reparam_conv", f"layers.{layer_idx}.proj_dw")
            copy_wb(ox, sd, f"{ds_src}.1.reparam_conv", f"layers.{layer_idx}.proj_pw")
            layer_idx += 1

        for block_idx in range(depth):
            src = f"{prefix}.stages.{stage_idx}.blocks.{block_idx}"
            dst = f"layers.{layer_idx}.{block_idx}"
            copy_wb(ox, sd, f"{src}.token_mixer.reparam_conv", f"{dst}.token_mixer")
            sd[f"{dst}.mlp_dw.weight"] = to_tensor(ox[f"{src}.mlp.conv.conv.weight"])
            sd[f"{dst}.mlp_dw.bias"] = to_tensor(ox[f"{src}.mlp.conv.conv.weight_bias"])
            copy_wb(ox, sd, f"{src}.mlp.fc1", f"{dst}.mlp_fc1")
            copy_wb(ox, sd, f"{src}.mlp.fc2", f"{dst}.mlp_fc2")
            sd[f"{dst}.layer_scale"] = to_tensor(ox[f"{src}.layer_scale.gamma"])
        layer_idx += 1

    copy_wb(ox, sd, f"{prefix}.final_conv.reparam_conv", f"layers.{layer_idx}.final_dw")
    copy_wb(ox, sd, f"{prefix}.final_conv.se.fc1", f"layers.{layer_idx}.se_fc1")
    copy_wb(ox, sd, f"{prefix}.final_conv.se.fc2", f"layers.{layer_idx}.se_fc2")
    copy_wb(ox, sd, f"{prefix}.head.fc", f"layers.{layer_idx}.head_fc")


def map_convnext_layers(
    ox: Dict[str, Any],
    stream: AnonStream,
    sd: Dict[str, torch.Tensor],
    src_prefix: str,
    start_idx: int,
    dims: Tuple[int, int, int, int],
    depths: Tuple[int, int, int, int],
    has_head_fc: bool,
) -> None:
    """Map ConvNeXt stem, stages, downsamples, and head into GraphModel layers."""
    copy_wb(ox, sd, f"{src_prefix}.stem.0", f"layers.{start_idx}.conv")
    copy_wb(ox, sd, f"{src_prefix}.stem.1", f"layers.{start_idx}.norm")

    layer_idx = start_idx + 1
    for stage_idx, (dim, depth) in enumerate(zip(dims, depths)):
        if stage_idx > 0:
            ds_src = f"{src_prefix}.stages.{stage_idx}.downsample"
            copy_wb(ox, sd, f"{ds_src}.0", f"layers.{layer_idx}.norm")
            copy_wb(ox, sd, f"{ds_src}.1", f"layers.{layer_idx}.conv")
            layer_idx += 1

        for block_idx in range(depth):
            src = f"{src_prefix}.stages.{stage_idx}.blocks.{block_idx}"
            dst = f"layers.{layer_idx}.{block_idx}"
            copy_wb(ox, sd, f"{src}.conv_dw", f"{dst}.conv_dw")
            copy_wb(ox, sd, f"{src}.norm", f"{dst}.mlp.norm")
            sd[f"{dst}.mlp.fc1.weight"] = stream.pop((dim * 4, dim), transpose=True)
            sd[f"{dst}.mlp.fc1.bias"] = to_tensor(ox[f"{src}.mlp.fc1.bias"])
            sd[f"{dst}.mlp.fc2.weight"] = stream.pop((dim, dim * 4), transpose=True)
            sd[f"{dst}.mlp.fc2.bias"] = to_tensor(ox[f"{src}.mlp.fc2.bias"])
            sd[f"{dst}.gamma"] = stream.pop((1, dim, 1, 1), transpose=False)
        layer_idx += 1

    if has_head_fc:
        copy_wb(ox, sd, f"{src_prefix}.head.norm", f"layers.{layer_idx}.norm")
        copy_wb(ox, sd, f"{src_prefix}.head.fc", f"layers.{layer_idx}.head_fc")
    else:
        copy_wb(ox, sd, f"{src_prefix}.head.norm", f"layers.{layer_idx}")


def map_point_policy_layer(
    ox: Dict[str, Any], sd: Dict[str, torch.Tensor], layer_idx: int
) -> None:
    """Map PointPolicy residual MLPs, linear heads, and pose scale."""
    prefix = "model.point_policy"
    dst = f"layers.{layer_idx}"
    copy_mlp(ox, sd, f"{prefix}.summarizer.mlp1", f"{dst}.mlp1")
    copy_mlp(ox, sd, f"{prefix}.summarizer.mlp2", f"{dst}.mlp2")
    for head in _POINT_HEADS:
        copy_wb(ox, sd, f"{prefix}.hydra.final_layer.{head}", f"{dst}.heads.{head}")
    copy_mlp(ox, sd, f"{prefix}.hydra.head_mlp.pose", f"{dst}.pose_mlp")
    sd[f"{dst}.pose_scale.scale"] = to_tensor(ox[f"{prefix}.hydra.scale_layer.pose.scale"])


def map_temporal_layers(
    ox: Dict[str, Any],
    stream: AnonStream,
    sd: Dict[str, torch.Tensor],
    src_prefix: str,
    embed_idx: int,
    block_idx: int,
    num_layers: int,
    spatial_tokens: int,
    desire_in_dim: int,
) -> None:
    """Map TemporalEmbed and QKNormBlock stack into GraphModel layers."""
    embed = f"layers.{embed_idx}"
    for mlp_name in ("mlp1", "mlp2"):
        copy_wb(ox, sd, f"{src_prefix}.{mlp_name}.norm", f"{embed}.{mlp_name}.norm")
        sd[f"{embed}.{mlp_name}.fc1.weight"] = stream.pop((1024, 512), transpose=True)
        sd[f"{embed}.{mlp_name}.fc2.weight"] = stream.pop((512, 1024), transpose=True)

    sd[f"{embed}.desire_encoder.fc1.weight"] = stream.pop((512, desire_in_dim), transpose=True)
    desire_bias = f"{src_prefix}.desire_encoder.net.0.bias"
    sd[f"{embed}.desire_encoder.fc1.bias"] = to_tensor(ox[desire_bias])
    sd[f"{embed}.desire_encoder.fc2.weight"] = stream.pop((512, 512), transpose=True)
    for enc in ("traffic_encoder", "action_t_encoder"):
        copy_wb(ox, sd, f"{src_prefix}.{enc}.net.0", f"{embed}.{enc}.fc1")
        sd[f"{embed}.{enc}.fc2.weight"] = to_tensor(ox[f"{src_prefix}.{enc}.net.2.weight"])

    if spatial_tokens > 1:
        sd[f"{embed}.temporal_pos_embedding"] = stream.pop((9, 1, 512), transpose=False)
        sd[f"{embed}.spatial_pos_embedding.weight"] = to_tensor(
            ox[f"{src_prefix}.spatial_pos_embedding.weight"]
        )
    else:
        sd[f"{embed}.pos_embedding"] = stream.pop((1, 9, 512), transpose=False)

    for layer_num in range(num_layers):
        src_l = f"{src_prefix}.transformer.layers.{layer_num}"
        dst_l = f"layers.{block_idx}.{layer_num}" if num_layers > 1 else f"layers.{block_idx}"
        copy_wb(ox, sd, f"{src_l}.attention.norm", f"{dst_l}.norm1")
        copy_wb(ox, sd, f"{src_l}.attention.q_norm", f"{dst_l}.q_norm")
        copy_wb(ox, sd, f"{src_l}.attention.k_norm", f"{dst_l}.k_norm")
        sd[f"{dst_l}.qkv.weight"] = stream.pop((1536, 512), transpose=True)
        sd[f"{dst_l}.qkv.bias"] = to_tensor(ox[f"{src_l}.attention.c_attn.bias"])
        sd[f"{dst_l}.proj.weight"] = stream.pop((512, 512), transpose=True)
        sd[f"{dst_l}.proj.bias"] = to_tensor(ox[f"{src_l}.attention.c_proj.bias"])
        copy_wb(ox, sd, f"{src_l}.mlp.norm", f"{dst_l}.mlp.norm")
        sd[f"{dst_l}.mlp.fc1.weight"] = stream.pop((1024, 512), transpose=True)
        sd[f"{dst_l}.mlp.fc1.bias"] = to_tensor(ox[f"{src_l}.mlp.c_fc.bias"])
        sd[f"{dst_l}.mlp.fc2.weight"] = stream.pop((512, 1024), transpose=True)
        sd[f"{dst_l}.mlp.fc2.bias"] = to_tensor(ox[f"{src_l}.mlp.c_proj.bias"])


def map_hydra_layer(
    ox: Dict[str, Any],
    sd: Dict[str, torch.Tensor],
    src_prefix: str,
    layer_idx: int,
    heads: Sequence[Tuple[str, bool]],
) -> None:
    """Map TemporalHydra head MLPs, final projections, and scale layers."""
    dst = f"layers.{layer_idx}"
    for name, has_scale in heads:
        copy_mlp(ox, sd, f"{src_prefix}.head_mlp.{name}", f"{dst}.head_mlps.{name}")
        copy_wb(ox, sd, f"{src_prefix}.final_layer.{name}", f"{dst}.final_layers.{name}")
        if has_scale:
            sd[f"{dst}.scale_layers.{name}.scale"] = to_tensor(
                ox[f"{src_prefix}.scale_layer.{name}.scale"]
            )


def map_residual_layer(ox: Dict[str, Any], sd: Dict[str, torch.Tensor], layer_idx: int) -> None:
    """Map ResidualActionHead action encoder, MLP, and final projection."""
    prefix = "model.on_policy"
    dst = f"layers.{layer_idx}"
    copy_wb(ox, sd, f"{prefix}.action_encoder.net.0", f"{dst}.action_encoder.fc1")
    sd[f"{dst}.action_encoder.fc2.weight"] = to_tensor(ox[f"{prefix}.action_encoder.net.2.weight"])
    rh = f"{prefix}.residual_hydra"
    copy_mlp(ox, sd, f"{rh}.head_mlp.residual_action", f"{dst}.head_mlp")
    copy_wb(ox, sd, f"{rh}.final_layer.residual_action", f"{dst}.final_layer")
