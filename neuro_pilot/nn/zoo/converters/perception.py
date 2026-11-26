"""ONNX-to-GraphModel weight converters for all four VisionPilot models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from neuro_pilot.nn.zoo.base import CONVERTER_REGISTRY, BaseOnnxConverter, BaseZooModel
from neuro_pilot.nn.zoo.converters.utils import rewrite_prefix, to_tensor

_MODULE_PREFIXES = ("net.", "backbone.", "fpn.", "head.", "encoder.", "decoder.")

_VP_BACKBONE_STAGES: Tuple[Tuple[str, int], ...] = (
    ("p1.", 0),
    ("p2.0.", 1),
    ("p2.1.", 2),
    ("p3.0.", 3),
    ("p3.1.", 4),
    ("p4.0.", 5),
    ("p4.1.", 6),
    ("p5.0.", 7),
    ("p5.1.", 8),
    ("p5.2.", 9),
    ("p5.3.", 10),
)


def _backbone_rules(root: str, offset: int = 0) -> List[Tuple[str, str]]:
    """Build ONNX-to-GraphModel prefix pairs for the 11-layer VisionPilot backbone."""
    return [(f"{root}.{stage}", f"layers.{offset + idx}.") for stage, idx in _VP_BACKBONE_STAGES]


def _normalize_onnx_name(onnx_name: str) -> str:
    """Convert fused BN bias suffixes (`*.weight_bias`) to standard PyTorch `*.bias`."""
    if onnx_name.endswith(".weight_bias"):
        return onnx_name[: -len(".weight_bias")] + ".bias"
    return onnx_name


def _map_by_rules(
    onnx_tensors: Dict[str, Any],
    model: BaseZooModel,
    mapper: Callable[[str], Optional[str]],
) -> Dict[str, torch.Tensor]:
    """Map normalized ONNX initializers into every parameter and buffer of a GraphModel."""
    remapped: Dict[str, np.ndarray] = {}
    unmapped_module_keys: Set[str] = set()

    for raw_name, array in onnx_tensors.items():
        norm_name = _normalize_onnx_name(raw_name)
        pt_key = mapper(norm_name)
        if pt_key is None:
            if norm_name.startswith(_MODULE_PREFIXES):
                unmapped_module_keys.add(norm_name)
            continue
        if pt_key in remapped:
            raise ValueError(f"Duplicate mapped key '{pt_key}' from ONNX '{raw_name}'")
        remapped[pt_key] = array

    if unmapped_module_keys:
        sample = sorted(unmapped_module_keys)[:5]
        raise ValueError(f"Unmapped ONNX module tensors for '{model.model_key}': {sample}")

    state_dict: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    for pt_key, ref_tensor in model.state_dict().items():
        if pt_key in remapped:
            tensor = to_tensor(remapped[pt_key], dtype=ref_tensor.dtype)
            if tuple(tensor.shape) != tuple(ref_tensor.shape):
                raise ValueError(
                    f"Shape mismatch for '{pt_key}': "
                    f"ONNX {tuple(tensor.shape)} != PT {tuple(ref_tensor.shape)}"
                )
            state_dict[pt_key] = tensor
        elif pt_key.endswith(".num_batches_tracked"):
            state_dict[pt_key] = torch.zeros_like(ref_tensor)
        else:
            missing.append(pt_key)

    if missing:
        raise KeyError(
            f"Model '{model.model_key}' missing {len(missing)} weights; first 5: {missing[:5]}"
        )
    return state_dict


def _map_scene3d_key(name: str) -> Optional[str]:
    """Translate Scene3D Lite EfficientNet-B1 + DeepLabV3+ ONNX names to GraphModel keys."""
    if name.startswith("encoder.blocks."):
        parts = name.split(".", 4)
        stage_idx, block_idx, rest = int(parts[2]), int(parts[3]), parts[4]
        layer_prefix = f"layers.{1 + stage_idx}.{block_idx}."
        if stage_idx == 0:
            leaf_rules = (("conv_dw.", "dw."), ("conv_pw.", "proj."), ("se.", "se."))
        else:
            leaf_rules = (
                ("conv_pw.", "expand."),
                ("conv_dw.", "dw."),
                ("conv_pwl.", "proj."),
                ("se.", "se."),
            )
        mapped_leaf = rewrite_prefix(rest, leaf_rules)
        return layer_prefix + mapped_leaf if mapped_leaf is not None else None

    return rewrite_prefix(name, _SCENE3D_HEAD_RULES)


_AUTOSTEER_RULES: List[Tuple[str, str]] = [
    *_backbone_rules("net", offset=0),
    ("fpn.h1.", "layers.13."),
    ("fpn.h2.", "layers.16."),
    ("head.", "layers.17."),
]

_AUTODRIVE_RULES: List[Tuple[str, str]] = [
    *_backbone_rules("backbone", offset=1),
    ("head.conv_1.", "layers.13.conv."),
    ("head.conv_2.", "layers.14.conv."),
    ("head.conv_3.", "layers.15.conv."),
    ("head.fc1.0.", "layers.17.fc1.linear."),
    ("head.fc2.0.", "layers.17.fc2.linear."),
    ("head.distance_head.0.", "layers.17.distance_head.linear."),
    ("head.curvature_head.0.", "layers.17.curvature_head.linear."),
    ("head.flag_head.", "layers.17.flag_head.linear."),
]

_AUTOSPEED_RULES: List[Tuple[str, str]] = [
    *_backbone_rules("net", offset=0),
    ("fpn.h1.", "layers.13."),
    ("fpn.h2.", "layers.16."),
    ("fpn.h3.", "layers.17."),
    ("fpn.h4.", "layers.19."),
    ("fpn.h5.", "layers.20."),
    ("fpn.h6.", "layers.22."),
    ("head.", "layers.23."),
]

_SCENE3D_HEAD_RULES: Tuple[Tuple[str, str], ...] = (
    ("encoder.conv_stem.", "layers.0.conv."),
    ("decoder.aspp.0.convs.0.0.", "layers.8.convs.0.conv."),
    ("decoder.aspp.0.convs.1.0.0.", "layers.8.convs.1.dw."),
    ("decoder.aspp.0.convs.1.0.1.", "layers.8.convs.1.pw."),
    ("decoder.aspp.0.convs.2.0.0.", "layers.8.convs.2.dw."),
    ("decoder.aspp.0.convs.2.0.1.", "layers.8.convs.2.pw."),
    ("decoder.aspp.0.convs.3.0.0.", "layers.8.convs.3.dw."),
    ("decoder.aspp.0.convs.3.0.1.", "layers.8.convs.3.pw."),
    ("decoder.aspp.0.convs.4.1.", "layers.8.convs.4.conv."),
    ("decoder.aspp.0.project.0.", "layers.8.project.conv."),
    ("decoder.aspp.1.0.", "layers.9.dw."),
    ("decoder.aspp.1.1.", "layers.9.pw."),
    ("decoder.block1.0.", "layers.11.conv."),
    ("decoder.block2.0.0.", "layers.13.dw."),
    ("decoder.block2.0.1.", "layers.13.pw."),
    ("head.0.", "layers.14.conv."),
)


class _PrefixGraphConverter(BaseOnnxConverter):
    """Rule-based ONNX converter mapping keys via `prefix_rules`."""

    prefix_rules: Tuple[Tuple[str, str], ...] = ()

    def map_key(self, onnx_name: str) -> Optional[str]:
        return rewrite_prefix(onnx_name, self.prefix_rules)

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        return _map_by_rules(onnx_tensors, model, self.map_key)


@CONVERTER_REGISTRY.register("lane")
@CONVERTER_REGISTRY.register("visionpilot_autosteer2")
class LaneConverter(_PrefixGraphConverter):
    """ONNX converter for row-anchor ego-lane predictor."""

    model_key: str = "lane"
    yaml_name: str = "lane"
    prefix_rules = tuple(_AUTOSTEER_RULES)


@CONVERTER_REGISTRY.register("drive")
@CONVERTER_REGISTRY.register("visionpilot_autodrive")
class DriveConverter(_PrefixGraphConverter):
    """ONNX converter for temporal CIPO distance and curvature predictor."""

    model_key: str = "drive"
    yaml_name: str = "drive"
    prefix_rules = tuple(_AUTODRIVE_RULES)


@CONVERTER_REGISTRY.register("cipo")
@CONVERTER_REGISTRY.register("visionpilot_autospeed2")
class CipoConverter(_PrefixGraphConverter):
    """ONNX converter for 4-class vehicle and CIPO detector."""

    model_key: str = "cipo"
    yaml_name: str = "cipo"
    prefix_rules = tuple(_AUTOSPEED_RULES)


@CONVERTER_REGISTRY.register("depth")
@CONVERTER_REGISTRY.register("visionpilot_scene3d_lite")
class DepthConverter(_PrefixGraphConverter):
    """ONNX converter for monocular 3D depth estimator."""

    model_key: str = "depth"
    yaml_name: str = "depth"

    def map_key(self, onnx_name: str) -> Optional[str]:
        return _map_scene3d_key(onnx_name)
