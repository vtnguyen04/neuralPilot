"""Deterministic ONNX-to-GraphModel converter for DDRNet23-Slim segmentation (`seg.yaml`)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from neuro_pilot.nn.zoo.base import CONVERTER_REGISTRY, BaseOnnxConverter, BaseZooModel
from neuro_pilot.nn.zoo.converters.utils import to_tensor

_CONV_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("model.conv1.0.weight_fused_1", "model.conv1.0.bias_fused_1", "layers.1.conv"),
    ("model.conv1.3.weight_fused_1", "model.conv1.3.bias_fused_1", "layers.2.conv"),
    ("model.layer1.0.conv1.weight_fused_1", "val_31_fused_1", "layers.3.0.conv1"),
    ("model.layer1.0.conv2.weight_fused_1", "val_42_fused_1", "layers.3.0.conv2"),
    ("model.layer1.1.conv1.weight_fused_1", "val_53_fused_1", "layers.3.1.conv1"),
    ("model.layer1.1.conv2.weight_fused_1", "val_64_fused_1", "layers.3.1.conv2"),
    ("model.layer2.0.conv1.weight_fused_1", "val_75_fused_1", "layers.4.conv1"),
    ("model.layer2.0.conv2.weight_fused_1", "val_86_fused_1", "layers.4.conv2"),
    ("model.layer2.0.downsample.0.weight_fused_1", "val_97_fused_1", "layers.4.downsample"),
    ("model.layer2.1.conv1.weight_fused_1", "val_108_fused_1", "layers.5.conv1"),
    ("model.layer2.1.conv2.weight_fused_1", "val_119_fused_1", "layers.5.conv2"),
    ("model.layer3.0.conv1.weight_fused_1", "val_130_fused_1", "layers.6.conv1"),
    ("model.layer3.0.conv2.weight_fused_1", "val_141_fused_1", "layers.6.conv2"),
    ("model.layer3.0.downsample.0.weight_fused_1", "val_152_fused_1", "layers.6.downsample"),
    ("model.layer3.1.conv1.weight_fused_1", "val_163_fused_1", "layers.7.conv1"),
    ("model.layer3.1.conv2.weight_fused_1", "val_174_fused_1", "layers.7.conv2"),
    ("model.layer3_.0.conv1.weight_fused_1", "val_185_fused_1", "layers.8.conv1"),
    ("model.layer3_.0.conv2.weight_fused_1", "val_196_fused_1", "layers.8.conv2"),
    ("model.layer3_.1.conv1.weight_fused_1", "val_207_fused_1", "layers.9.conv1"),
    ("model.layer3_.1.conv2.weight_fused_1", "val_218_fused_1", "layers.9.conv2"),
    ("model.down3.0.weight_fused_1", "val_229_fused_1", "layers.10.down.0"),
    ("model.compression3.0.weight_fused_1", "val_240_fused_1", "layers.10.comp"),
    ("model.layer4.0.conv1.weight_fused_1", "val_255_fused_1", "layers.11.conv1"),
    ("model.layer4.0.conv2.weight_fused_1", "val_266_fused_1", "layers.11.conv2"),
    ("model.layer4.0.downsample.0.weight_fused_1", "val_277_fused_1", "layers.11.downsample"),
    ("model.layer4.1.conv1.weight_fused_1", "val_288_fused_1", "layers.12.conv1"),
    ("model.layer4.1.conv2.weight_fused_1", "val_299_fused_1", "layers.12.conv2"),
    ("model.layer4_.0.conv1.weight_fused_1", "val_310_fused_1", "layers.13.conv1"),
    ("model.layer4_.0.conv2.weight_fused_1", "val_321_fused_1", "layers.13.conv2"),
    ("model.layer4_.1.conv1.weight_fused_1", "val_332_fused_1", "layers.14.conv1"),
    ("model.layer4_.1.conv2.weight_fused_1", "val_343_fused_1", "layers.14.conv2"),
    ("model.down4.0.weight_fused_1", "val_354_fused_1", "layers.15.down.0"),
    ("model.down4.3.weight_fused_1", "val_365_fused_1", "layers.15.down.2"),
    ("model.compression4.0.weight_fused_1", "val_376_fused_1", "layers.15.comp"),
    ("model.layer5_.0.conv1.weight_fused_1", "val_390_fused_1", "layers.16.conv1"),
    ("model.layer5_.0.conv2.weight_fused_1", "val_401_fused_1", "layers.16.conv2"),
    ("model.layer5_.0.conv3.weight_fused_1", "val_412_fused_1", "layers.16.conv3"),
    ("model.layer5_.0.downsample.0.weight_fused_1", "val_423_fused_1", "layers.16.downsample"),
    ("model.layer5.0.conv1.weight_fused_1", "val_434_fused_1", "layers.17.conv1"),
    ("model.layer5.0.conv2.weight_fused_1", "val_445_fused_1", "layers.17.conv2"),
    ("model.layer5.0.conv3.weight_fused_1", "val_456_fused_1", "layers.17.conv3"),
    ("model.layer5.0.downsample.0.weight_fused_1", "val_467_fused_1", "layers.17.downsample"),
    ("model.final_layer.conv1.weight_fused_1", "val_621_fused_1", "layers.19.conv1"),
    ("model.final_layer.conv2.weight", "model.final_layer.conv2.bias", "layers.19.conv2"),
)

_AFFINE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("val_19", "val_18", "layers.18.s0_act"),
    ("val_35", "val_34", "layers.18.scale_acts.0"),
    ("val_55", "val_54", "layers.18.proc_acts.0"),
    ("val_70", "val_69", "layers.18.scale_acts.1"),
    ("val_90", "val_89", "layers.18.proc_acts.1"),
    ("val_110", "val_109", "layers.18.scale_acts.2"),
    ("val_125", "val_124", "layers.18.proc_acts.2"),
    ("val_145", "val_144", "layers.18.scale_acts.3"),
    ("val_165", "val_164", "layers.18.proc_acts.3"),
    ("val_180", "val_179", "layers.18.comp_act"),
    ("val_200", "val_199", "layers.18.short_act"),
    ("val_220", "val_219", "layers.19.act"),
)

_DAPPM_CONVS: Tuple[Tuple[str, str], ...] = (
    ("model.spp.scale0.2.weight", "layers.18.s0_conv.weight"),
    ("model.spp.scale1.3.weight", "layers.18.scale_convs.0.weight"),
    ("model.spp.process1.2.weight", "layers.18.proc_convs.0.weight"),
    ("model.spp.scale2.3.weight", "layers.18.scale_convs.1.weight"),
    ("model.spp.process2.2.weight", "layers.18.proc_convs.1.weight"),
    ("model.spp.scale3.3.weight", "layers.18.scale_convs.2.weight"),
    ("model.spp.process3.2.weight", "layers.18.proc_convs.2.weight"),
    ("model.spp.scale4.3.weight", "layers.18.scale_convs.3.weight"),
    ("model.spp.process4.2.weight", "layers.18.proc_convs.3.weight"),
    ("model.spp.compression.2.weight", "layers.18.comp_conv.weight"),
    ("model.spp.shortcut.2.weight", "layers.18.short_conv.weight"),
)


@CONVERTER_REGISTRY.register("seg")
@CONVERTER_REGISTRY.register("ddrnet23_slim")
class SegConverter(BaseOnnxConverter):
    """Map DDRNet23-Slim ONNX initializers into exact `seg.yaml` GraphModel tensors."""

    model_key: str = "seg"
    yaml_name: str = "seg"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {
            "layers.0.mean": to_tensor(onnx_tensors["view"]).reshape(1, 3, 1, 1),
            "layers.0.std": to_tensor(onnx_tensors["view_1"]).reshape(1, 3, 1, 1),
        }
        for w_src, b_src, dst in _CONV_PAIRS:
            sd[f"{dst}.weight"] = to_tensor(onnx_tensors[w_src])
            sd[f"{dst}.bias"] = to_tensor(onnx_tensors[b_src])
        for w_src, b_src, dst in _AFFINE_PAIRS:
            sd[f"{dst}.weight"] = to_tensor(onnx_tensors[w_src])
            sd[f"{dst}.bias"] = to_tensor(onnx_tensors[b_src])
        for src, dst in _DAPPM_CONVS:
            sd[dst] = to_tensor(onnx_tensors[src])
        return sd
