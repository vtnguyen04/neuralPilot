"""ONNX-to-GraphModel converters for dual-camera policy planners and cabin monitoring."""

from __future__ import annotations

from typing import Any, Dict

import torch

from neuro_pilot.nn.zoo.base import CONVERTER_REGISTRY, BaseOnnxConverter, BaseZooModel
from neuro_pilot.nn.zoo.converters.planner_map import (
    AnonStream,
    copy_wb,
    map_convnext_layers,
    map_fastvit_layers,
    map_hydra_layer,
    map_point_policy_layer,
    map_queue_and_pack,
    map_residual_layer,
    map_temporal_layers,
    to_tensor,
)

_OFF_HEADS_WITH_ACTION = (
    ("plan", True),
    ("lead", True),
    ("lead_prob", False),
    ("action", True),
    ("desire_state", False),
)
_OFF_HEADS_BASE = (
    ("plan", True),
    ("lead", True),
    ("lead_prob", False),
    ("desire_state", False),
)


@CONVERTER_REGISTRY.register("pilot_s")
@CONVERTER_REGISTRY.register("openpilot_small")
class PilotSmallConverter(BaseOnnxConverter):
    """Converter for FastViT-T12 dual-camera E2E planner (34.30M)."""

    model_key: str = "pilot_s"
    yaml_name: str = "pilot_s"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        stream = AnonStream(onnx_tensors)
        map_queue_and_pack(onnx_tensors, sd, queue_idx=0, pack_idx=20)
        map_fastvit_layers(onnx_tensors, sd, start_idx=1)
        map_point_policy_layer(onnx_tensors, sd, layer_idx=11)
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.off_policy.temporal_summarizer", 12, 13, 4, 1, 200
        )
        map_hydra_layer(
            onnx_tensors, sd, "model.off_policy.temporal_hydra", 15, _OFF_HEADS_WITH_ACTION
        )
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.on_policy.temporal_summarizer", 16, 17, 1, 1, 200
        )
        map_residual_layer(onnx_tensors, sd, layer_idx=19)
        stream.assert_exhausted()
        return sd


@CONVERTER_REGISTRY.register("pilot_l")
@CONVERTER_REGISTRY.register("openpilot_chestnut_big")
class PilotLargeConverter(BaseOnnxConverter):
    """Converter for ConvNeXt-XL dual-camera E2E planner (382.25M)."""

    model_key: str = "pilot_l"
    yaml_name: str = "pilot_l"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        stream = AnonStream(onnx_tensors)
        map_queue_and_pack(onnx_tensors, sd, queue_idx=0, pack_idx=20)
        map_convnext_layers(
            onnx_tensors, stream, sd, "model.vision.encoder", 1,
            (256, 512, 1024, 2048), (3, 3, 27, 3), has_head_fc=True,
        )
        map_point_policy_layer(onnx_tensors, sd, layer_idx=11)
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.off_policy.temporal_summarizer", 12, 13, 4, 32, 200
        )
        map_hydra_layer(onnx_tensors, sd, "model.off_policy.temporal_hydra", 15, _OFF_HEADS_BASE)
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.on_policy.temporal_summarizer", 16, 17, 4, 32, 200
        )
        map_hydra_layer(
            onnx_tensors, sd, "model.on_policy.temporal_hydra", 19, (("action", True),)
        )
        stream.assert_exhausted()
        return sd


def _map_nav_encoders(
    ox: Dict[str, Any], stream: AnonStream, sd: Dict[str, torch.Tensor]
) -> None:
    src = "model.off_policy.temporal_summarizer"
    map_convnext_layers(
        ox, stream, sd, f"{src}.map_encoder", 11, (64, 128, 256, 512), (2, 2, 6, 2), False
    )
    copy_wb(ox, sd, f"{src}.map_projection", "layers.21.linear")
    sd["layers.22.fc1.weight"] = stream.pop((64, 47), transpose=True)
    sd["layers.22.fc1.bias"] = to_tensor(ox[f"{src}.navigation_encoder.net.0.bias"])
    sd["layers.22.fc2.weight"] = stream.pop((64, 64), transpose=True)


@CONVERTER_REGISTRY.register("pilot_nav")
@CONVERTER_REGISTRY.register("openpilot_n16_nav")
class PilotNavConverter(BaseOnnxConverter):
    """Converter for ConvNeXt-XL map-guided navigation planner (395.38M unique)."""

    model_key: str = "pilot_nav"
    yaml_name: str = "pilot_nav"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        stream = AnonStream(onnx_tensors)
        map_queue_and_pack(onnx_tensors, sd, queue_idx=0, pack_idx=33)
        map_convnext_layers(
            onnx_tensors, stream, sd, "model.vision.encoder", 1,
            (256, 512, 1024, 2048), (3, 3, 27, 3), has_head_fc=True,
        )
        _map_nav_encoders(onnx_tensors, stream, sd)
        map_point_policy_layer(onnx_tensors, sd, layer_idx=24)
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.off_policy.temporal_summarizer", 25, 26, 4, 32, 328
        )
        map_hydra_layer(onnx_tensors, sd, "model.off_policy.temporal_hydra", 28, _OFF_HEADS_BASE)

        # Verify duplicated on_policy.off_policy summarizer weights are identical
        dup_sd: Dict[str, torch.Tensor] = {}
        map_temporal_layers(
            onnx_tensors, stream, dup_sd, "model.off_policy.temporal_summarizer", 25, 26, 4, 32, 328
        )
        for key, val in dup_sd.items():
            if not torch.equal(sd[key], val):
                raise ValueError(f"Expected identical on_policy.off_policy tensor for '{key}'")

        map_hydra_layer(
            onnx_tensors, sd, "model.on_policy.off_policy.temporal_hydra", 28, (("action", True),)
        )
        map_temporal_layers(
            onnx_tensors, stream, sd, "model.on_policy.temporal_summarizer", 29, 30, 4, 32, 328
        )
        map_residual_layer(onnx_tensors, sd, layer_idx=32)
        stream.assert_exhausted()
        return sd


def _dmon_uib_dst(stage_idx: int, block_idx: int) -> str:
    if stage_idx == 2:
        if block_idx == 0:
            return "layers.4"
        return f"layers.5.{block_idx - 1}" if block_idx <= 4 else "layers.6"
    return f"layers.{7 + block_idx}" if block_idx <= 3 else f"layers.11.{block_idx - 4}"


def _map_dmon_heads(ox: Dict[str, Any], sd: Dict[str, torch.Tensor]) -> None:
    mlp_layers = (("0.1", "0"), ("1.0", "2"), ("2.0", "4"), ("3", "6"))
    for side_src, side_dst in (("lhd", "driver_lhd"), ("rhd", "driver_rhd")):
        for head in ("pose", "faceprob", "eyeprob", "eyeclosed", "phone", "sleep"):
            for src_l, dst_l in mlp_layers:
                src = f"model._driver_predictor_{side_src}.heads.{head}.{src_l}"
                copy_wb(ox, sd, src, f"layers.17.{side_dst}.{head}.{dst_l}")
    for src_l, dst_l in mlp_layers:
        copy_wb(ox, sd, f"model._summary_predictor.{src_l}", f"layers.17.summary_predictor.{dst_l}")


@CONVERTER_REGISTRY.register("driver")
@CONVERTER_REGISTRY.register("openpilot_dmonitoring")
class DriverConverter(BaseOnnxConverter):
    """Converter for MobileNetV4 cabin driver monitoring model (3.62M)."""

    model_key: str = "driver"
    yaml_name: str = "driver"

    def extract_state_dict(
        self, onnx_tensors: Dict[str, Any], model: BaseZooModel
    ) -> Dict[str, torch.Tensor]:
        ox = onnx_tensors
        sd: Dict[str, torch.Tensor] = {
            "layers.0.calib_mean": to_tensor(ox["unsqueeze"]),
            "layers.0.calib_rsqrt": to_tensor(ox["rsqrt"]),
        }
        bb = "model._vision._backbone"
        conv_map = (
            (f"{bb}.conv_stem", "layers.1.conv"),
            (f"{bb}.blocks.0.0.conv", "layers.2.conv_exp"),
            (f"{bb}.blocks.0.1.conv", "layers.2.conv_proj"),
            (f"{bb}.blocks.1.0.conv", "layers.3.conv_exp"),
            (f"{bb}.blocks.1.1.conv", "layers.3.conv_proj"),
            (f"{bb}.blocks.4.0.conv", "layers.12.conv"),
            (f"{bb}.conv_head", "layers.14.conv"),
        )
        for src, dst in conv_map:
            sd[f"{dst}.weight"] = to_tensor(ox[f"{src}.weight"])
            sd[f"{dst}.bias"] = to_tensor(ox[f"{src}.weight_bias"])

        for stage_idx in (2, 3):
            for block_idx in range(6):
                dst_base = _dmon_uib_dst(stage_idx, block_idx)
                for sub in ("dw_start", "pw_exp", "dw_mid", "pw_proj"):
                    wk = f"{bb}.blocks.{stage_idx}.{block_idx}.{sub}.conv.weight"
                    if wk in ox:
                        sd[f"{dst_base}.{sub}.weight"] = to_tensor(ox[wk])
                        sd[f"{dst_base}.{sub}.bias"] = to_tensor(ox[f"{wk}_bias"])

        copy_wb(ox, sd, f"{bb}.classifier", "layers.16.linear")
        _map_dmon_heads(ox, sd)
        return sd
