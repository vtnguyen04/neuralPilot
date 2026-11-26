"""ModelZoo helper and command-conditioned adapter for fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo import (
    CONVERTER_REGISTRY,
    ZOO_REGISTRY,
    BaseZooModel,
    GraphModel,
    ParityReport,
    build_zoo_model,
    convert_onnx_to_pt,
    download_weights,
    list_downloadable,
    load_zoo_pt,
    register_all_zoo_models,
    verify_onnx_to_pt_parity,
)


class CommandAdapter(nn.Module):
    """Command-conditioned waypoint and control head attached to any Zoo model."""

    def __init__(
        self,
        base: BaseZooModel,
        feat_dim: int,
        num_commands: int = 7,
        num_waypoints: int = 12,
        hidden_dim: int = 256,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.feat_dim = feat_dim
        self.num_waypoints = num_waypoints
        self.set_base_trainable(not freeze_base)

        self.cmd_embed = nn.Embedding(num_commands, 32)
        self.speed_proj = nn.Linear(1, 16)
        in_dim = feat_dim + 48
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.wp_head = nn.Linear(hidden_dim, num_waypoints * 2)
        self.ctrl_head = nn.Linear(hidden_dim, 2)

    def set_base_trainable(self, trainable: bool) -> None:
        for param in self.base.parameters():
            param.requires_grad = trainable

    def extract_feature(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "outputs" in outputs:
            vec = outputs["outputs"]
            if vec.shape[-1] >= self.feat_dim + 2:
                return vec[..., -(self.feat_dim + 2) : -2]
            return vec[..., -self.feat_dim :]

        if "plan" in outputs:
            plan = outputs["plan"]
            return plan.reshape(plan.shape[0], -1)[..., : self.feat_dim]

        first = next(iter(outputs.values()))
        return first.reshape(first.shape[0], -1)[..., : self.feat_dim]

    def forward(
        self,
        onnx_inputs: Dict[str, torch.Tensor],
        command_idx: torch.Tensor,
        speed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_out = self.base.forward_onnx_dict(onnx_inputs)
        feat = self.extract_feature(base_out)
        cmd_feat = self.cmd_embed(command_idx.long())
        speed_feat = self.speed_proj(speed.float().reshape(-1, 1))

        hidden = self.mlp(torch.cat((feat, cmd_feat, speed_feat), dim=-1))
        waypoints = self.wp_head(hidden).reshape(-1, self.num_waypoints, 2)
        ctrl = self.ctrl_head(hidden)
        return {
            "waypoints_xy": waypoints,
            "target_speed": ctrl[:, 0],
            "target_curvature": ctrl[:, 1],
            "base_outputs": base_out,
        }


class ModelZoo:
    """Entry point for building, converting, and verifying Zoo models."""

    @staticmethod
    def list_models() -> List[str]:
        register_all_zoo_models()
        return ZOO_REGISTRY.keys()

    @staticmethod
    def list_converters() -> List[str]:
        register_all_zoo_models()
        return CONVERTER_REGISTRY.keys()

    @staticmethod
    def list_weights() -> List[str]:
        """Return canonical model keys supporting automatic checkpoint download."""
        return list_downloadable()

    @staticmethod
    def download(
        key: str,
        dest_dir: Optional[str | Path] = None,
        force: bool = False,
    ) -> Path:
        """Download and cache a pretrained `.pt` checkpoint for `key`."""
        return download_weights(key, dest_dir=dest_dir, force=force)

    @staticmethod
    def build(
        cfg_or_key: str | Path,
        scale: Optional[str] = None,
        pt_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> GraphModel:
        return build_zoo_model(cfg_or_key, scale=scale, pt_path=pt_path, **kwargs)

    @staticmethod
    def load(pt_path: str | Path, device: str = "cpu", strict: bool = True) -> GraphModel:
        """Load any converted `.pt` checkpoint directly (no `.onnx` file required)."""
        return load_zoo_pt(pt_path, device=device, strict=strict)

    @staticmethod
    def export_onnx(
        model_or_pt: GraphModel | str | Path,
        onnx_path: str | Path,
        opset: int = 17,
        **kwargs: Any,
    ) -> Path:
        """Export a GraphModel or `.pt` checkpoint back to ONNX."""
        model = (
            model_or_pt
            if isinstance(model_or_pt, GraphModel)
            else load_zoo_pt(model_or_pt)
        )
        return model.export_onnx(onnx_path, opset=opset, **kwargs)

    @staticmethod
    def predictor(
        model_or_pt: GraphModel | str | Path,
        device: Optional[str] = None,
    ) -> Any:
        """Create a domain-aware predictor for a GraphModel or `.pt` checkpoint."""
        from neuro_pilot.engine.predictors import create_zoo_predictor

        model = (
            model_or_pt
            if isinstance(model_or_pt, GraphModel)
            else load_zoo_pt(model_or_pt, device=device or "cpu")
        )
        return create_zoo_predictor(model, device=device)

    @staticmethod
    def convert_onnx(
        model_key: str,
        onnx_path: str | Path,
        pt_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BaseZooModel, List[str], List[str], int]:
        register_all_zoo_models()
        return convert_onnx_to_pt(model_key, onnx_path, pt_path, config=config)

    @staticmethod
    def verify_parity(
        model_key: str,
        onnx_path: str | Path,
        pt_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
        max_abs_tol: float = 1e-2,
        min_cos_sim: float = 0.9999,
        device: str = "cpu",
    ) -> ParityReport:
        register_all_zoo_models()
        return verify_onnx_to_pt_parity(
            model_key=model_key,
            onnx_path=onnx_path,
            pt_output_path=pt_path,
            config=config,
            max_abs_tol=max_abs_tol,
            min_cos_sim=min_cos_sim,
            device=device,
        )

    @staticmethod
    def attach_adapter(
        base: BaseZooModel,
        feat_dim: int,
        num_commands: int = 7,
        num_waypoints: int = 12,
        freeze_base: bool = True,
    ) -> CommandAdapter:
        return CommandAdapter(
            base=base,
            feat_dim=feat_dim,
            num_commands=num_commands,
            num_waypoints=num_waypoints,
            freeze_base=freeze_base,
        )
