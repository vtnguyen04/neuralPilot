"""ONNX->PT numerical parity, YAML build, and backward tests for openpilot models."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

from neuro_pilot.nn.zoo import (
    CONVERTER_REGISTRY,
    ZOO_REGISTRY,
    build_zoo_model,
    register_all_zoo_models,
    verify_onnx_to_pt_parity,
)

ROOT = Path(__file__).resolve().parents[2]
YAML_DIR = ROOT / "neuro_pilot" / "cfg" / "models" / "zoo"
DEFAULT_VINFAST = ROOT.parent / "Vinfast" / "onnx_reverse_engineering"
VINFAST_DIR = Path(os.environ.get("NEUROPILOT_VINFAST_DIR", str(DEFAULT_VINFAST)))
ONNX_ROOT = VINFAST_DIR / "models"
PT_DIR = VINFAST_DIR / "converted_pt" / "planner"

# (model_key, yaml_name, rel_onnx, min_params, max_abs, max_rel_l2, max_mean_abs)
CASES: Tuple[Tuple[str, str, str, int, float, float, float], ...] = (
    (
        "driver",
        "driver.yaml",
        "openpilot/dmonitoring_model.onnx",
        3_600_000,
        0.005,
        0.001,
        0.001,
    ),
    (
        "pilot_s",
        "pilot_s.yaml",
        "openpilot/driving_supercombo.onnx",
        34_000_000,
        0.20,
        0.002,
        0.005,
    ),
    (
        "pilot_l",
        "pilot_l.yaml",
        "openpilot_chestnut/fairy_tale_big.onnx",
        380_000_000,
        0.10,
        0.002,
        0.003,
    ),
    (
        "pilot_nav",
        "pilot_nav.yaml",
        "openpilot_chestnut/n16_nav_big.onnx",
        390_000_000,
        0.25,
        0.002,
        0.003,
    ),
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_openpilot_registry() -> None:
    register_all_zoo_models()
    for key, yaml_name, *_ in CASES:
        assert key in ZOO_REGISTRY
        assert key in CONVERTER_REGISTRY
        assert (YAML_DIR / yaml_name).is_file()


@pytest.mark.parametrize("key,yaml_name,rel_onnx,min_params,max_abs,max_rel_l2,max_mean_abs", CASES)
def test_openpilot_parity(
    key: str,
    yaml_name: str,
    rel_onnx: str,
    min_params: int,
    max_abs: float,
    max_rel_l2: float,
    max_mean_abs: float,
) -> None:
    onnx_path = ONNX_ROOT / rel_onnx
    if not onnx_path.is_file():
        pytest.skip(f"ONNX file not present: {onnx_path}")
    pt_path = PT_DIR / f"{key}.pt"

    report = verify_onnx_to_pt_parity(
        model_key=key,
        onnx_path=onnx_path,
        pt_output_path=pt_path,
        max_abs_tol=max_abs,
        min_cos_sim=0.9999,
        seed=42,
        device=_device(),
    )
    assert report.missing_keys == [] and report.unexpected_keys == []
    assert report.num_params >= min_params and report.all_passed and pt_path.is_file()
    for out in report.outputs:
        assert out.cosine_sim >= 0.9999
        assert out.rel_l2_err <= max_rel_l2
        assert out.mean_abs_diff <= max_mean_abs

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "yaml_name,pt_name",
    [
        ("driver.yaml", "driver.pt"),
        ("pilot_s.yaml", "pilot_s.pt"),
    ],
)
def test_planner_yaml_and_backward(yaml_name: str, pt_name: str) -> None:
    """Build directly from module-list YAML, load .pt strict=True, and verify backprop."""
    pt_path = PT_DIR / pt_name
    model = build_zoo_model(YAML_DIR / yaml_name, pt_path=pt_path if pt_path.is_file() else None)
    model.train()
    inputs: Dict[str, torch.Tensor] = model.make_sample_inputs(seed=7)
    loss = model.forward_onnx_dict(inputs)["outputs"].float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    del model
    gc.collect()
