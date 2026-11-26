"""ONNX-to-PT parity, YAML graph build, and training tests for World Models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from neuro_pilot.nn.zoo import (
    CONVERTER_REGISTRY,
    ZOO_REGISTRY,
    GraphModel,
    build_zoo_model,
    register_all_zoo_models,
    verify_onnx_to_pt_parity,
)
from neuro_pilot.nn.zoo.blocks.world import sample_dit_euler

ROOT = Path(__file__).resolve().parents[2]
YAML_DIR = ROOT / "neuro_pilot" / "cfg" / "models" / "zoo"
VINFAST_DIR = Path(
    os.environ.get(
        "NEUROPILOT_VINFAST_DIR",
        str(ROOT.parent / "Vinfast" / "onnx_reverse_engineering"),
    )
)
ONNX_ROOT = VINFAST_DIR / "models"
PT_DIR = VINFAST_DIR / "converted_pt" / "world"

LATENT_DRIVE_ONNX = ONNX_ROOT / "trajectory/latent_drive/camfront-rptw-lane2-e1.onnx"
COMMA_VQ_ENC_ONNX = ONNX_ROOT / "comma_world/encoder.onnx"
COMMA_VQ_DEC_ONNX = ONNX_ROOT / "comma_world/decoder.onnx"


def test_world_models_registry() -> None:
    register_all_zoo_models()
    expected_models = {"jepa", "vq_enc", "vq_dec", "gpt", "dit", "ae"}
    expected_converters = {"jepa", "vq_enc", "vq_dec"}
    assert expected_models.issubset(set(ZOO_REGISTRY.keys()))
    assert expected_converters.issubset(set(CONVERTER_REGISTRY.keys()))
    assert isinstance(build_zoo_model("gpt", scale="tiny"), GraphModel)


@pytest.mark.skipif(not LATENT_DRIVE_ONNX.is_file(), reason="LatentDrive ONNX not present")
def test_jepa_parity() -> None:
    pt_path = PT_DIR / "jepa.pt"
    report = verify_onnx_to_pt_parity(
        model_key="jepa",
        onnx_path=LATENT_DRIVE_ONNX,
        pt_output_path=pt_path,
        max_abs_tol=1e-3,
        min_cos_sim=0.99999,
        seed=42,
        device="cpu",
    )
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert report.num_params == 310_024_963
    assert report.all_passed
    assert pt_path.is_file()


@pytest.mark.skipif(not COMMA_VQ_ENC_ONNX.is_file(), reason="VQ encoder ONNX not present")
def test_vq_enc_parity() -> None:
    pt_path = PT_DIR / "vq_enc.pt"
    gen = torch.Generator().manual_seed(42)
    sample = {
        "big_img": torch.rand((1, 3, 128, 256), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    }
    report = verify_onnx_to_pt_parity(
        model_key="vq_enc",
        onnx_path=COMMA_VQ_ENC_ONNX,
        pt_output_path=pt_path,
        max_abs_tol=0.0,
        min_cos_sim=0.99999,
        custom_inputs=sample,
        device="cpu",
    )
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert report.all_passed

    reloaded = build_zoo_model(YAML_DIR / "vq_enc.yaml", pt_path=pt_path).eval()
    with torch.no_grad():
        assert reloaded.forward_onnx_dict(sample)["encoding_indices"].shape == (1, 8, 16)


@pytest.mark.skipif(not COMMA_VQ_DEC_ONNX.is_file(), reason="VQ decoder ONNX not present")
def test_vq_dec_parity() -> None:
    pt_path = PT_DIR / "vq_dec.pt"
    report = verify_onnx_to_pt_parity(
        model_key="vq_dec",
        onnx_path=COMMA_VQ_DEC_ONNX,
        pt_output_path=pt_path,
        max_abs_tol=1e-4,
        min_cos_sim=0.99999,
        seed=42,
        device="cpu",
    )
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert report.all_passed

    reloaded = build_zoo_model(YAML_DIR / "vq_dec.yaml", pt_path=pt_path).eval()
    with torch.no_grad():
        out = reloaded.forward_onnx_dict(reloaded.make_sample_inputs(seed=42))["big_decoded_img"]
        assert out.shape == (1, 3, 128, 256)


def test_gpt_kv_cache_and_reload(tmp_path: Path) -> None:
    model = build_zoo_model(YAML_DIR / "gpt.yaml", scale="tiny").eval()
    ids = model.make_sample_inputs(seed=7)["input_ids"]
    with torch.no_grad():
        full = model(ids, use_cache=False)
        pre, past = model(ids[:, :-4], use_cache=True)
        suf, _ = model(ids[:, -4:], past_key_values=past, use_cache=True)
    assert full.shape == (1, ids.size(1), 1025)
    assert torch.allclose(full, torch.cat((pre, suf), dim=1), atol=1e-4)

    model.train()
    model(ids).pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    ckpt = tmp_path / "gpt_tiny.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    reloaded = build_zoo_model(YAML_DIR / "gpt.yaml", scale="tiny", pt_path=ckpt).eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), reloaded(ids), atol=1e-6)


def test_dit_euler_and_reload(tmp_path: Path) -> None:
    model = build_zoo_model(YAML_DIR / "dit.yaml", scale="tiny")
    sample = model.make_sample_inputs(seed=11)
    model.train()
    out = model.forward_onnx_dict(sample)
    assert out["velocity"].shape == (1, 15, 32, 16, 32)
    assert out["plan"].shape == (1, 33, 3)

    (out["velocity"].pow(2).mean() + out["plan"].pow(2).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    sampled = sample_dit_euler(
        model=model,
        context_latents=sample["latents"][:, :14],
        rel_pose=sample["rel_pose"],
        pose_mask=sample["pose_mask"],
        frame_indices=sample["frame_indices"],
        num_steps=3,
    )
    assert sampled["latent"].shape == (1, 1, 32, 16, 32)
    assert sampled["plan"].shape == (1, 33, 3)

    ckpt = tmp_path / "dit_tiny.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    reloaded = build_zoo_model(YAML_DIR / "dit.yaml", scale="tiny", pt_path=ckpt).eval()
    with torch.no_grad():
        ref_plan = model.forward_onnx_dict(sample)["plan"]
        new_plan = reloaded.forward_onnx_dict(sample)["plan"]
        assert torch.allclose(ref_plan, new_plan)


def test_ae_and_reload(tmp_path: Path) -> None:
    full = build_zoo_model(YAML_DIR / "ae.yaml")
    enc_params = sum(p.numel() for layer in full.layers[:3] for p in layer.parameters())
    dec_params = sum(p.numel() for layer in full.layers[3:] for p in layer.parameters())
    assert 42_500_000 <= enc_params <= 43_200_000
    assert 85_000_000 <= dec_params <= 85_700_000

    tiny = build_zoo_model("ae", scale="tiny")
    sample = tiny.make_sample_inputs(seed=3)
    out = tiny.forward_onnx_dict(sample)
    assert out["latents"].shape == (1, 32, 16, 32)
    assert out["recon"].shape == (1, 6, 128, 256)
    assert tiny(torch.randn(1, 3, 256, 256))["latents"].shape == (1, 32, 16, 32)

    (out["recon"].pow(2).mean() + out["latents"].pow(2).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tiny.parameters())

    ckpt = tmp_path / "ae_tiny.pt"
    torch.save({"state_dict": tiny.state_dict()}, ckpt)
    reloaded = build_zoo_model("ae", scale="tiny", pt_path=ckpt).eval()
    tiny.eval()
    with torch.no_grad():
        assert torch.allclose(
            tiny(sample["image_pair"])["recon"],
            reloaded(sample["image_pair"])["recon"],
        )


def test_jepa_and_vq_backward() -> None:
    ld = build_zoo_model("jepa", scale="tiny")
    sample = ld.make_sample_inputs(seed=5)
    plan = ld(sample["video"], sample["status"])
    assert plan.shape == (1, 8, 3)
    plan.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ld.parameters())

    enc = build_zoo_model("vq_enc", scale="tiny")
    dec = build_zoo_model("vq_dec", scale="tiny")
    img = torch.randn(1, 3, 32, 64)

    z = img
    for layer in enc.layers[:-1]:
        z = layer(z)
    z_q, _ = enc.layers[-1].forward_ste(z)
    recon = z_q
    for layer in dec.layers[1:]:
        recon = layer(recon)

    assert recon.shape == img.shape
    (recon - img).pow(2).mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for layer in enc.layers[:-1]
        for p in layer.parameters()
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for layer in dec.layers[1:]
        for p in layer.parameters()
    )
