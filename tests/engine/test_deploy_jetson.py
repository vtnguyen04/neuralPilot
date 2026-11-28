"""Unit tests for TensorRT BF16 builder, Jetson SSH deploy module, and CLI hooks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from neuro_pilot import NeuroPilot
from neuro_pilot.deploy import (
    JetsonTarget,
    build_remote_trtexec_cmd,
    build_scp_cmd,
    build_ssh_cmd,
    deploy_to_jetson,
)
from neuro_pilot.engine.trt_builder import build_trtexec_cmd


def test_trtexec_cmd_supports_bf16_fp16_int8_and_workspace_mb(tmp_path: Path) -> None:
    onnx_p = tmp_path / "pilot_mhp_v2.onnx"
    eng_p = tmp_path / "pilot_mhp_v2.engine"

    bf16_cmd = build_trtexec_cmd(
        onnx_p, eng_p, half=True, bf16=True, workspace_mb=1024, trtexec_bin="/opt/trtexec"
    )
    assert bf16_cmd[0] == "/opt/trtexec"
    assert "--bf16" in bf16_cmd and "--fp16" not in bf16_cmd
    assert "--memPoolSize=workspace:1024M" in bf16_cmd

    fp16_cmd = build_trtexec_cmd(onnx_p, eng_p, half=True, bf16=False, workspace_gb=2)
    assert "--fp16" in fp16_cmd and "--bf16" not in fp16_cmd
    assert "--memPoolSize=workspace:2048M" in fp16_cmd

    fp32_cmd = build_trtexec_cmd(onnx_p, eng_p, half=False, bf16=False, int8=True)
    assert "--fp16" not in fp32_cmd and "--bf16" not in fp32_cmd and "--int8" in fp32_cmd


def test_jetson_target_validation_and_ssh_scp_builders(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JetsonTarget(host="", remote_dir="~/models")
    with pytest.raises(ValueError):
        JetsonTarget(host="orin@10.0.0.2", remote_dir="~/models", precision="fp8")

    target = JetsonTarget(
        host="orin@10.0.0.2",
        remote_dir="~/BFMC/models",
        port=2222,
        precision="bf16",
        workspace_mb=1024,
        ssh_key="~/.ssh/id_orin",
    )
    onnx_file = tmp_path / "pilot_mhp_v2.onnx"
    meta_file = tmp_path / "pilot_mhp_v2.meta.json"
    scp_cmd = build_scp_cmd([onnx_file, meta_file], target)
    assert scp_cmd[:5] == ["scp", "-P", "2222", "-i", "~/.ssh/id_orin"]
    assert scp_cmd[-1] == "orin@10.0.0.2:~/BFMC/models/"

    trt_shell = build_remote_trtexec_cmd(
        "~/BFMC/models/pilot_mhp_v2.onnx",
        "~/BFMC/models/pilot_mhp_v2.engine",
        target,
    )
    assert "/usr/src/tensorrt/bin/trtexec" in trt_shell
    assert "--bf16" in trt_shell and "--fp16" not in trt_shell
    assert "--memPoolSize=workspace:1024M" in trt_shell
    assert "~/BFMC/models/pilot_mhp_v2.onnx" in trt_shell

    ssh_cmd = build_ssh_cmd(target, trt_shell)
    assert ssh_cmd[:5] == ["ssh", "-p", "2222", "-i", "~/.ssh/id_orin"]
    assert ssh_cmd[5] == "orin@10.0.0.2"


def test_deploy_to_jetson_dry_run_and_mocked_execution(tmp_path: Path) -> None:
    onnx_file = tmp_path / "pilot_mhp_v2.onnx"
    onnx_file.write_bytes(b"fake-onnx")
    meta_file = tmp_path / "pilot_mhp_v2.meta.json"
    meta_file.write_text("{}", encoding="utf-8")

    target = JetsonTarget(host="orin@10.0.0.2", remote_dir="~/models", precision="bf16")
    plan = deploy_to_jetson(onnx_file, target, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["remote_engine"] == "~/models/pilot_mhp_v2.engine"
    assert len(plan["commands"]) == 3
    assert "mkdir -p ~/models" in plan["commands"][0][-1]
    assert str(meta_file) in plan["commands"][1]
    assert "--bf16" in plan["commands"][2][-1]

    ran = []
    with patch("neuro_pilot.deploy.jetson.subprocess.run") as mock_run:
        mock_run.side_effect = lambda cmd, check=True: ran.append(cmd)
        res = deploy_to_jetson(onnx_file, target, dry_run=False)
    assert res["dry_run"] is False and len(ran) == 3

    skip_res = deploy_to_jetson(onnx_file, target, skip_build=True, dry_run=True)
    assert skip_res["remote_engine"] == "" and len(skip_res["commands"]) == 2


def test_neuropilot_deploy_method_and_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    model = NeuroPilot("depth", device="cpu")
    res = model.deploy(
        host="orin@10.0.0.2",
        remote_dir="~/models",
        precision="bf16",
        workspace_mb=1024,
        work_dir=tmp_path / "out_deploy",
        dry_run=True,
    )
    assert Path(res["local_onnx"]).is_file()
    assert res["remote_engine"] == "~/models/depth.engine"
    assert "--bf16" in res["commands"][2][-1]

    from neuro_pilot.entrypoint import main as cli_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neuropilot",
            "deploy",
            "--model",
            res["local_onnx"],
            "--host",
            "orin@10.0.0.2",
            "--remote-dir",
            "~/models",
            "--precision",
            "bf16",
            "--dry-run",
        ],
    )
    cli_main()


def test_examples_run_as_standalone_scripts_and_packaging_clean(tmp_path: Path, monkeypatch) -> None:
    import runpy

    import cv2
    import numpy as np

    from neuro_pilot.tools.demo import _resolve_model

    root = Path(__file__).resolve().parents[2]

    # 1. Verify examples/ is NOT an importable Python package and pyproject packages ONLY neuro_pilot
    assert list((root / "examples").rglob("__init__.py")) == []
    pyproj = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["neuro_pilot*"]' in pyproj
    assert "website*" in pyproj and "examples*" in pyproj
    assert (root / "MANIFEST.in").is_file()

    # 2. Run standalone example scripts via runpy (exactly how users invoke `python3 examples/*.py`)
    img_path = tmp_path / "road.jpg"
    cv2.imwrite(str(img_path), np.full((256, 512, 3), 90, dtype=np.uint8))

    perc_out = tmp_path / "perc_grid.jpg"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "examples/perception.py",
            "--image", str(img_path),
            "--out", str(perc_out),
            "--seg", "seg",
            "--depth", "depth",
            "--cipo", "cipo",
            "--lane", "lane",
            "--device", "cpu",
        ],
    )
    runpy.run_path(str(root / "examples" / "perception.py"), run_name="__main__")
    assert perc_out.is_file() and cv2.imread(str(perc_out)).shape == (720, 1280, 3)

    vq_out = tmp_path / "vq.jpg"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "examples/world_model.py",
            "--image", str(img_path),
            "--out", str(vq_out),
            "--enc", "vq_enc",
            "--dec", "vq_dec",
            "--scale", "tiny",
            "--device", "cpu",
        ],
    )
    runpy.run_path(str(root / "examples" / "world_model.py"), run_name="__main__")
    assert vq_out.is_file()

    # Demo resolver falls back to hub key when pt_root is None or missing file
    assert _resolve_model(None, "planner/pilot_l.pt", "pilot_l.pt") == "pilot_l.pt"
    assert _resolve_model(tmp_path, "planner/pilot_l.pt", "pilot_l.pt") == "pilot_l.pt"

    # Deploy example dry-run as standalone script
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "examples/deploy_jetson.py",
            "--model", "depth",
            "--host", "orin@10.0.0.2",
            "--remote-dir", "~/models",
            "--work-dir", str(tmp_path / "ex_deploy"),
            "--dry-run",
        ],
    )
    runpy.run_path(str(root / "examples" / "deploy_jetson.py"), run_name="__main__")
    assert (tmp_path / "ex_deploy" / "depth.onnx").is_file()
