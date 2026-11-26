"""Integration, YAML graph, plug-and-play task, adapter, loss, and zero-DRY tests."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import torch
import yaml

from neuro_pilot.core.registry import Registry
from neuro_pilot.engine.tasks import register_driving_tasks
from neuro_pilot.engine.zoo import CommandAdapter, ModelZoo
from neuro_pilot.nn.factory import build_model
from neuro_pilot.nn.zoo import (
    FlowMatchingLoss,
    GraphModel,
    LaplaceNLL,
    MHPLaplaceLoss,
    WaypointLoss,
    build_zoo_model,
    is_zoo_yaml,
)

ROOT = Path(__file__).resolve().parents[2]
ZOO_DIR = ROOT / "neuro_pilot" / "nn" / "zoo"
YAML_DIR = ROOT / "neuro_pilot" / "cfg" / "models" / "zoo"

EXPECTED_MODELS = {
    "pilot_s",
    "pilot_l",
    "pilot_nav",
    "driver",
    "cipo",
    "lane",
    "drive",
    "depth",
    "seg",
    "seg_l",
    "jepa",
    "vq_enc",
    "vq_dec",
    "ae",
    "dit",
    "gpt",
    "pilot_det",
    "pilot_mhp",
}

EXPECTED_CONVERTERS = {
    "pilot_s",
    "pilot_l",
    "pilot_nav",
    "driver",
    "cipo",
    "lane",
    "drive",
    "depth",
    "seg",
    "seg_l",
    "jepa",
    "vq_enc",
    "vq_dec",
}


def test_registry_and_yamls_detailed_layer_graphs() -> None:
    assert EXPECTED_MODELS.issubset(set(ModelZoo.list_models()))
    assert EXPECTED_MODELS.issubset(set(Registry.list_zoo_models()))
    assert EXPECTED_CONVERTERS.issubset(set(ModelZoo.list_converters()))

    yamls = sorted(YAML_DIR.glob("*.yaml"))
    assert len(yamls) == 18
    forbidden = (
        "/home/",
        "/media/",
        ".onnx",
        ".pt",
        "pt_checkpoint",
        "onnx_source",
        "onnx_checkpoint",
    )
    for y in yamls:
        assert len(y.stem.split("_")) <= 2, f"YAML {y.name} must be 1-2 short words"
        for vendor in ("openpilot", "visionpilot", "comma"):
            assert not y.stem.startswith(vendor), f"YAML {y.name} must not have vendor prefix"
        raw = y.read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in raw, f"YAML {y.name} must not contain '{bad}'"
        cfg = yaml.safe_load(raw)
        assert "backbone" in cfg and "head" in cfg, f"{y.name} missing backbone/head"
        assert is_zoo_yaml(cfg), f"{y.name} not recognized as Zoo graph"
        total_layers = len(cfg["backbone"]) + len(cfg["head"])
        assert total_layers >= 3, f"{y.name} has too few explicit layers ({total_layers})"


def test_factory_builds_graph_models_and_respects_yaml_edits(tmp_path: Path) -> None:
    for name in (
        "driver.yaml",
        "gpt.yaml",
        "dit.yaml",
        "lane.yaml",
        "depth.yaml",
    ):
        model = build_model(str(YAML_DIR / name))
        assert isinstance(model, GraphModel)
        assert model.num_parameters() > 0

    # Verify editing QKNormBlock repeat count in YAML directly changes GraphModel depth & params
    base_cfg = yaml.safe_load((YAML_DIR / "pilot_s.yaml").read_text(encoding="utf-8"))
    base_model = build_zoo_model(YAML_DIR / "pilot_s.yaml")
    # head[2] is off-policy 4x QKNormBlock -> edit repeat from 4 to 2
    base_cfg["head"][2][1] = 2
    edited_yaml = tmp_path / "pilot_2l.yaml"
    edited_yaml.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    edited_model = build_zoo_model(edited_yaml)
    assert len(edited_model.layers[13]) == 2
    assert edited_model.num_parameters() < base_model.num_parameters()


def test_plug_and_play_self_driving_tasks() -> None:
    register_driving_tasks()
    for task_key in ("planner", "world", "lane", "depth", "driver", "bfmc"):
        task_cls = Registry.get_task(task_key)
        task = task_cls(overrides={"model_cfg": "driver"})
        model = task.build_model()
        criterion = task.build_criterion()
        assert isinstance(model, GraphModel)
        assert isinstance(criterion, torch.nn.Module)


def test_adapter_and_losses_backward() -> None:
    base = ModelZoo.build(YAML_DIR / "driver.yaml")
    adapter = ModelZoo.attach_adapter(
        base, feat_dim=64, num_commands=7, num_waypoints=12, freeze_base=True
    )
    assert isinstance(adapter, CommandAdapter)
    assert all(not p.requires_grad for p in adapter.base.parameters())

    out = adapter(base.make_sample_inputs(seed=3), torch.tensor([2]), torch.tensor([1.5]))
    wp_loss = WaypointLoss()(
        out["waypoints_xy"],
        torch.zeros_like(out["waypoints_xy"]),
        torch.stack([out["target_speed"], out["target_curvature"]], dim=-1),
        torch.zeros((1, 2)),
    )
    wp_loss["loss"].backward()
    assert all(p.grad is not None for p in adapter.wp_head.parameters())

    mu = torch.randn(2, 5, 10, 3, requires_grad=True)
    log_b = torch.zeros(2, 5, 10, 3, requires_grad=True)
    logits = torch.randn(2, 5, requires_grad=True)
    tgt = torch.randn(2, 10, 3)
    l_nll = LaplaceNLL()(mu[:, 0], log_b[:, 0], tgt)
    l_mhp = MHPLaplaceLoss()(mu, log_b, logits, tgt)["loss"]
    x0, x1 = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    l_rf = FlowMatchingLoss()(torch.randn(2, 4, 8, requires_grad=True), x0, x1)
    (l_nll + l_mhp + l_rf).backward()
    assert mu.grad is not None and logits.grad is not None


def test_pt_load_and_export_onnx_roundtrip(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = build_zoo_model("driver").eval()
    pt_file = tmp_path / "driver_test.pt"
    # Verify legacy model_key checkpoint alias still loads transparently
    torch.save({"model_key": "openpilot_dmonitoring", "state_dict": model.state_dict()}, pt_file)

    loaded = ModelZoo.load(pt_file)
    sample = loaded.make_sample_inputs(seed=11)
    with torch.no_grad():
        pt_out = loaded.forward_onnx_dict(sample)

    onnx_file = tmp_path / "dmon_roundtrip.onnx"
    loaded.export_onnx(onnx_file, opset=17, sample_inputs=sample)
    onnx.checker.check_model(str(onnx_file))

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    out_names = list(pt_out.keys())
    ort_vals = sess.run(out_names, {k: v.numpy() for k, v in sample.items()})
    for name, ort_arr in zip(out_names, ort_vals):
        assert float(np.max(np.abs(pt_out[name].numpy() - ort_arr))) < 1e-4


def test_neuropilot_unified_predict_and_export(tmp_path: Path) -> None:
    import numpy as np
    import onnx

    from neuro_pilot import NeuroPilot

    frame = np.zeros((512, 1024, 3), dtype=np.uint8)

    # 1. Save & reload .pt purely via NeuroPilot(pt_path), predict on BGR frame, export ONNX
    s3d = NeuroPilot("depth", device="cpu")
    pt_file = tmp_path / "depth.pt"
    s3d.save(pt_file)
    reloaded = NeuroPilot(pt_file, device="cpu")
    assert reloaded.task_name == "depth"
    depth_out = reloaded.predict(frame)
    assert depth_out["depth"].shape == (320, 640) and depth_out["depth_vis"].shape == (320, 640, 3)

    onnx_out = reloaded.export(format="onnx", file=str(tmp_path / "depth.onnx"), opset=17)
    onnx.checker.check_model(onnx_out)

    # 2. Verify all driving predictor families directly via NeuroPilot(...)
    op = NeuroPilot("pilot_s", device="cpu")
    op_out = op.predict(frame, wide_img=frame)
    assert op_out["plan"].shape == (33, 15) and op_out["action"].shape == (4,)

    assert "boxes" in NeuroPilot("cipo", device="cpu").predict(frame)
    assert "points" in NeuroPilot("lane", device="cpu").predict(frame)
    assert "class_map" in NeuroPilot("seg", device="cpu").predict(frame, width=256, height=128)
    assert "class_map" in NeuroPilot("seg_l", scale="tiny", device="cpu").predict(
        frame, width=128, height=128
    )
    ad = NeuroPilot("drive", device="cpu")
    assert "distance_m" in ad.predict(frame) and "cipo_prob" in ad.predict(frame)
    assert "outputs" in NeuroPilot("driver", device="cpu").predict(frame)
    mt = NeuroPilot("pilot_mhp", device="cpu")
    mt_out = mt.predict(frame, command=0)
    assert mt_out["waypoints"].shape == (10, 2) and "detections" in mt_out

    vq_enc = NeuroPilot("vq_enc", scale="tiny", device="cpu")
    vq_dec = NeuroPilot("vq_dec", scale="tiny", device="cpu")
    toks = vq_enc.predict(frame)["tokens"]
    assert vq_dec.predict(toks)["image_bgr"].shape == (32, 64, 3)

    ld = NeuroPilot("jepa", scale="tiny", device="cpu")
    assert ld.predict([frame, frame, frame, frame])["plan"].shape == (8, 3)
    assert "logits" in NeuroPilot("gpt", scale="tiny", device="cpu").predict([1, 2, 3])
    dit = NeuroPilot("dit", scale="tiny", device="cpu")
    assert "plan" in dit.predict(dit.model.make_sample_inputs(seed=1), sample_steps=2)
    assert "recon" in NeuroPilot("ae", scale="tiny", device="cpu").predict(frame)


def test_multi_backend_onnx_torchscript_and_trt_builder(tmp_path: Path) -> None:
    import numpy as np

    from neuro_pilot import NeuroPilot
    from neuro_pilot.engine.trt_builder import build_trtexec_cmd

    frame = np.full((512, 1024, 3), 120, dtype=np.uint8)

    # 1. PyTorch -> ONNX -> NeuroPilot("model.onnx").predict() auto-discovers task & matches PT
    s3d_pt = NeuroPilot("depth", device="cpu")
    ref_depth = s3d_pt.predict(frame)["depth"]

    onnx_path = s3d_pt.export(format="onnx", file=str(tmp_path / "depth.onnx"), opset=17)
    s3d_onnx = NeuroPilot(onnx_path, device="cpu")
    assert s3d_onnx.task_name == "depth"
    onnx_depth = s3d_onnx.predict(frame)["depth"]
    assert float(np.max(np.abs(ref_depth - onnx_depth))) < 1e-4

    # 2. Multi-input 7-tensor pilot_s -> ONNX -> NeuroPilot("pilot_s.onnx").predict()
    op_pt = NeuroPilot("pilot_s", device="cpu")
    ref_op = op_pt.predict(frame, wide_img=frame)
    op_pt.predictor.reset_state()
    op_onnx_path = op_pt.export(format="onnx", file=str(tmp_path / "pilot_s.onnx"), opset=17)
    op_onnx = NeuroPilot(op_onnx_path, device="cpu")
    assert op_onnx.task_name == "planner"
    got_op = op_onnx.predict(frame, wide_img=frame)
    assert float(np.max(np.abs(ref_op["plan"] - got_op["plan"]))) < 5e-4

    # 3. PyTorch -> TorchScript -> NeuroPilot("model.torchscript").predict()
    ts_path = s3d_pt.export(format="torchscript", file=str(tmp_path / "depth.torchscript"))
    s3d_ts = NeuroPilot(ts_path, device="cpu")
    assert s3d_ts.task_name == "depth"
    ts_depth = s3d_ts.predict(frame)["depth"]
    assert float(np.max(np.abs(ref_depth - ts_depth))) < 1e-5

    # 4. TensorRT 10 multi-input trtexec command generation (no hardcoded BFMC shapes)
    cmd = build_trtexec_cmd(
        onnx_path=tmp_path / "pilot_s.onnx",
        engine_path=tmp_path / "pilot_s.engine",
        half=True,
        dynamic=True,
        workspace_gb=4,
        input_specs=op_pt.model.input_specs,
        max_batch=2,
    )
    joined = " ".join(cmd)
    assert "--fp16" in joined and "new_img:" in joined and "state_feat_q:" in joined


def test_weight_hub_cache_and_auto_resolution(tmp_path: Path, monkeypatch) -> None:
    from neuro_pilot import NeuroPilot
    from neuro_pilot.nn.zoo import (
        WEIGHT_SPECS,
        download_weights,
        list_downloadable,
        resolve_weight_key,
    )

    assert len(list_downloadable()) == 15
    assert set(WEIGHT_SPECS.keys()) == set(EXPECTED_CONVERTERS) | {"pilot_mhp", "pilot_det"}
    assert resolve_weight_key("pilot_l.pt") == "pilot_l"
    assert resolve_weight_key("pilot_mhp.pt") == "pilot_mhp"
    assert resolve_weight_key("bfmc_mhp.pt") == "pilot_mhp"
    assert resolve_weight_key("openpilot_chestnut_big.pt") == "pilot_l"
    assert resolve_weight_key("seg_l.pt") == "seg_l"
    assert resolve_weight_key("nonexistent_model.pt") is None

    # Pre-populate local cache and confirm zero-network auto-resolution in NeuroPilot("driver.pt")
    cache_dir = tmp_path / "weights_cache"
    monkeypatch.setenv("NEUROPILOT_CACHE_DIR", str(cache_dir))
    cached_pt = cache_dir / "driver.pt"
    NeuroPilot("driver", device="cpu").save(cached_pt)

    assert download_weights("driver") == cached_pt
    loaded = NeuroPilot("driver.pt", device="cpu")
    assert loaded.task_name == "driver"
    loaded_flag = NeuroPilot("driver", pretrained=True, device="cpu")
    assert loaded_flag.task_name == "driver"


def test_clean_code_and_zero_dry_guardrails() -> None:
    py_files = sorted(
        list(ZOO_DIR.rglob("*.py"))
        + list((ROOT / "neuro_pilot" / "engine" / "predictors").rglob("*.py"))
        + list((ROOT / "neuro_pilot" / "engine" / "backend").rglob("*.py"))
        + [ROOT / "neuro_pilot" / "engine" / "zoo.py"]
        + [ROOT / "neuro_pilot" / "engine" / "tasks.py"]
        + [ROOT / "neuro_pilot" / "engine" / "trt_builder.py"]
    )
    assert len(py_files) >= 15
    block_classes = {}
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) < 300, f"{path.name} has {len(lines)} lines (limit < 300)"
        assert "/home/" not in text and "/media/" not in text, f"Hardcoded path in {path.name}"
        assert "from __future__ import annotations" in text, f"Missing future import in {path.name}"
        assert not re.search(r"^\s*print\(", text, re.M), f"Debug print() in {path.name}"
        for line_no, line in enumerate(lines, 1):
            assert len(line) <= 100, f"{path.name}:{line_no} too long ({len(line)} chars)"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                assert span <= 80, f"{path.name}:{node.name} is {span} lines (> 80)"
        if "blocks/" in str(path):
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    prev = block_classes.get(node.name)
                    assert prev is None, f"Duplicate block '{node.name}' in {path.name} and {prev}"
                    block_classes[node.name] = path.name
