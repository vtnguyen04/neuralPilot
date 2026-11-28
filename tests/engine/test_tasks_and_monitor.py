"""Comprehensive unit and E2E tests for tasks, RunMonitor, visualize, and Planner API."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from neuro_pilot import NeuroPilot
from neuro_pilot.core.registry import Registry
from neuro_pilot.data.datasets import DATASET_REGISTRY, PlannerDataset
from neuro_pilot.engine.monitor import RunMonitor, gpu_memory_mb
from neuro_pilot.nn.zoo import ZOO_YAML_DIR, is_zoo_yaml
from neuro_pilot.tasks import infer_driving_task, register_driving_tasks
from neuro_pilot.utils.visualize import render_image_grid, visualize_prediction

ROOT = Path(__file__).resolve().parents[2]


def test_is_zoo_yaml_strictly_separates_native_and_zoo() -> None:
    """Native YOLO YAMLs (sharing Concat/Conv names) must never route to GraphModel."""
    for native_name in ("neuralPilot.yaml", "neuralPilot_dual.yaml"):
        path = ROOT / "neuro_pilot" / "cfg" / "models" / native_name
        if path.is_file():
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert is_zoo_yaml(cfg) is False, f"{native_name} misclassified as Zoo YAML"

    for zoo_yaml in sorted(ZOO_YAML_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(zoo_yaml.read_text(encoding="utf-8"))
        assert is_zoo_yaml(cfg) is True, f"{zoo_yaml.name} not recognized as Zoo YAML"


def test_all_tasks_registered_and_inferred_cleanly() -> None:
    """Every canonical task family and model-key inference mapping must resolve."""
    register_driving_tasks()
    expected_tasks = (
        "multitask",
        "detection",
        "detect",
        "trajectory",
        "heatmap",
        "planner",
        "e2e_planning",
        "bfmc",
        "seg",
        "depth",
        "lane",
        "cipo",
        "drive",
        "world",
        "world_model",
        "driver",
        "dmonitoring",
    )
    for name in expected_tasks:
        assert Registry.get_task(name) is not None

    mapping = {
        "pilot_s": "planner",
        "pilot_l": "planner",
        "pilot_nav": "planner",
        "pilot_det": "planner",
        "pilot_mhp": "planner",
        "seg": "seg",
        "seg_l": "seg",
        "depth": "depth",
        "lane": "lane",
        "cipo": "cipo",
        "drive": "cipo",
        "driver": "driver",
        "jepa": "world",
        "vq_enc": "world",
        "vq_dec": "world",
        "ae": "world",
        "dit": "world",
        "gpt": "world",
    }
    for key, task_name in mapping.items():
        assert infer_driving_task(key) == task_name


def test_run_monitor_dynamic_csv_jsonl_and_curves(tmp_path: Path) -> None:
    """RunMonitor writes run_config, dynamic CSV headers, JSONL, summary JSON, and curves."""
    run_dir = tmp_path / "exp_monitor"
    mon = RunMonitor(run_dir, task_name="planner")
    mon.start(total_epochs=3, meta={"model_key": "pilot_mhp", "epochs": 3})
    assert (run_dir / "run_config.json").is_file()
    assert gpu_memory_mb() >= 0.0

    # Ep 1 has train-only metrics; Ep 2 & 3 add validation keys -> CSV must rewrite headers cleanly
    mon.log_epoch(1, 3, {"det": 2.1, "traj": 3.4}, val_metrics=None, lr=1e-3, samples=16, epoch_sec=1.0)
    mon.log_epoch(
        2, 3, {"det": 1.4, "traj": 2.0},
        val_metrics={"score": 58.2, "mse": 0.005, "map50": 0.55, "ade_px640": 32.0},
        lr=8e-4, samples=16, epoch_sec=0.9,
    )
    mon.log_epoch(
        3, 3, {"det": 0.9, "traj": 1.3},
        val_metrics={"score": 66.5, "mse": 0.0022, "map50": 0.68, "ade_px640": 24.5},
        lr=5e-4, samples=16, epoch_sec=0.8,
    )
    mon.finish({"epoch": 3, "score": 66.5})

    csv_lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 4 and "score" in csv_lines[0] and "train/det" in csv_lines[0]
    jsonl_lines = (run_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 3
    assert json.loads(jsonl_lines[-1])["score"] == pytest.approx(66.5)
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "best_metrics.json").is_file()
    assert (run_dir / "training_curves.png").is_file()


def test_task_visualizers_and_corridor_no_flip_kink() -> None:
    """Every task visualizer renders valid uint8 BGR frames, incl. sharp >90deg turns."""
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    # Sharp >90 deg left-turn curve where dy crosses zero (regression check for ribbon twist)
    theta = np.linspace(0.0, 0.72 * np.pi, 33, dtype=np.float32)
    wps = np.column_stack([0.65 - 0.45 * (1.0 - np.cos(theta)), 0.92 - 0.50 * np.sin(theta)])
    dets = [(0.15, 0.25, 0.35, 0.55, 0.92, 0), (0.60, 0.30, 0.78, 0.50, 0.85, 2)]

    vis_plan = visualize_prediction(
        "planner", frame, {"waypoints": wps, "detections": dets}, command=1, gt_waypoints=wps
    )
    assert vis_plan.shape == frame.shape and vis_plan.dtype == np.uint8
    assert not np.array_equal(vis_plan, frame)

    vis_seg = visualize_prediction("seg", frame, {"class_map": np.ones((60, 80), dtype=np.int64)})
    vis_dep = visualize_prediction("depth", frame, {"depth": np.linspace(0, 10, 64).reshape(8, 8)})
    vis_lane = visualize_prediction("lane", frame, {"points": [(400, 480), (500, 320), (520, 200)]})
    vis_cipo = visualize_prediction("cipo", frame, {"boxes": [], "distance_m": 12.4, "curvature": 0.01})
    vis_drv = visualize_prediction("driver", frame, {"outputs": np.zeros((1, 84), dtype=np.float32)})
    vis_wld = visualize_prediction("world", frame, {"image_bgr": frame})

    for img in (vis_seg, vis_dep, vis_lane, vis_cipo, vis_drv):
        assert img.shape == frame.shape and img.dtype == np.uint8
    assert vis_wld.shape == (frame.shape[0], frame.shape[1] * 2, 3)
    grid = render_image_grid([vis_plan, vis_seg, vis_dep, vis_lane], cols=2)
    assert grid.shape == (frame.shape[0] * 2, frame.shape[1] * 2, 3)


def test_neuropilot_planner_train_val_predict_visualize_e2e(tmp_path: Path) -> None:
    """End-to-end 1-epoch CPU smoke test of NeuroPilot train -> val -> predict -> visualize."""
    assert "planner_json" in DATASET_REGISTRY
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    records = []
    for i in range(4):
        img = np.full((256, 512, 3), 60 + i * 20, dtype=np.uint8)
        img_path = img_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        ys = np.linspace(0.92, 0.35, 10)
        xs = np.linspace(0.50, 0.50 + 0.05 * (i - 1.5), 10)
        records.append({
            "id": i,
            "session": "unit_test",
            "img_path": str(img_path),
            "command": i % 4,
            "waypoints": [[float(x), float(y)] for x, y in zip(xs, ys)],
            "bboxes": [[0, 0.5, 0.4, 0.12, 0.14]] if i % 2 == 0 else [],
        })
    manifest = tmp_path / "mini.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")

    ds = PlannerDataset(manifest, img_w=512, img_h=256, augment=True)
    assert len(ds) == 4 and ds[0]["yuv6"].shape == (6, 128, 256)

    save_dir = tmp_path / "run_smoke"
    model = NeuroPilot("pilot_mhp", scale="v1", device="cpu")
    best_metrics = model.train(
        train_json=str(manifest),
        val_json=str(manifest),
        pilot_s_pt="",
        save_dir=str(save_dir),
        img_w=512,
        img_h=256,
        epochs=1,
        batch_size=2,
        num_workers=0,
        freeze_epochs=0,
        eval_every=1,
        balance_commands=False,
        copy_paste=0.0,
        affine_prob=0.0,
    )
    assert "score" in best_metrics and "mse" in best_metrics
    assert (save_dir / "best.pt").is_file()
    assert (save_dir / "metrics.csv").is_file()
    assert (save_dir / "history.jsonl").is_file()
    assert (save_dir / "val_grid.jpg").is_file()

    val_res = model.val(val_json=str(manifest), batch_size=2)
    assert "score" in val_res and "map50" in val_res

    sample_bgr = cv2.imread(str(img_dir / "frame_0.jpg"))
    pred = model.predict(sample_bgr, command=0, conf=0.25)
    assert pred["waypoints"].shape == (10, 2)
    for box in pred["detections"]:
        assert 0.0 <= box[0] <= box[2] <= 1.0 and 0.0 <= box[1] <= box[3] <= 1.0
    vis = model.visualize(sample_bgr, pred, command=0)
    assert vis.shape == sample_bgr.shape and vis.dtype == np.uint8

    # Verify non-default scale ("v2") preserves scale metadata through save() -> reload()
    m_v2 = NeuroPilot("pilot_mhp", scale="v2", device="cpu")
    assert m_v2.model.scale == "v2"
    v2_pt = tmp_path / "mhp_v2_roundtrip.pt"
    m_v2.save(v2_pt)
    reloaded_v2 = NeuroPilot(v2_pt, device="cpu")
    assert reloaded_v2.model.scale == "v2"
    assert reloaded_v2.predict(sample_bgr, command=0)["waypoints"].shape == (33, 2)
