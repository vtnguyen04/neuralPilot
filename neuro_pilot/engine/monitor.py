"""Unified run monitor for CSV/JSONL metrics, GPU telemetry, console tables, and curve plots."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from neuro_pilot.utils.logger import logger


def gpu_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return peak allocated CUDA memory in MiB, or 0.0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    if device is not None and device.type != "cuda":
        return 0.0
    return round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 1)


class RunMonitor:
    """Task-agnostic training monitor handling CSV, JSONL, GPU stats, ETA, and plots."""

    def __init__(self, save_dir: str | Path, task_name: str = "planner") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = str(task_name)
        self.csv_path = self.save_dir / "metrics.csv"
        self.jsonl_path = self.save_dir / "history.jsonl"
        self.history: List[Dict[str, Any]] = []
        self._csv_headers: Optional[List[str]] = None
        self._t0 = time.perf_counter()

    def start(self, total_epochs: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """Reset run clock and record run header metadata."""
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if meta:
            (self.save_dir / "run_config.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )
        logger.info(
            f"({self.task_name}) Run started | epochs={total_epochs} | save_dir={self.save_dir}"
        )

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, Any]] = None,
        lr: float = 0.0,
        samples: int = 0,
        epoch_sec: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Record one epoch to JSONL, CSV, and structured console logs."""
        elapsed = time.perf_counter() - self._t0
        eta = (elapsed / max(1, epoch)) * max(0, total_epochs - epoch)
        vram = gpu_memory_mb(device)
        ips = round(samples / max(epoch_sec, 1e-3), 1) if samples > 0 else 0.0

        record: Dict[str, Any] = {
            "epoch": int(epoch),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
            "epoch_s": round(epoch_sec, 1),
            "img_per_s": ips,
            "vram_mb": vram,
            "lr": float(lr),
            **{f"train/{k}": round(float(v), 5) for k, v in train_losses.items()},
        }
        if val_metrics:
            record.update(val_metrics)
        self.history.append(record)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._append_csv(record)
        self._log_console(epoch, total_epochs, train_losses, val_metrics, elapsed, eta, vram, ips)
        return record

    def _append_csv(self, record: Dict[str, Any]) -> None:
        flat = {k: v for k, v in record.items() if isinstance(v, (int, float, str, bool))}
        if self._csv_headers is None:
            self._csv_headers = list(flat.keys())
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_headers, extrasaction="ignore")
                w.writeheader()
                w.writerow(flat)
            return
        new_keys = [k for k in flat if k not in self._csv_headers]
        if new_keys:
            self._csv_headers.extend(new_keys)
            rows = [
                {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool))}
                for r in self.history
            ]
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_headers, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            return
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_headers, extrasaction="ignore")
            w.writerow(flat)

    def _log_console(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, Any]],
        elapsed: float,
        eta: float,
        vram: float,
        ips: float,
    ) -> None:
        loss_str = " ".join(f"{k}={v:.3f}" for k, v in train_losses.items())
        head = (
            f"({self.task_name}) Ep {epoch:02d}/{total_epochs} "
            f"({elapsed:5.0f}s, ETA {eta:4.0f}s, {ips:.0f} img/s, {vram:.0f}MB) | {loss_str}"
        )
        if not val_metrics:
            logger.info(head)
            return
        parts = []
        for k, fmt in (
            ("score", "Score={:.2f}"),
            ("mse", "MSE={:.5f}"),
            ("ade_px640", "ADE={:.1f}px"),
            ("lat_px640", "Lat={:.1f}px"),
            ("end_kink_deg", "Kink={:.1f}d"),
            ("map50", "mAP50={:.4f}"),
        ):
            if k in val_metrics:
                parts.append(fmt.format(float(val_metrics[k])))
        logger.info(f"{head} | {' '.join(parts)}" if parts else head)
        cmd_m = val_metrics.get("cmd_metrics") or {}
        if cmd_m:
            c_str = " | ".join(
                f"{k.replace('FOLLOW_LANE', 'FL').replace('TURN_', 'T_').replace('STRAIGHT', 'ST')}"
                f"={v['ade_px']:.1f}/{v['entry25_px']:.1f}px({v['kink_deg']:.1f}d)"
                for k, v in cmd_m.items()
            )
            logger.info(f"  Cmd ADE/Ent25(Kink): {c_str}")

    def finish(self, best_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Persist summary JSON and plot training curves if matplotlib is available."""
        (self.save_dir / "history.json").write_text(
            json.dumps(self.history, indent=2, default=str), encoding="utf-8"
        )
        if best_metrics:
            (self.save_dir / "best_metrics.json").write_text(
                json.dumps(best_metrics, indent=2, default=str), encoding="utf-8"
            )
        self.plot_curves()

    def plot_curves(self) -> Optional[Path]:
        """Render loss and validation curves into `save_dir/training_curves.png`."""
        if len(self.history) < 2:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        specs = [
            ("Composite Score", ["score"], "#0f766e"),
            ("Trajectory MSE", ["mse", "min_mse5"], "#2563eb"),
            ("Detection mAP", ["map50", "map50_95"], "#d97706"),
            ("Train Losses", [k for k in self.history[0] if k.startswith("train/")], "#be123c"),
        ]
        active = [
            (t, [k for k in ks if any(k in r for r in self.history)], c) for t, ks, c in specs
        ]
        active = [(t, ks, c) for t, ks, c in active if ks]
        if not active:
            return None
        fig, axes = plt.subplots(1, len(active), figsize=(4.2 * len(active), 3.6), dpi=130)
        if len(active) == 1:
            axes = [axes]
        for ax, (title, keys, _) in zip(axes, active):
            for key in keys:
                xs = [r["epoch"] for r in self.history if key in r]
                ys = [float(r[key]) for r in self.history if key in r]
                ax.plot(xs, ys, "-o", ms=3, lw=1.6, label=key.replace("train/", ""))
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        plt.tight_layout()
        out = self.save_dir / "training_curves.png"
        plt.savefig(out, dpi=130)
        plt.close(fig)
        return out
