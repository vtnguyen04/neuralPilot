"""
Benchmark Script: Trajectory Loss Ablation Study
=================================================
Runs 5 configurations (3 baselines + 2 FDAT variants) on the dual-branch
NeuralPilot model. Results are printed as a formatted table at the end.

Usage:
    python3 scripts/benchmark_losses.py [--epochs 50] [--batch 16]
    python3 scripts/benchmark_losses.py --resume fdat_smooth_l1   # skip earlier exps
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_pilot import NeuroPilot
import neuro_pilot.engine.task  # triggers @Registry.register_task


# ──────────────────────────────────────────────────────────────────────
# Experiment Definitions
# ──────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "baseline_mse",
        "traj_loss_type": "l2",
        "use_fdat": False,
        "desc": "MSE (L2) — basic Cartesian point regression",
    },
    {
        "name": "baseline_smooth_l1",
        "traj_loss_type": "smooth_l1",
        "use_fdat": False,
        "desc": "Smooth-L1 (β=0.1) — industrial standard",
    },
    {
        "name": "baseline_wing",
        "traj_loss_type": "wing",
        "use_fdat": False,
        "desc": "Wing Loss — non-vanishing small-error gradients",
    },
    {
        "name": "fdat_smooth_l1",
        "traj_loss_type": "smooth_l1",
        "use_fdat": True,
        "desc": "FDAT + Smooth-L1 — Frenet-decomposed anisotropic (ours)",
    },
    {
        "name": "fdat_wing",
        "traj_loss_type": "wing",
        "use_fdat": True,
        "desc": "FDAT + Wing — geometry-aware high-precision (ours)",
    },
]

# Training hyperparameters (held constant across all experiments)
# Matches examples/train.py for reproducibility
TRAIN_KWARGS = dict(
    # --- Multi-task loss balancing ---
    lambda_traj=2.0,
    lambda_det=1.0,
    lambda_heatmap=1.5,
    lambda_gate=0.5,
    lambda_smooth=0.1,
    lambda_cls=1.0,
    # --- Detection sub-losses ---
    box=1.0,
    cls_det=1.0,
    dfl=1.0,
    # --- Representation learning ---
    lambda_jepa=1.0,
    lambda_sigreg=1.0,
    # --- FDAT hyperparameters (only active when use_fdat=True) ---
    fdat_alpha_lane=10.0,
    fdat_beta_lane=1.0,
    fdat_alpha_inter=5.0,
    fdat_beta_inter=3.0,
    fdat_lambda_heading=0.5,
    fdat_lambda_endpoint=2.0,
    fdat_tau_start=2.0,
    fdat_tau_end=2.0,
    # --- Optimizer ---
    learning_rate=1e-3,
    use_amp=False,
    use_uncertainty=True,
    warmup_bias_lr=1e-4,
    warmup_epochs=2.0,
    # --- Augmentation (held constant) ---
    rotate_deg=5.0,
    translate=0.1,
    scale=0.1,
    color_jitter=0.1,
    shear=0.0,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    perspective=0.05,
    mosaic=0.0,
    noise_prob=0.0,
    blur_prob=0.05,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Loss Ablation Benchmark")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per experiment")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--data", type=str,
                        default="/home/quynhthu/Documents/waypoint_data/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model_cfg", type=str,
                        default="neuro_pilot/cfg/models/neuralPilot_dual.yaml",
                        help="Model config YAML")
    parser.add_argument("--scale", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"],
                        help="Model scale")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a specific experiment name (skip earlier ones)")
    return parser.parse_args()


def print_header(exp):
    """Print a formatted experiment header."""
    w = 70
    print("\n" + "=" * w)
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  LOSS:       {exp['desc']}")
    print(f"  FDAT:       {'ENABLED' if exp['use_fdat'] else 'DISABLED'}")
    print(f"  BASE:       {exp['traj_loss_type']}")
    print("=" * w + "\n")


def print_results_table(results):
    """Print a formatted comparison table."""
    if not results:
        print("\n[WARNING] No results to display.\n")
        return

    print("\n" + "=" * 90)
    print("  LOSS ABLATION BENCHMARK — FINAL RESULTS")
    print("=" * 90)

    header = f"{'Loss Config':<22} {'FDAT':^6} {'ADE':>8} {'FDE':>8} {'L1':>8} {'wL1':>8} {'Lat':>8} {'Long':>8}"
    print(header)
    print("-" * 90)

    sorted_results = sorted(results, key=lambda r: r.get("ADE", float("inf")))
    best_ade = sorted_results[0].get("ADE", float("inf")) if sorted_results else float("inf")

    for r in sorted_results:
        if "error" in r:
            print(f"{r['loss_type']:<22} {'Yes' if r['use_fdat'] else 'No':^6}  ** FAILED: {r['error'][:40]} **")
            continue

        ade = r.get("ADE", float("nan"))
        fde = r.get("FDE", float("nan"))
        l1 = r.get("L1", float("nan"))
        wl1 = r.get("Weighted_L1", float("nan"))
        lat = r.get("Lateral_Error", float("nan"))
        lon = r.get("Longitudinal_Error", float("nan"))
        fdat_str = "Yes" if r["use_fdat"] else "No"
        marker = " ★" if ade == best_ade else ""

        print(f"{r['loss_type']:<22} {fdat_str:^6} {ade:>8.4f} {fde:>8.4f} {l1:>8.4f} {wl1:>8.4f} {lat:>8.4f} {lon:>8.4f}{marker}")

    print("-" * 90)
    print(f"  ★ = Best ADE")
    print("=" * 90 + "\n")


def run_benchmark():
    args = parse_args()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments") / f"loss_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    skip = args.resume is not None

    for exp in EXPERIMENTS:
        if skip:
            if exp["name"] == args.resume:
                skip = False
            else:
                print(f"[SKIP] {exp['name']} (resuming from {args.resume})")
                continue

        print_header(exp)

        # Fresh model per experiment
        model = NeuroPilot(args.model_cfg, scale=args.scale)

        # Merge constant kwargs with experiment-specific ones
        train_kwargs = {
            **TRAIN_KWARGS,
            "data": args.data,
            "experiment_name": exp["name"],
            "epochs": args.epochs,
            "batch": args.batch,
            "patience": args.patience,
            "traj_loss_type": exp["traj_loss_type"],
            "use_fdat": exp["use_fdat"],
        }

        try:
            metrics = model.train(**train_kwargs)

            result = {
                "name": exp["name"],
                "loss_type": exp["traj_loss_type"],
                "use_fdat": exp["use_fdat"],
                "desc": exp["desc"],
            }

            if isinstance(metrics, dict):
                for key in ["ADE", "FDE", "L1", "Weighted_L1", "Lateral_Error", "Longitudinal_Error"]:
                    result[key] = metrics.get(key, float("nan"))

            all_results.append(result)

            # Save incremental results
            with open(results_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            print(f"\n[DONE] {exp['name']} — ADE: {result.get('ADE', '?'):.4f}, FDE: {result.get('FDE', '?'):.4f}")

        except Exception as e:
            print(f"\n[ERROR] {exp['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": exp["name"],
                "loss_type": exp["traj_loss_type"],
                "use_fdat": exp["use_fdat"],
                "desc": exp["desc"],
                "error": str(e),
            })

    # Final summary
    print_results_table(all_results)

    final_file = results_dir / "results_final.json"
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {final_file}")


if __name__ == "__main__":
    run_benchmark()
