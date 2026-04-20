import os
from neuro_pilot import NeuroPilot
import neuro_pilot.engine.task  # triggers @Registry.register_task


def run_benchmark():
    """
    Automated Benchmark Script for Trajectory Prediction Losses.
    Compiles results for 5 different SOTA loss configurations.
    """
    # Define the 5 SOTA experiments
    experiments = [
        {
            "name": "baseline_l2",
            "traj_loss_type": "l2",
            "use_fdat": False,
            "desc": "Baseline L2 Loss (MSE) - Basic point regression",
        },
        {
            "name": "standard_smooth_l1",
            "traj_loss_type": "smooth_l1",
            "use_fdat": False,
            "desc": "Standard SmoothL1 Loss - Industrial baseline",
        },
        {
            "name": "sota_wing",
            "traj_loss_type": "wing",
            "use_fdat": False,
            "desc": "SOTA Wing Loss - High-precision landmark regression",
        },
        {
            "name": "fdat_ours",
            "traj_loss_type": "smooth_l1",
            "use_fdat": True,
            "desc": "FDAT Loss (Frenet-Decomposed) - Our core contribution",
        },
        {
            "name": "fdat_wing_combined",
            "traj_loss_type": "wing",
            "use_fdat": True,
            "desc": "FDAT + Wing Loss - Optimal geometry-aware high-precision loss",
        },
    ]

    # Data and Model base configuration
    # Note: Using paths found in examples/train.py
    base_data_path = "/home/quynhthu/Documents/waypoint_data/data.yaml"
    model_cfg = "neuro_pilot/cfg/models/neuralPilot_deformable.yaml"

    for exp in experiments:
        print("\n" + "#" * 60)
        print(f"### STARTING EXPERIMENT: {exp['name']}")
        print(f"### DESCRIPTION: {exp['desc']}")
        print("#" * 60 + "\n")

        # Initialize fresh model for each experiment
        model = NeuroPilot(model_cfg, scale="s")

        # Start training with the specific loss configuration
        model.train(
            data=base_data_path,
            experiment_name=exp["name"],
            # Loss Configuration (The parameters we want to benchmark)
            traj_loss_type=exp["traj_loss_type"],
            use_fdat=exp["use_fdat"],
            # Training Hyperparameters (Inherited from examples/train.py)
            epochs=50,  # Recommended for meaningful benchmark results
            batch=16,
            learning_rate=1e-3,
            patience=20,  # Early stopping to save time
            # Task balancing
            use_uncertainty=True,  # Critical for multi-task stability
            lambda_traj=2.0,
            lambda_det=1.0,
            lambda_heatmap=1.5,
            lambda_gate=0.5,
            lambda_smooth=0.1,
            # Auxiliary representation losses
            lambda_jepa=1.0,
            lambda_sigreg=1.0,
            # Advanced FDAT Parameters (Applied when use_fdat=True)
            fdat_alpha_lane=10.0,
            fdat_beta_lane=1.0,
            fdat_alpha_inter=5.0,
            fdat_beta_inter=3.0,
            fdat_lambda_heading=2.0,
            fdat_lambda_endpoint=5.0,
            # Augmentation (Standard set)
            rotate_deg=5.0,
            translate=0.1,
            scale=0.1,
            blur_prob=0.05,
            # Hardware settings
            use_amp=False,  # Disabled for higher trajectory precision
        )

        print(f"\n[DONE] Finished Experiment: {exp['name']}\n")


if __name__ == "__main__":
    run_benchmark()
