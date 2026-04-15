import os
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot

def main():
    print("🚀 Initializing NeuroPilot Temporal Training Pipeline...")

    # Initialize the model with the temporal video configuration
    model = NeuroPilot("neuralPilot_video.yaml", scale="n")

    # Path to the dataset configuration previously generated
    dataset_yaml_path = Path(__file__).parent / "covla_sample_dataset.yaml"

    model.train(
        data=str(dataset_yaml_path),

        # ---------------------------------------------------------
        # 1. CORE TRAINER SETTINGS
        # ---------------------------------------------------------
        epochs=1,
        batch=2,                      # Batch size
        imgsz=320,                    # Input resolution
        device="cuda:0",              # Use 'cuda' or 'cuda:0' for GPU training
        workers=0,                    # Dataloader workers (0 for debugging)
        learning_rate=1e-3,           # Initial Learning Rate
        lr_final=0.01,                # Final Learning Rate ratio (cosine annealing)
        optimizer="AdamW",            # Optimizer used
        momentum=0.937,               # Momentum / Beta1
        weight_decay=0.0005,          # Weight decay regularization
        warmup_epochs=3.0,            # Epochs to perform LR warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        resume=False,                 # Auto-resume from latest checkpoint if True

        # Advanced hardware/trainer flags
        use_amp=True,                 # Automatic Mixed Precision (FP16)
        use_ema=True,                 # Exponential Moving Average of weights
        ema_decay=0.999,
        lr_schedule="cosine",         # LR Scheduler style ("cosine" or "linear")
        gradient_accumulation_steps=1,
        grad_clip_norm=10.0,
        checkpoint_top_k=3,           # Keep top N checkpoints
        early_stop_patience=100,      # Stop if val loss doesn't improve for N epochs
        experiment_name="temporal_test_run",
        verbose=True,

        # ---------------------------------------------------------
        # 2. TEMPORAL PIPELINE CONFIGURATION
        # ---------------------------------------------------------
        temporal_jitter=0.2,          # Random temporal noise adding
        temporal_dropout=0.1,         # Frame dropout probability
        max_cached_frames=8,          # Max sequence allowed in VRAM
        use_flow=False,               # Optical Flow integration flag

        # ---------------------------------------------------------
        # 3. TASK LOSS WEIGHTS (Trajectory Only configuration)
        # ---------------------------------------------------------
        lambda_traj=2.0,              # Core Trajectory Loss
        lambda_smooth=0.1,            # Enforce smooth path predictions
        lambda_velocity=0.0,
        lambda_collision=0.0,
        lambda_progress=0.0,

        # Idle Tasks (Explicitly disabled to clean TQDM progress logging)
        lambda_det=0.0,
        lambda_heatmap=0.0,
        box=0.0,
        cls_det=0.0,
        dfl=0.0,

        # ---------------------------------------------------------
        # 4. FDAT / TRAJECTORY ADVANCED PARAMS
        # ---------------------------------------------------------
        use_fdat=True,                # Enable Frequency Domain Alignment
        use_uncertainty=True,         # Auto-weighting between multiple tasks
        fdat_alpha_lane=10.0,
        fdat_beta_lane=1.0,
        fdat_alpha_inter=5.0,
        fdat_beta_inter=3.0,
        fdat_lambda_heading=2.0,
        fdat_lambda_endpoint=5.0,
        fdat_tau_start=2.0,           # Dynamic bathtub weighting early sequence
        fdat_tau_end=2.0,             # Dynamic bathtub weighting late sequence

        # ---------------------------------------------------------
        # 5. WORLD MODEL & AUXILIARY METRICS
        # ---------------------------------------------------------
        lambda_jepa=0.0,
        lambda_sigreg=0.1,
        lambda_temporal_consistency=0.5,
        lambda_motion_prior=0.1,
        rotate_deg=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        color_jitter=0.2,             # Color/Brightness noise
        hsv_h=0.015,                  # HSV Hue Shift
        hsv_s=0.4,                    # HSV Saturation Shift
        hsv_v=0.4,                    # HSV Value shift
        noise_prob=0.1,               # Random Gaussian noise probability
        blur_prob=0.1,                # Random Box/Gaussian Blur probability
        mosaic=0.0,                   # MOSAIC DOES NOT WORK WELL WITH RAW TRAJECTORY JSONL!
        mixup=0.0,
        copy_paste=0.0
    )

if __name__ == "__main__":
    main()
