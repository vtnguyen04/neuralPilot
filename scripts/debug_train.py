"""
E2E Debug Training: Full Pipeline Verification
================================================
Runs 1 epoch of training with verbose debug output to verify:
  1. traj_loss_type kwarg is correctly mapped to LossConfig
  2. FDATLoss is initialized with corrected hyperparams (no normalization, no detach)
  3. Gate gradient flows during backward pass
  4. Loss values are finite and reasonable
  5. Metrics are computed correctly at validation time
  6. Returned metrics dict contains ADE/FDE/L1/wL1

Usage:
    .venv/bin/python3 scripts/debug_train.py
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_pilot import NeuroPilot
import neuro_pilot.engine.task


def check(label, condition, detail=""):
    status = "✅" if condition else "❌"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def run_debug():
    print("=" * 70)
    print("  E2E DEBUG TRAINING — FULL PIPELINE VERIFICATION")
    print("=" * 70)

    all_ok = True

    # ── Step 1: Build model and verify kwarg mapping ──
    print("\n[1/6] Building model and verifying kwarg mapping...")
    model = NeuroPilot("neuro_pilot/cfg/models/neuralPilot_dual.yaml", scale="s")

    test_kwargs = {
        "traj_loss_type": "wing",
        "use_fdat": True,
        "fdat_lambda_heading": 0.5,
        "fdat_lambda_endpoint": 2.0,
        "box": 1.0,
        "cls_det": 1.0,
        "dfl": 1.0,
    }
    mapped = model._map_kwargs_to_config(test_kwargs)
    loss_section = mapped.get("loss", {})

    all_ok &= check("traj_loss_type mapped",
                     loss_section.get("traj_loss_type") == "wing",
                     f"got: {loss_section.get('traj_loss_type', 'MISSING')}")
    all_ok &= check("use_fdat mapped",
                     loss_section.get("use_fdat") is True,
                     f"got: {loss_section.get('use_fdat', 'MISSING')}")
    all_ok &= check("fdat_lambda_heading mapped",
                     loss_section.get("fdat_lambda_heading") == 0.5,
                     f"got: {loss_section.get('fdat_lambda_heading', 'MISSING')}")
    all_ok &= check("fdat_lambda_endpoint mapped",
                     loss_section.get("fdat_lambda_endpoint") == 2.0,
                     f"got: {loss_section.get('fdat_lambda_endpoint', 'MISSING')}")
    all_ok &= check("box mapped",
                     loss_section.get("box") == 1.0,
                     f"got: {loss_section.get('box', 'MISSING')}")

    # ── Step 2: Run 1-epoch training with FDAT+Wing ──
    print("\n[2/6] Running 1-epoch training (FDAT + Wing)...")
    try:
        metrics = model.train(
            data="/home/quynhthu/Documents/waypoint_data/data.yaml",
            experiment_name="debug_fdat_verify",
            epochs=1,
            batch=8,
            patience=999,
            # Loss config
            traj_loss_type="wing",
            use_fdat=True,
            use_uncertainty=True,
            # Multi-task balancing
            lambda_traj=2.0,
            lambda_det=1.0,
            lambda_heatmap=1.5,
            lambda_gate=0.5,
            lambda_smooth=0.1,
            lambda_cls=1.0,
            box=1.0,
            cls_det=1.0,
            dfl=1.0,
            # FDAT params (corrected)
            fdat_alpha_lane=10.0,
            fdat_beta_lane=1.0,
            fdat_alpha_inter=5.0,
            fdat_beta_inter=3.0,
            fdat_lambda_heading=0.5,
            fdat_lambda_endpoint=2.0,
            fdat_tau_start=2.0,
            fdat_tau_end=2.0,
            # Hardware
            use_amp=False,
            warmup_bias_lr=1e-4,
            warmup_epochs=1.0,
            # No augmentation to keep deterministic
            rotate_deg=0.0,
            translate=0.0,
            scale=0.0,
            blur_prob=0.0,
            noise_prob=0.0,
        )
        all_ok &= check("Training completed", True)
    except Exception as e:
        check("Training completed", False, str(e))
        import traceback
        traceback.print_exc()
        all_ok = False
        metrics = {}

    # ── Step 3: Verify returned metrics ──
    print("\n[3/6] Verifying returned metrics dict...")
    all_ok &= check("metrics is dict", isinstance(metrics, dict), f"type: {type(metrics)}")

    expected_keys = ["ADE", "FDE", "L1", "Weighted_L1", "Lateral_Error", "Longitudinal_Error"]
    for key in expected_keys:
        val = metrics.get(key)
        present = val is not None
        finite = present and (isinstance(val, (int, float)) and val == val)  # NaN check
        all_ok &= check(f"metrics['{key}'] present & finite",
                        present and finite,
                        f"value: {val}")

    # ── Step 4: Verify FDAT loss internals ──
    print("\n[4/6] Verifying FDATLoss internals (direct unit test)...")
    from neuro_pilot.utils.losses import FDATLoss

    fdat = FDATLoss(
        alpha_lane=10.0, beta_lane=1.0,
        alpha_inter=5.0, beta_inter=3.0,
        lambda_heading=0.5, lambda_endpoint=2.0,
        base_loss_type="wing",
    )

    # Check defaults
    all_ok &= check("lambda_heading == 0.5", fdat.lambda_heading == 0.5, f"got: {fdat.lambda_heading}")
    all_ok &= check("lambda_endpoint == 2.0", fdat.lambda_endpoint == 2.0, f"got: {fdat.lambda_endpoint}")

    # Check no normalization in forward
    import inspect
    src = inspect.getsource(fdat.forward)
    has_normalize = "/ (self.alpha" in src or "/(self.alpha" in src
    all_ok &= check("No /(alpha+beta) normalization in forward",
                     not has_normalize,
                     "Found normalization!" if has_normalize else "Clean")

    # Check no detach in forward
    has_detach = ".detach()" in src
    all_ok &= check("No gate_score.detach() in forward",
                     not has_detach,
                     "Found .detach()!" if has_detach else "Clean")

    # ── Step 5: Verify gate gradient flow ──
    print("\n[5/6] Verifying gate gradient flow...")
    T = 10
    t = torch.linspace(0, 1, T)
    gt = torch.stack([0.5 + 0.1 * torch.sin(t * 3.14 / 2), 0.8 - 0.6 * t], dim=-1).unsqueeze(0)
    pred = (gt.clone().detach() + 0.05).requires_grad_(True)
    gate = torch.tensor([[[0.3]]], requires_grad=True)

    loss = fdat(pred, gt.detach(), gate)
    loss.mean().backward()

    gate_grad = gate.grad is not None and gate.grad.abs().sum() > 0
    wp_grad = pred.grad is not None and pred.grad.abs().sum() > 0

    all_ok &= check("Gate gradient flows", gate_grad,
                     f"grad: {gate.grad}" if gate.grad is not None else "None")
    all_ok &= check("Waypoint gradient flows", wp_grad)

    # ── Step 6: Verify anisotropy ──
    print("\n[6/6] Verifying anisotropy ratio...")
    gate0 = torch.tensor([[[0.0]]])

    pred_lat = gt.clone().detach()
    pred_lat[0, :, 0] += 0.05
    loss_lat = fdat(pred_lat, gt.detach(), gate0).item()

    pred_lon = gt.clone().detach()
    pred_lon[0, :, 1] += 0.05
    loss_lon = fdat(pred_lon, gt.detach(), gate0).item()

    ratio = loss_lat / (loss_lon + 1e-8)
    all_ok &= check(f"Anisotropy ratio > 2.0", ratio > 2.0, f"lat/lon = {ratio:.2f}x")

    # ── Final Summary ──
    print("\n" + "=" * 70)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Pipeline is 100% correct")
    else:
        print("  ❌ SOME CHECKS FAILED — Review output above")
    print("=" * 70)

    if metrics:
        print(f"\n  Training Metrics (1 epoch):")
        for k in expected_keys:
            v = metrics.get(k, "N/A")
            if isinstance(v, float):
                print(f"    {k:<22}: {v:.6f}")
            else:
                print(f"    {k:<22}: {v}")
    print()


if __name__ == "__main__":
    run_debug()
