"""
Quick sanity-check for the FDATLoss fix.
Verifies that:
  1. Lateral (cross-track) error is penalized MORE than longitudinal (along-track)
  2. The anisotropic ratio is preserved (no over-normalization)
  3. Gate gradient flows through (no .detach())
"""

import torch
import torch.nn as nn
from neuro_pilot.utils.losses import FDATLoss, WingLoss

torch.manual_seed(42)

# ── 1. Build a curved GT trajectory ──
T = 10
t = torch.linspace(0, 1, T)
gt_x = 0.5 + 0.1 * torch.sin(t * 3.14 / 2)  # curves left
gt_y = 0.8 - 0.6 * t                         # moves forward
gt_wp = torch.stack([gt_x, gt_y], dim=-1).unsqueeze(0)  # [1, 10, 2]

# ── 2. Baselines ──
sl1 = nn.SmoothL1Loss(reduction="none", beta=0.1)
wing = WingLoss(w=0.05, epsilon=0.01, reduction="none")
fdat_sl1 = FDATLoss(base_loss_type="smooth_l1")
fdat_wing = FDATLoss(base_loss_type="wing")

print("=" * 65)
print("  FDAT LOSS SANITY CHECK (Post-Fix)")
print("=" * 65)

# ── 3. Test: Pure lateral offset ──
pred_lat = gt_wp.clone()
pred_lat[0, :, 0] += 0.05  # 0.05 lateral shift

loss_sl1_lat = sl1(pred_lat, gt_wp).mean().item()
loss_wing_lat = wing(pred_lat, gt_wp).mean().item()
gate = torch.tensor([[[0.0]]])  # lane-following mode
loss_fdat_sl1_lat = fdat_sl1(pred_lat, gt_wp, gate).mean().item()
loss_fdat_wing_lat = fdat_wing(pred_lat, gt_wp, gate).mean().item()

print(f"\n[Test A] Lateral Offset = 0.05 (cross-track)")
print(f"  Smooth-L1:       {loss_sl1_lat:.6f}")
print(f"  Wing:            {loss_wing_lat:.6f}")
print(f"  FDAT+Smooth-L1:  {loss_fdat_sl1_lat:.6f}")
print(f"  FDAT+Wing:       {loss_fdat_wing_lat:.6f}")

# ── 4. Test: Pure longitudinal offset ──
pred_lon = gt_wp.clone()
pred_lon[0, :, 1] += 0.05  # 0.05 longitudinal shift

loss_sl1_lon = sl1(pred_lon, gt_wp).mean().item()
loss_wing_lon = wing(pred_lon, gt_wp).mean().item()
loss_fdat_sl1_lon = fdat_sl1(pred_lon, gt_wp, gate).mean().item()
loss_fdat_wing_lon = fdat_wing(pred_lon, gt_wp, gate).mean().item()

print(f"\n[Test B] Longitudinal Offset = 0.05 (along-track)")
print(f"  Smooth-L1:       {loss_sl1_lon:.6f}")
print(f"  Wing:            {loss_wing_lon:.6f}")
print(f"  FDAT+Smooth-L1:  {loss_fdat_sl1_lon:.6f}")
print(f"  FDAT+Wing:       {loss_fdat_wing_lon:.6f}")

# ── 5. Verify anisotropy ratio ──
ratio_fdat_sl1 = loss_fdat_sl1_lat / (loss_fdat_sl1_lon + 1e-8)
ratio_fdat_wing = loss_fdat_wing_lat / (loss_fdat_wing_lon + 1e-8)
ratio_baseline = loss_wing_lat / (loss_wing_lon + 1e-8)

print(f"\n[Anisotropy Ratio] lat/lon (higher = more cross-track penalty)")
print(f"  Baseline Wing:   {ratio_baseline:.2f}x (should be ~1.0 = isotropic)")
print(f"  FDAT+Smooth-L1:  {ratio_fdat_sl1:.2f}x (should be >>1.0)")
print(f"  FDAT+Wing:       {ratio_fdat_wing:.2f}x (should be >>1.0)")

# ── 6. Verify gate gradient flows ──
gate_param = torch.tensor([[[0.3]]], requires_grad=True)
pred_test = (gt_wp.clone().detach() + 0.05).requires_grad_(True)  # Non-zero error needed for gradient
loss = fdat_wing(pred_test, gt_wp.detach(), gate_param)
loss.mean().backward()

gate_grad_ok = gate_param.grad is not None and gate_param.grad.abs().sum() > 0
wp_grad_ok = pred_test.grad is not None and pred_test.grad.abs().sum() > 0

print(f"\n[Gradient Flow]")
print(f"  Gate gradient:     {'✅ PASS' if gate_grad_ok else '❌ FAIL (gate.detach still present!)'}")
print(f"  Waypoint gradient: {'✅ PASS' if wp_grad_ok else '❌ FAIL'}")

# ── 7. Summary ──
all_pass = (
    ratio_fdat_sl1 > 2.0
    and ratio_fdat_wing > 2.0
    and gate_grad_ok
    and wp_grad_ok
)

print(f"\n{'=' * 65}")
if all_pass:
    print("  ✅ ALL CHECKS PASSED — FDAT is correctly anisotropic with gradient flow")
else:
    print("  ❌ SOME CHECKS FAILED — Review the output above")
print(f"{'=' * 65}\n")
