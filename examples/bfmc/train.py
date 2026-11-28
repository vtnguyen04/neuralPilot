"""BFMC multi-task training example using the public `NeuroPilot` framework API."""

from __future__ import annotations

import argparse

from neuro_pilot import NeuroPilot


def main() -> None:
    p = argparse.ArgumentParser(description="Train BFMC multi-task planner via NeuroPilot.")
    p.add_argument(
        "--arch", default="pilot_mhp", choices=["pilot_det", "pilot_mhp", "bfmc", "bfmc_mhp"]
    )
    p.add_argument("--scale", default="v2", choices=["v1", "v2", ""])
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--unlabeled-json", default="")
    p.add_argument("--pilot-s-pt", default="pilot_s.pt")
    p.add_argument("--resume-weights", default="")
    p.add_argument("--dinov3-path", default="")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--img-w", type=int, default=640)
    p.add_argument("--img-h", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-backbone", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=1.2e-3)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--copy-paste", type=float, default=0.50)
    p.add_argument("--affine-prob", type=float, default=0.65)
    p.add_argument("--affine-strength", type=float, default=1.0)
    p.add_argument("--polish-epochs", type=int, default=0)
    p.add_argument("--no-balance-commands", dest="balance_commands", action="store_false")
    p.set_defaults(balance_commands=True)
    p.add_argument("--distill", action="store_true")
    p.add_argument("--distill-weight", type=float, default=1.2)
    p.add_argument("--det-weight", type=float, default=1.8)
    p.add_argument("--det-cls-weight", type=float, default=0.85)
    p.add_argument("--smooth-weight", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.994)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = (
        NeuroPilot(args.resume_weights)
        if args.resume_weights
        else NeuroPilot(args.arch, scale=args.scale)
    )
    model.train(
        train_json=args.train_json,
        val_json=args.val_json,
        unlabeled_json=args.unlabeled_json,
        pilot_s_pt="" if args.resume_weights else args.pilot_s_pt,
        dinov3_path=args.dinov3_path,
        save_dir=args.save_dir,
        img_w=args.img_w,
        img_h=args.img_h,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        freeze_epochs=args.freeze_epochs,
        warmup_epochs=args.warmup_epochs,
        copy_paste=args.copy_paste,
        affine_prob=args.affine_prob,
        affine_strength=args.affine_strength,
        polish_epochs=args.polish_epochs,
        balance_commands=args.balance_commands,
        distill=args.distill,
        distill_weight=args.distill_weight,
        det_weight=args.det_weight,
        det_cls_weight=args.det_cls_weight,
        smooth_weight=args.smooth_weight,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
