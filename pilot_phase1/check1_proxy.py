# -*- coding: utf-8 -*-
"""
Check 1: Validate the Taylor proxy c_{i,l,h} predictive power.

For each sampled block l (default: block 0, 6, 11 covering shallow/mid/deep),
removes each of the 12 heads one at a time, measures the real Dice drop on the
test set, and computes Spearman ρ between |c_{i,l,h}| ranking and Dice-drop ranking.

Decision thresholds (from phase1_next.md §6 Check 1):
    ρ > 0.6   : strong proxy → proceed with confidence
    0.3 < ρ ≤ 0.6 : weak proxy → proceed but note in paper
    ρ ≤ 0.3   : proxy insufficient → resolve before full experiments

Usage:
    python -m pilot_phase1.check1_proxy \\
        --data_root assert/CVC-ColonDB \\
        --checkpoint work_dir/MedSAM/medsam_vit_b.pth \\
        --device cuda:0 \\
        --n_calibration 128 \\
        --check_blocks 0 6 11
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.head_pruning import (
    compute_head_gradient_projections_fast,
    apply_head_mask_to_model,
    remove_hooks,
)
from pilot_phase1.evaluate import evaluate_pruned_model


def main():
    parser = argparse.ArgumentParser(description="Check 1: Proxy fidelity (Spearman ρ)")
    parser.add_argument("--data_root",    type=str, default="assert/CVC-ColonDB")
    parser.add_argument("--checkpoint",   type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--device",       type=str, default="cuda:0")
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--output_dir",   type=str, default="results/phase1_v2")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--check_blocks", type=int, nargs="+", default=[0, 6, 11],
                        help="Block indices to validate (default: 0 6 11)")
    parser.add_argument("--num_heads",    type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("Check 1: Taylor proxy fidelity (Spearman ρ)")
    print(f"  Blocks: {args.check_blocks}")
    print("=" * 60)

    # Load model
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    # Data
    cal_loader, test_loader, _, _, cal_freq_weights = build_dataloaders(
        args.data_root, n_calibration=args.n_calibration,
        batch_size=args.batch_size, seed=args.seed,
    )

    # Baseline Dice
    baseline = evaluate_pruned_model(model, test_loader, head_mask=None, device=device)
    baseline_dice = baseline["avg_metrics"]["mean_dice"]
    print(f"\nBaseline Dice: {baseline_dice:.4f}")

    # Compute Taylor proxy
    print("\nComputing Taylor proxy c_{i,l,h} ...")
    _, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device, freq_weights=cal_freq_weights
    )

    # Per-head mean |c_{i,l,h}| (proxy importance)
    proxy_importance = np.abs(per_sample_proj).mean(axis=0)  # (144,)

    num_blocks = 12
    num_heads  = args.num_heads

    results = {}

    for blk_idx in args.check_blocks:
        print(f"\n--- Block {blk_idx} ---")

        dice_drops = []
        proxy_vals = []

        for h in range(num_heads):
            # Build single-head-removal mask
            head_mask = np.ones(num_blocks * num_heads, dtype=np.float32)
            head_mask[blk_idx * num_heads + h] = 0.0

            hooks  = apply_head_mask_to_model(model, head_mask)
            result = evaluate_pruned_model(model, test_loader, head_mask=None, device=device)
            remove_hooks(hooks)

            dice_h = result["avg_metrics"]["mean_dice"]
            drop_h = baseline_dice - dice_h
            prx_h  = proxy_importance[blk_idx * num_heads + h]

            dice_drops.append(drop_h)
            proxy_vals.append(prx_h)

            print(f"  Head {h:>2}: Dice drop = {drop_h:+.4f},  |c| = {prx_h:.4f}")

        dice_drops = np.array(dice_drops)
        proxy_vals = np.array(proxy_vals)

        rho, pval = spearmanr(proxy_vals, dice_drops)
        print(f"\n  Spearman ρ = {rho:.3f}  (p = {pval:.4f})")

        if rho > 0.6:
            verdict = "STRONG proxy (ρ > 0.6) → proceed"
        elif rho > 0.3:
            verdict = "WEAK proxy (0.3 < ρ ≤ 0.6) → proceed with caveat"
        else:
            verdict = "INSUFFICIENT proxy (ρ ≤ 0.3) → investigate before full experiments"
        print(f"  Verdict: {verdict}")

        results[f"block_{blk_idx}"] = {
            "spearman_rho":   float(rho),
            "spearman_pval":  float(pval),
            "verdict":        verdict,
            "dice_drops":     dice_drops.tolist(),
            "proxy_vals":     proxy_vals.tolist(),
        }

    # Summary
    rhos = [results[f"block_{b}"]["spearman_rho"] for b in args.check_blocks]
    print(f"\n{'='*60}")
    print(f"Check 1 Summary:  mean ρ = {np.mean(rhos):.3f}  "
          f"(blocks {args.check_blocks})")
    if np.mean(rhos) > 0.6:
        print("  Overall: STRONG → V2 proxy is substantially better than parameter proxy")
    elif np.mean(rhos) > 0.3:
        print("  Overall: WEAK → proxy has limited but non-trivial predictive value")
    else:
        print("  Overall: INSUFFICIENT → consider output-space regularisation or "
              "multi-token representation before proceeding")

    out_path = os.path.join(args.output_dir, "check1_proxy_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
