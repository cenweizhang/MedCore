# -*- coding: utf-8 -*-
"""
Phase 1 Experiment Runner — Version 2 (Output-space Taylor + blockwise exhaustive).

Methods evaluated:
  - random          : global ranking (3 seeds, averaged)
  - magnitude       : global ranking (L2 norm)
  - pointwise_legacy: global ranking using mean |c_{i,l,h}| (for comparison)
  - pointwise_bw    : per-block exhaustive, Q_pw objective
  - ewr_bw_alphaX   : per-block exhaustive, Q_ewr objective, ε=α×median_pairwise

Key V2 changes vs v1:
  - c_{i,l,h} = <δ_{i,l,h}, a_{i,l,h}> replaces g_j^T θ_j
  - Block-wise exhaustive search replaces global greedy ranking
  - Relative epsilon ε_l = α × median_pairwise_dist replaces absolute ε

Usage:
    python -m pilot_phase1.run_phase1 \\
        --data_root assert/CVC-ColonDB \\
        --dataset CVC-ColonDB \\
        --checkpoint work_dir/MedSAM/medsam_vit_b.pth \\
        --device cuda:0 \\
        --n_calibration 128 \\
        --alpha_values 0.1 0.5 1.0 \\
        --sparsities 0.5

    # Add --check_eps_sensitivity to run Check 3 diagnostics.
    # Add --multi_seed to re-run pointwise_bw / ewr_bw with 3 seeds for std.
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.head_pruning import (
    compute_head_gradient_projections_fast,
    score_heads_random,
    score_heads_magnitude,
    score_heads_pointwise,
    generate_head_mask,
    generate_head_mask_blockwise,
    diagnose_epsilon_sensitivity,
)
from pilot_phase1.evaluate import evaluate_pruned_model


# ==============================================================================
# Helpers
# ==============================================================================

def _avg_results(results_list):
    """Average a list of avg_metrics dicts; add std_ fields."""
    avg = {}
    for key in results_list[0]:
        vals = [r[key] for r in results_list]
        if isinstance(vals[0], (int, float, np.integer, np.floating)):
            avg[key]          = float(np.mean(vals))
            avg[f"std_{key}"] = float(np.std(vals))
    return avg


def _print_row(method, sparsity, exp):
    print(f"  {method:<22}: Dice={exp['mean_dice']:.4f}  "
          f"IoU={exp['mean_iou']:.4f}  BF1={exp['mean_boundary_f1']:.4f}  "
          f"HD95={exp['mean_hd95']:.2f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1 V2: Taylor proxy + blockwise exhaustive")

    parser.add_argument("--dataset", type=str, default="CVC-ColonDB",
                        choices=["CVC-ColonDB", "CVC-ClinicDB", "Kvasir-SEG"])
    parser.add_argument("--data_root",  type=str, default="assert/CVC-ColonDB")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--batch_size",    type=int, default=32,
                        help="Calibration batch size. Increase to saturate GPU "
                             "(40 GB VRAM → 32-64 safe for ViT-B).")
    parser.add_argument("--output_dir",    type=str, default="results/phase1_v2")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--num_workers",   type=int, default=4,
                        help="DataLoader worker processes for parallel data loading.")
    parser.add_argument("--sparsities",    type=float, nargs="+", default=[0.3, 0.5, 0.7])

    # V2: relative epsilon α values (replaces absolute --epsilons)
    parser.add_argument("--alpha_values", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                        help="Relative ε scale: ε_l = α × median_pairwise_dist(T_{i,l})")

    # Diagnostics
    parser.add_argument("--check_eps_sensitivity", action="store_true",
                        help="Run Check 3: ε sensitivity diagnostic (overlap matrix)")
    parser.add_argument("--multi_seed", action="store_true",
                        help="Re-run pointwise_bw / ewr_bw_alpha0.5 with 3 seeds for std")

    # Keep legacy EWR for comparison
    parser.add_argument("--include_legacy", action="store_true",
                        help="Include v1 global-ranking pointwise_legacy in results")

    # Sinkhorn iterations (can reduce to 50-100 to speed up exhaustive search)
    parser.add_argument("--sinkhorn_iters", type=int, default=100,
                        help="Sinkhorn iterations for EWR blockwise search (default 100)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("Phase 1 V2: Output-space Taylor proxy + Blockwise exhaustive search")
    print(f"  Dataset:        {args.dataset}  ({args.data_root})")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Device:         {device}")
    print(f"  Calibration:    {args.n_calibration} samples")
    print(f"  Sparsities:     {args.sparsities}")
    print(f"  Alpha values:   {args.alpha_values}  (relative ε)")
    print(f"  Sinkhorn iters: {args.sinkhorn_iters}")
    print(f"  Output:         {args.output_dir}")
    print("=" * 70)

    # ================================================================ #
    # 1. Load model
    # ================================================================ #
    print("\n[Step 1] Loading MedSAM model ...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = False

    print(f"  Encoder params: {sum(p.numel() for p in model.image_encoder.parameters()):,}")

    # ================================================================ #
    # 2. Build data loaders
    # ================================================================ #
    print("\n[Step 2] Building data loaders ...")
    cal_loader, test_loader, cal_dataset, test_dataset, cal_freq_weights = build_dataloaders(
        args.data_root,
        n_calibration=args.n_calibration,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    cv = cal_freq_weights.std() / max(cal_freq_weights.mean(), 1e-8)
    print(f"\n  ω_freq cv = {cv:.3f}  "
          f"({'discriminative (cv >= 0.3)' if cv >= 0.3 else 'low discriminativeness (cv < 0.3)'})")

    # ================================================================ #
    # 3. Baseline
    # ================================================================ #
    print("\n[Step 3] Evaluating unpruned baseline ...")
    baseline_results = evaluate_pruned_model(model, test_loader, head_mask=None, device=device)
    b = baseline_results["avg_metrics"]
    print(f"  Dice={b['mean_dice']:.4f}  IoU={b['mean_iou']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ================================================================ #
    # 4. Compute Taylor proxy c_{i,l,h}
    # ================================================================ #
    print("\n[Step 4] Computing output-space Taylor proxy ...")
    t0 = time.time()
    head_importance, per_sample_proj = compute_head_gradient_projections_fast(
        model, cal_loader, device, freq_weights=cal_freq_weights
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Save raw proxies
    np.save(os.path.join(args.output_dir, "per_sample_proj.npy"), per_sample_proj)
    np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance)

    # ================================================================ #
    # 4b. Check 3 — ε sensitivity diagnostic (optional)
    # ================================================================ #
    if args.check_eps_sensitivity:
        print("\n[Check 3] ε sensitivity diagnostic ...")
        # Use first sparsity (or 0.5 as default)
        diag_sparsity = 0.5 if 0.5 in args.sparsities else args.sparsities[0]
        masks_by_alpha, overlap_mat = diagnose_epsilon_sensitivity(
            per_sample_proj, diag_sparsity,
            alpha_values=tuple(args.alpha_values),
            n_iter=args.sinkhorn_iters,
            device=args.device,
        )

        print(f"\n  Block-mask overlap matrix (sparsity={diag_sparsity}):")
        print(f"  {'':>8}" + "".join(f"  α={a:.1f}" for a in args.alpha_values))
        for i, a1 in enumerate(args.alpha_values):
            row = f"  α={a1:.1f}  " + "  ".join(
                f"  {overlap_mat[i, j]:.2f}" for j in range(len(args.alpha_values))
            )
            print(row)

        all_same = all(
            overlap_mat[i, j] >= 0.99
            for i in range(len(args.alpha_values))
            for j in range(len(args.alpha_values))
        )
        if all_same:
            print("  ⚠️  All α values select identical masks → ε still not in effective range.")
        else:
            print("  ✅ α values produce different masks → ε is in effective working range.")

        # Save
        np.save(os.path.join(args.output_dir, "eps_overlap_matrix.npy"), overlap_mat)

    # ================================================================ #
    # 5. Compute scores for random / magnitude (global ranking)
    # ================================================================ #
    print("\n[Step 5] Computing global-ranking scores (random, magnitude) ...")
    random_scores   = {s: score_heads_random(seed=s)    for s in [42, 123, 456]}
    magnitude_score = score_heads_magnitude(model)
    pw_legacy_score = score_heads_pointwise(per_sample_proj)  # legacy global

    # ================================================================ #
    # 6. Experiment loop
    # ================================================================ #
    print("\n[Step 6] Running pruning experiments ...")

    all_results = {
        "config":      vars(args),
        "baseline":    b,
        "experiments": [],
    }

    for sparsity in args.sparsities:
        print(f"\n--- Sparsity: {sparsity * 100:.0f}% ---")

        # ---- Random (global ranking, 3 seeds averaged) ----
        rand_metrics = [
            evaluate_pruned_model(
                model, test_loader,
                generate_head_mask(random_scores[s], sparsity),
                device
            )["avg_metrics"]
            for s in [42, 123, 456]
        ]
        rand_avg = _avg_results(rand_metrics)
        rand_avg.update({"method": "random", "sparsity": sparsity, "n_seeds": 3})
        all_results["experiments"].append(rand_avg)
        _print_row("random", sparsity, rand_avg)

        # ---- Magnitude (global ranking) ----
        mag_mask   = generate_head_mask(magnitude_score, sparsity)
        mag_result = evaluate_pruned_model(model, test_loader, mag_mask, device)["avg_metrics"]
        mag_result.update({"method": "magnitude", "sparsity": sparsity})
        all_results["experiments"].append(mag_result)
        _print_row("magnitude", sparsity, mag_result)

        # ---- Pointwise legacy (global ranking, for comparison) ----
        if args.include_legacy:
            pw_leg_mask   = generate_head_mask(pw_legacy_score, sparsity)
            pw_leg_result = evaluate_pruned_model(
                model, test_loader, pw_leg_mask, device
            )["avg_metrics"]
            pw_leg_result.update({"method": "pointwise_legacy", "sparsity": sparsity})
            all_results["experiments"].append(pw_leg_result)
            _print_row("pointwise_legacy", sparsity, pw_leg_result)

        # ---- Pointwise blockwise (exhaustive) ----
        t0 = time.time()
        pw_bw_mask   = generate_head_mask_blockwise(
            per_sample_proj, sparsity, method='pointwise', device=args.device
        )
        pw_bw_result = evaluate_pruned_model(
            model, test_loader, pw_bw_mask, device
        )["avg_metrics"]
        pw_bw_result.update({
            "method": "pointwise_bw",
            "sparsity": sparsity,
            "search_time_s": round(time.time() - t0, 1),
        })
        all_results["experiments"].append(pw_bw_result)
        _print_row("pointwise_bw", sparsity, pw_bw_result)

        # ---- EWR blockwise (exhaustive, per alpha) ----
        for alpha in args.alpha_values:
            t0 = time.time()
            ewr_mask   = generate_head_mask_blockwise(
                per_sample_proj, sparsity, method='ewr',
                alpha=alpha, n_iter=args.sinkhorn_iters, device=args.device,
            )
            ewr_result = evaluate_pruned_model(
                model, test_loader, ewr_mask, device
            )["avg_metrics"]
            ewr_result.update({
                "method": f"ewr_bw_alpha{alpha}",
                "sparsity": sparsity,
                "alpha": alpha,
                "search_time_s": round(time.time() - t0, 1),
            })
            all_results["experiments"].append(ewr_result)
            _print_row(f"ewr_bw_alpha{alpha}", sparsity, ewr_result)

    # ================================================================ #
    # 6b. Multi-seed stability (optional)
    # ================================================================ #
    if args.multi_seed:
        print("\n[Step 6b] Multi-seed stability for pointwise_bw / ewr_bw_alpha0.5 ...")
        # Use fixed sparsity 0.5 (or first if not present)
        ms_sparsity = 0.5 if 0.5 in args.sparsities else args.sparsities[0]
        ms_alpha    = 0.5 if 0.5 in args.alpha_values else args.alpha_values[0]

        for method_name, method_kw in [
            ("pointwise_bw",           dict(method="pointwise")),
            (f"ewr_bw_alpha{ms_alpha}", dict(method="ewr", alpha=ms_alpha,
                                             n_iter=args.sinkhorn_iters)),
        ]:
            seed_metrics = []
            for seed in [42, 123, 456]:
                cal_l, _, _, _, fw = build_dataloaders(
                    args.data_root, n_calibration=args.n_calibration,
                    batch_size=args.batch_size, seed=seed,
                    num_workers=args.num_workers,
                )
                _, proj_s = compute_head_gradient_projections_fast(
                    model, cal_l, device, freq_weights=fw
                )
                mask_s = generate_head_mask_blockwise(
                    proj_s, ms_sparsity, verbose=False,
                    device=args.device, **method_kw
                )
                seed_metrics.append(
                    evaluate_pruned_model(model, test_loader, mask_s, device)["avg_metrics"]
                )

            ms_avg = _avg_results(seed_metrics)
            ms_avg.update({
                "method": f"{method_name}_multiseed",
                "sparsity": ms_sparsity,
                "n_seeds": 3,
            })
            all_results["experiments"].append(ms_avg)
            print(f"  {method_name}(ms): "
                  f"Dice={ms_avg['mean_dice']:.4f}±{ms_avg['std_mean_dice']:.4f}")

    # ================================================================ #
    # 7. Save results
    # ================================================================ #
    results_path = os.path.join(args.output_dir, "phase1_results.json")

    def _ser(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(json.loads(json.dumps(all_results, default=_ser)), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ================================================================ #
    # 8. Summary table
    # ================================================================ #
    print("\n" + "=" * 90)
    print("PHASE 1 V2 RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<24} {'Sparsity':>8} {'Dice':>8} {'IoU':>8} {'BF1':>8} {'HD95':>8}")
    print("-" * 72)
    print(f"{'unpruned':<24} {'0%':>8} {b['mean_dice']:>8.4f} {b['mean_iou']:>8.4f} "
          f"{b['mean_boundary_f1']:>8.4f} {b['mean_hd95']:>8.2f}")
    print("-" * 72)

    for sparsity in args.sparsities:
        exps = [e for e in all_results["experiments"]
                if e["sparsity"] == sparsity and "multiseed" not in e["method"]]
        for e in exps:
            print(f"{e['method']:<24} {sparsity*100:>7.0f}%"
                  f" {e['mean_dice']:>8.4f} {e['mean_iou']:>8.4f}"
                  f" {e['mean_boundary_f1']:>8.4f} {e['mean_hd95']:>8.2f}")
        print("-" * 72)

    # ================================================================ #
    # 9. Go/No-Go gate
    # ================================================================ #
    print("\n" + "=" * 70)
    print("GO/NO-GO GATE EVALUATION (V2: blockwise exhaustive)")
    print("=" * 70)

    baseline_dice = b["mean_dice"]

    # Gate 1: Best EWR_bw @ 50% >= Pointwise_bw + 1.5%
    ewr_50 = [e for e in all_results["experiments"]
              if "ewr_bw" in e["method"] and "multiseed" not in e["method"]
              and e["sparsity"] == 0.5]
    pw_50  = [e for e in all_results["experiments"]
              if e["method"] == "pointwise_bw" and e["sparsity"] == 0.5]

    if ewr_50 and pw_50:
        best_ewr = max(ewr_50, key=lambda x: x["mean_dice"])
        pw_dice  = pw_50[0]["mean_dice"]
        gap      = best_ewr["mean_dice"] - pw_dice
        print(f"\n  [Gate 1 — 50% sparsity]")
        print(f"    Best EWR ({best_ewr['method']}): {best_ewr['mean_dice']:.4f}")
        print(f"    Pointwise_bw:                    {pw_dice:.4f}")
        print(f"    Gap:                             {gap:+.4f}  (need >= +0.0150)")
        if gap >= 0.015:
            print("    PASS: EWR_bw >= Pointwise_bw + 1.5%")
        else:
            print("    FAIL: EWR_bw < Pointwise_bw + 1.5%")
            if gap > 0:
                print("       → EWR is better but gap < 1.5%. Try different α or more iters.")
            else:
                print("       → EWR still not beating Pointwise. 1D block-response may be "
                      "insufficient; see §7 of phase1_next.md (fallback plan).")

    # Gate 2: 70% sparsity collapse check
    pw_70  = [e for e in all_results["experiments"]
              if e["method"] == "pointwise_bw" and e["sparsity"] == 0.7]
    ewr_70 = [e for e in all_results["experiments"]
              if "ewr_bw" in e["method"] and "multiseed" not in e["method"]
              and e["sparsity"] == 0.7]

    if pw_70 and ewr_70:
        best_ewr_70 = max(ewr_70, key=lambda x: x["mean_dice"])
        pw_drop     = baseline_dice - pw_70[0]["mean_dice"]
        ewr_drop    = baseline_dice - best_ewr_70["mean_dice"]
        print(f"\n  [Gate 2 — 70% sparsity]")
        print(f"    Pointwise_bw drop: {pw_drop:.4f}  (collapse if > 0.05)")
        print(f"    EWR_bw drop:       {ewr_drop:.4f}  (need <= 0.03)")
        pw_collapse = pw_drop > 0.05
        ewr_robust  = ewr_drop <= 0.03
        print(f"    {'  OK' if pw_collapse  else '  --'} Pointwise collapse: "
              f"{'YES' if pw_collapse else 'NO'}")
        print(f"    {'PASS' if ewr_robust else 'FAIL'} EWR robustness")

    print("\n" + "=" * 70)
    print(f"Experiment complete. Results: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
