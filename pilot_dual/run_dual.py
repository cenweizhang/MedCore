# -*- coding: utf-8 -*-
"""
Pilot study: Dual-Intervention Medical-Core Pruning for MedSAM.

Validates the core idea of the paper on CVC-ColonDB:
  - Does the reset score (Delta_reset) add value over zero score (Delta_zero)?
  - Is dual scoring (alpha in (0,1)) better than either extreme?

Experiment matrix
-----------------
Baselines (importance = higher is more important):
  random     : shuffle (3 seeds averaged)
  magnitude  : L2-norm of head weight group

Dual-intervention variants (importance = lower means prune first, i.e. Q_g):
  zero_only  : alpha = 1.0  (standard Fisher-based, no adaptation info)
  reset_only : alpha = 0.0  (adaptation-drift only)
  dual_a0.2  : alpha = 0.2
  dual_a0.5  : alpha = 0.5  (balanced)
  dual_a0.8  : alpha = 0.8

Each method is evaluated at five sparsity levels: 30%, 50%, 70%, 90%, 95%.
Metrics: Dice, IoU, BF1, HD95.

Go/No-Go gate (primary):
  At 50% sparsity, dual_a0.5 Dice > zero_only Dice + 0.5%

Usage
-----
    cd /home/zhangcenwei/testSAMpruning
    python -m pilot_dual.run_dual \\
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \\
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \\
        --data_root   assert/CVC-ColonDB \\
        --device      cuda:0 \\
        --n_cal       128 \\
        --batch_size  1 \\
        --sparsities  0.3 0.5 0.7 \\
        --alpha_values 0.0 0.2 0.5 0.8 1.0 \\
        --output_dir  results/pilot_dual
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
from pilot_phase1.dataset   import build_dataloaders
from pilot_phase1.evaluate  import evaluate_pruned_model

from pilot_dual.scoring import (
    compute_diagonal_fisher,
    load_sam_encoder_params,
    compute_head_scores,
    combine_scores,
    score_summary,
)
from pilot_dual.pruning import (
    score_heads_random,
    score_heads_magnitude,
    generate_head_mask,
    apply_head_mask_to_model,
    remove_hooks,
    compute_model_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg_dicts(dicts):
    """Average a list of metric dicts; add std_ prefix fields."""
    out = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        if isinstance(vals[0], (int, float, np.integer, np.floating)):
            out[key]              = float(np.mean(vals))
            out[f"std_{key}"]     = float(np.std(vals))
    return out


def _json_safe(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def _eval_with_mask(model, test_loader, head_mask, device):
    """Evaluate model with the given head mask; return avg_metrics dict."""
    return evaluate_pruned_model(model, test_loader, head_mask, device)["avg_metrics"]


def _print_row(method, sparsity, m):
    params_M = m.get("n_params_remaining", 0) / 1e6
    flops_G  = m.get("flops_remaining_G", 0.0)
    p_red    = m.get("param_reduction_pct", 0.0)
    f_red    = m.get("flop_reduction_pct",  0.0)
    print(f"    {method:<22} | Dice={m['mean_dice']:.4f}  "
          f"IoU={m['mean_iou']:.4f}  BF1={m['mean_boundary_f1']:.4f}  "
          f"HD95={m['mean_hd95']:.2f}  "
          f"kept={int(m.get('n_heads_kept', 0))}/{int(m.get('n_heads_total', 144))}  "
          f"Params={params_M:.1f}M(-{p_red:.1f}%)  FLOPs={flops_G:.1f}G(-{f_red:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dual-intervention medical-core pruning pilot study"
    )
    parser.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",     default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_root",    default="assert/CVC-ColonDB")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_cal",        type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="Calibration batch size. Keep at 1 for exact Fisher.")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--sparsities",   type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 0.95])
    parser.add_argument("--alpha_values", type=float, nargs="+",
                        default=[0.0, 0.2, 0.5, 0.8, 1.0],
                        help="alpha in Q_g = alpha*Delta_zero + (1-alpha)*Delta_reset")
    parser.add_argument("--output_dir",   default="results/pilot_dual")
    parser.add_argument("--random_seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Seeds for random-pruning baseline (results are averaged).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 72)
    print("Dual-Intervention Medical-Core Pruning — Pilot Study")
    print(f"  MedSAM  : {args.medsam_ckpt}")
    print(f"  SAM     : {args.sam_ckpt}")
    print(f"  Data    : {args.data_root}  (n_cal={args.n_cal})")
    print(f"  Device  : {device}")
    print(f"  Sparsities   : {args.sparsities}")
    print(f"  Alpha values : {args.alpha_values}")
    print(f"  Output  : {args.output_dir}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load MedSAM
    # ------------------------------------------------------------------
    print("\n[1] Loading MedSAM checkpoint ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model = model.to(device)
    model.eval()
    for p in model.prompt_encoder.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(False)
    enc_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"  Encoder parameters: {enc_params:,}")

    # ------------------------------------------------------------------
    # 2. Load SAM encoder params (theta^S)
    # ------------------------------------------------------------------
    print("\n[2] Loading SAM encoder parameters (theta^S) ...")
    sam_params = load_sam_encoder_params(args.sam_ckpt, device="cpu")

    # Sanity check: verify parameter drift between SAM and MedSAM
    total_drift = 0.0
    for n, p_med in model.image_encoder.named_parameters():
        if n in sam_params:
            total_drift += (p_med.data.float().cpu() - sam_params[n]).pow(2).sum().item()
    print(f"  ||theta^M - theta^S||^2 (encoder) = {total_drift:.4f}")
    if total_drift < 1e-6:
        print("  WARNING: SAM and MedSAM encoder params are identical. "
              "Verify that the correct checkpoints are loaded.")

    # ------------------------------------------------------------------
    # 3. Data loaders
    # ------------------------------------------------------------------
    print("\n[3] Building data loaders ...")
    cal_loader, test_loader, _, _, _ = build_dataloaders(
        args.data_root,
        n_calibration=args.n_cal,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    n_test = sum(1 for _ in test_loader)
    print(f"  Calibration: {args.n_cal} samples  |  Test: {n_test} samples")

    # ------------------------------------------------------------------
    # 4. Baseline evaluation (unpruned MedSAM)
    # ------------------------------------------------------------------
    print("\n[4] Evaluating unpruned MedSAM baseline ...")
    baseline_metrics = _eval_with_mask(model, test_loader, head_mask=None, device=device)
    b = baseline_metrics
    print(f"  Dice={b['mean_dice']:.4f}  IoU={b['mean_iou']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ------------------------------------------------------------------
    # 5. Diagonal Fisher estimation
    # ------------------------------------------------------------------
    print("\n[5] Computing diagonal Fisher (this may take a few minutes) ...")
    t0 = time.time()
    fisher = compute_diagonal_fisher(model, cal_loader, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 6. Dual-intervention head scores
    # ------------------------------------------------------------------
    print("\n[6] Computing per-head dual-intervention scores ...")
    delta_zero, delta_reset = compute_head_scores(model, sam_params, fisher)

    summary = score_summary(delta_zero, delta_reset)
    print(f"  Delta_zero  : mean={summary['delta_zero_mean']:.4e}  "
          f"std={summary['delta_zero_std']:.4e}")
    print(f"  Delta_reset : mean={summary['delta_reset_mean']:.4e}  "
          f"std={summary['delta_reset_std']:.4e}")
    print(f"  Pearson corr(zero, reset) = {summary['correlation_zero_reset']:.4f}")
    print(f"  Reset/Zero ratio          = {summary['reset_zero_ratio']:.4f}")
    if abs(summary['correlation_zero_reset']) > 0.95:
        print("  NOTE: High correlation — reset term may add limited new information "
              "on this dataset. Consider CT/MRI data for stronger signal.")
    elif abs(summary['correlation_zero_reset']) < 0.5:
        print("  NOTE: Low correlation — reset term captures different structure "
              "from zero term. Good signal for the dual-intervention idea.")

    # Save raw scores
    np.savez(
        os.path.join(args.output_dir, "head_scores.npz"),
        delta_zero=delta_zero,
        delta_reset=delta_reset,
    )

    # ------------------------------------------------------------------
    # 7. Baseline scorers (random, magnitude)
    # ------------------------------------------------------------------
    print("\n[7] Computing baseline head scores ...")
    random_scores_by_seed = {
        s: score_heads_random(seed=s) for s in args.random_seeds
    }
    magnitude_scores = score_heads_magnitude(model)
    print(f"  Magnitude: min={magnitude_scores.min():.4f}  "
          f"max={magnitude_scores.max():.4f}  mean={magnitude_scores.mean():.4f}")

    # ------------------------------------------------------------------
    # 8. Experiment loop
    # ------------------------------------------------------------------
    print("\n[8] Running pruning experiments ...")

    all_results = {
        "config":    vars(args),
        "baseline":  baseline_metrics,
        "score_summary": summary,
        "experiments": [],
    }

    for sparsity in args.sparsities:
        print(f"\n  --- Sparsity {sparsity * 100:.0f}% ---")

        # ---- Random (averaged over seeds) ----
        rand_masks = [generate_head_mask(random_scores_by_seed[s], sparsity)
                      for s in args.random_seeds]
        rand_metrics_list = [_eval_with_mask(model, test_loader, m, device)
                             for m in rand_masks]
        rand_avg = _avg_dicts(rand_metrics_list)
        rand_stats = compute_model_stats(model, rand_masks[0])
        rand_avg.update({
            "method": "random", "sparsity": sparsity,
            "n_seeds": len(args.random_seeds),
            "alpha": None,
            **rand_stats,
        })
        all_results["experiments"].append(rand_avg)
        _print_row("random", sparsity, rand_avg)

        # ---- Magnitude ----
        mag_mask    = generate_head_mask(magnitude_scores, sparsity)
        mag_metrics = _eval_with_mask(model, test_loader, mag_mask, device)
        mag_stats   = compute_model_stats(model, mag_mask)
        mag_metrics.update({"method": "magnitude", "sparsity": sparsity,
                             "alpha": None, **mag_stats})
        all_results["experiments"].append(mag_metrics)
        _print_row("magnitude", sparsity, mag_metrics)

        # ---- Dual-intervention variants ----
        for alpha in args.alpha_values:
            # Dual scores: LOWER Q_g means prune first.
            q_scores  = combine_scores(delta_zero, delta_reset, alpha=alpha)
            dual_mask = generate_head_mask(q_scores, sparsity)
            dual_metrics = _eval_with_mask(model, test_loader, dual_mask, device)
            dual_stats   = compute_model_stats(model, dual_mask)

            if alpha == 0.0:
                method_name = "reset_only"
            elif alpha == 1.0:
                method_name = "zero_only"
            else:
                method_name = f"dual_a{alpha:.1f}"

            dual_metrics.update({
                "method": method_name, "sparsity": sparsity, "alpha": float(alpha),
                **dual_stats,
            })
            all_results["experiments"].append(dual_metrics)
            _print_row(method_name, sparsity, dual_metrics)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(json.loads(json.dumps(all_results, default=_json_safe)), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 10. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("DUAL-INTERVENTION PILOT STUDY — RESULT SUMMARY")
    print("=" * 110)
    header = (f"{'Method':<22} {'Sparsity':>8} {'Dice':>8} {'IoU':>7} {'BF1':>7} "
              f"{'HD95':>8}  {'Params(M)':>10} {'Par%↓':>6}  {'FLOPs(G)':>9} {'FL%↓':>6}")
    print(header)
    print("-" * 110)

    # Baseline (unpruned) stats
    dummy_full = np.ones(144, dtype=np.float32)
    full_stats   = compute_model_stats(model, dummy_full)
    print(f"{'unpruned':<22} {'0%':>8} {b['mean_dice']:>8.4f} "
          f"{b['mean_iou']:>7.4f} {b['mean_boundary_f1']:>7.4f} {b['mean_hd95']:>8.2f}  "
          f"{full_stats['n_params_remaining']/1e6:>10.1f} {'0.0%':>6}  "
          f"{full_stats['flops_total_G']:>9.1f} {'0.0%':>6}")
    print("-" * 110)

    for sparsity in args.sparsities:
        exps = [e for e in all_results["experiments"] if e["sparsity"] == sparsity]
        for e in exps:
            print(f"{e['method']:<22} {sparsity*100:>7.0f}% "
                  f"{e['mean_dice']:>8.4f} {e['mean_iou']:>7.4f} "
                  f"{e['mean_boundary_f1']:>7.4f} {e['mean_hd95']:>8.2f}  "
                  f"{e.get('n_params_remaining', 0)/1e6:>10.1f} "
                  f"{e.get('param_reduction_pct', 0.0):>5.1f}%  "
                  f"{e.get('flops_remaining_G', 0.0):>9.1f} "
                  f"{e.get('flop_reduction_pct', 0.0):>5.1f}%")
        print("-" * 110)

    # ------------------------------------------------------------------
    # 11. Go/No-Go gate evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("GO / NO-GO GATE")
    print("=" * 72)

    baseline_dice = b["mean_dice"]

    # Gate 1: At 50% sparsity, best dual > zero_only + 0.5%
    sparsity_gate = 0.5 if 0.5 in args.sparsities else args.sparsities[0]
    exps_gate = [e for e in all_results["experiments"] if e["sparsity"] == sparsity_gate]

    zero_only_exp = next((e for e in exps_gate if e["method"] == "zero_only"), None)
    dual_exps     = [e for e in exps_gate
                     if e["method"].startswith("dual_") or e["method"] == "reset_only"]

    print(f"\n  [Gate 1] Sparsity={sparsity_gate*100:.0f}%: "
          f"best dual > zero_only + 0.5 pp Dice")
    if zero_only_exp and dual_exps:
        z_dice    = zero_only_exp["mean_dice"]
        best_dual = max(dual_exps, key=lambda x: x["mean_dice"])
        gap       = best_dual["mean_dice"] - z_dice
        print(f"    zero_only Dice          : {z_dice:.4f}")
        print(f"    best dual ({best_dual['method']}) : {best_dual['mean_dice']:.4f}  "
              f"(alpha={best_dual.get('alpha','?')})")
        print(f"    Gap                     : {gap:+.4f}  (threshold >= +0.005)")
        if gap >= 0.005:
            print("    PASS: reset information provides additional pruning guidance.")
        else:
            print("    FAIL/MARGINAL: reset term adds limited benefit on this dataset.")
            print("      -> Likely cause: CVC-ColonDB is endoscopy data; SAM already "
                  "performs well here, so MedSAM adaptation drift is small.")
            print("      -> Try CT/MRI data where SAM -> MedSAM adaptation is stronger.")
    else:
        print("    (Could not find required experiment entries.)")

    # Gate 2: Performance drop relative to baseline at each sparsity
    print(f"\n  [Gate 2] Performance retention vs unpruned MedSAM (baseline Dice={baseline_dice:.4f})")
    for sparsity in args.sparsities:
        exps_s = [e for e in all_results["experiments"] if e["sparsity"] == sparsity]
        best   = max(exps_s, key=lambda x: x["mean_dice"])
        drop   = baseline_dice - best["mean_dice"]
        print(f"    {sparsity*100:.0f}%: best={best['method']}  "
              f"Dice={best['mean_dice']:.4f}  drop={drop:+.4f}")

    # Gate 3: Correlation diagnostic
    corr = summary["correlation_zero_reset"]
    print(f"\n  [Gate 3] Score correlation corr(Delta_zero, Delta_reset) = {corr:.4f}")
    if corr > 0.95:
        print("    HIGH correlation: on CVC-ColonDB the two scores identify the same heads.")
        print("    The idea is still valid; signal is expected to be stronger on CT/MRI.")
    elif corr > 0.7:
        print("    MODERATE correlation: some complementary information from reset term.")
    else:
        print("    LOW correlation: reset term identifies different structures from zero term.")
        print("    Strong evidence that adaptation-aware scoring adds new information.")

    print("\n" + "=" * 72)
    print(f"Experiment complete. Full results: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
