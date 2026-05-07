# -*- coding: utf-8 -*-
"""
Phase-2 Cascade Pruning: Head → MLP dual-intervention pilot.

Experiment design
-----------------
Phase 1 – Attention head pruning:
  Sparsities : [0.3, 0.5, 0.7]
  Method     : configurable via --phase1_alpha (default 1.0 = zero_only)

Phase 2 – MLP neuron pruning on the Phase-1-pruned model:
  Sparsities : [0.3, 0.5, 0.7, 0.9]
  Methods    : random, magnitude, reset_only, dual_a0.2, dual_a0.5,
               dual_a0.8, zero_only

Both phases use the same Fisher information estimated once on the full model.
The MLP scores are computed with the same Eq. 25–27 as attention head scores,
with group g^mlp_{l,n} = { lin1.weight[n,:], lin1.bias[n], lin2.weight[:,n] }.

Implementation note
-------------------
Head hooks are applied externally (no weight modification).
MLP hooks are then layered on top for inner-loop evaluation.
evaluate_pruned_model is called with head_mask=None so it does not
re-add head hooks that are already active.

Usage
-----
    cd /home/zhangcenwei/testSAMpruning
    /home/zhangcenwei/miniconda3/envs/medsam/bin/python -m pilot_dual.run_cascade \\
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \\
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \\
        --data_root   assert/CVC-ColonDB \\
        --device      cuda:0 \\
        --output_dir  results/pilot_cascade
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
    compute_mlp_neuron_scores,
    combine_scores,
    score_summary,
)
from pilot_dual.pruning import (
    score_heads_random,
    score_heads_magnitude,
    score_mlp_magnitude,
    generate_head_mask,
    generate_neuron_mask,
    apply_head_mask_to_model,
    apply_mlp_mask_to_model,
    remove_hooks,
    compute_cascade_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def _eval_cascade(model, test_loader, device):
    """Evaluate model with currently-active hooks (head + MLP).
    Passes head_mask=None so evaluate_pruned_model does not re-add head hooks."""
    return evaluate_pruned_model(model, test_loader, head_mask=None, device=device)["avg_metrics"]


def _print_cascade_row(head_sp, mlp_method, mlp_sp, m, stats):
    p_red  = stats.get("param_reduction_pct", 0.0)
    f_red  = stats.get("flop_reduction_pct",  0.0)
    fl_rem = stats.get("flops_remaining_G",   0.0)
    print(f"    MLP {mlp_method:<22} sp={mlp_sp*100:.0f}% | "
          f"Dice={m['mean_dice']:.4f}  IoU={m['mean_iou']:.4f}  "
          f"HD95={m['mean_hd95']:.2f}  "
          f"Params↓{p_red:.1f}%  FLOPs={fl_rem:.1f}G↓{f_red:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cascade head→MLP dual-intervention pruning pilot"
    )
    parser.add_argument("--medsam_ckpt",   default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",      default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_root",     default="assert/CVC-ColonDB")
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--n_cal",         type=int, default=128)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--seed",          type=int, default=42)
    # Phase 1
    parser.add_argument("--head_sparsities", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7],
                        help="Phase-1 head pruning sparsity levels.")
    parser.add_argument("--phase1_alpha",  type=float, default=1.0,
                        help="alpha for Phase-1 head scoring. 1.0=zero_only (best on CVC).")
    # Phase 2
    parser.add_argument("--mlp_sparsities", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="Phase-2 MLP neuron pruning sparsity levels.")
    parser.add_argument("--alpha_values",  type=float, nargs="+",
                        default=[0.0, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--random_seeds",  type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output_dir",    default="results/pilot_cascade")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 72)
    print("CASCADE DUAL-INTERVENTION PRUNING — Head → MLP Pilot Study")
    print(f"  MedSAM  : {args.medsam_ckpt}")
    print(f"  SAM     : {args.sam_ckpt}")
    print(f"  Data    : {args.data_root}  (n_cal={args.n_cal})")
    print(f"  Device  : {device}")
    print(f"  Phase-1 head sparsities : {args.head_sparsities}  "
          f"(alpha={args.phase1_alpha})")
    print(f"  Phase-2 MLP  sparsities : {args.mlp_sparsities}")
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

    total_drift = sum(
        (p.data.float().cpu() - sam_params[n]).pow(2).sum().item()
        for n, p in model.image_encoder.named_parameters()
        if n in sam_params
    )
    print(f"  ||theta^M - theta^S||^2 (encoder) = {total_drift:.4f}")

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
    # 4. Baseline evaluation (unpruned)
    # ------------------------------------------------------------------
    print("\n[4] Evaluating unpruned MedSAM baseline ...")
    baseline_metrics = _eval_cascade(model, test_loader, device)
    b = baseline_metrics
    print(f"  Dice={b['mean_dice']:.4f}  IoU={b['mean_iou']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ------------------------------------------------------------------
    # 5. Diagonal Fisher (once, on full model)
    # ------------------------------------------------------------------
    print("\n[5] Computing diagonal Fisher on full model ...")
    t0 = time.time()
    fisher = compute_diagonal_fisher(model, cal_loader, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 6. Head scores (Phase 1)
    # ------------------------------------------------------------------
    print("\n[6] Computing attention head scores ...")
    delta_zero_head, delta_reset_head = compute_head_scores(model, sam_params, fisher)
    head_summary = score_summary(delta_zero_head, delta_reset_head)
    print(f"  corr(zero,reset)={head_summary['correlation_zero_reset']:.4f}  "
          f"reset/zero ratio={head_summary['reset_zero_ratio']:.4f}")

    # Phase-1 head scores using specified alpha
    phase1_alpha = args.phase1_alpha
    head_q_scores = combine_scores(delta_zero_head, delta_reset_head, alpha=phase1_alpha)
    phase1_method = ("zero_only" if phase1_alpha == 1.0 else
                     "reset_only" if phase1_alpha == 0.0 else
                     f"dual_a{phase1_alpha:.1f}")
    print(f"  Phase-1 head method: {phase1_method}")

    # Baseline head scorers
    random_head_scores = {s: score_heads_random(seed=s) for s in args.random_seeds}
    magnitude_head_scores = score_heads_magnitude(model)

    # ------------------------------------------------------------------
    # 7. MLP neuron scores (Phase 2)
    # ------------------------------------------------------------------
    print("\n[7] Computing MLP neuron scores ...")
    delta_zero_mlp, delta_reset_mlp = compute_mlp_neuron_scores(
        model, sam_params, fisher
    )
    mlp_summary = score_summary(delta_zero_mlp, delta_reset_mlp)
    print(f"  MLP neurons: {len(delta_zero_mlp):,}  "
          f"corr(zero,reset)={mlp_summary['correlation_zero_reset']:.4f}")
    print(f"  delta_zero_mean={mlp_summary['delta_zero_mean']:.4e}  "
          f"delta_reset_mean={mlp_summary['delta_reset_mean']:.4e}")

    mlp_magnitude_scores = score_mlp_magnitude(model)
    mlp_random_scores    = {s: np.random.RandomState(s).rand(
                                len(delta_zero_mlp)).astype(np.float32)
                            for s in args.random_seeds}

    # Save raw scores
    np.savez(
        os.path.join(args.output_dir, "scores.npz"),
        delta_zero_head=delta_zero_head,
        delta_reset_head=delta_reset_head,
        delta_zero_mlp=delta_zero_mlp,
        delta_reset_mlp=delta_reset_mlp,
    )

    # ------------------------------------------------------------------
    # 8. Cascade experiment loop
    # ------------------------------------------------------------------
    print("\n[8] Running cascade pruning experiments ...")
    print("    (Head hooks applied externally; MLP hooks layered inside)")

    all_results = {
        "config":           vars(args),
        "baseline":         baseline_metrics,
        "head_score_summary": head_summary,
        "mlp_score_summary":  mlp_summary,
        "cascade_results":  [],
    }

    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    for head_sp in args.head_sparsities:
        print(f"\n  ====== Phase-1 head sparsity {head_sp*100:.0f}% "
              f"(method={phase1_method}) ======")

        # Generate Phase-1 head mask
        head_mask = generate_head_mask(head_q_scores, head_sp)
        n_heads_kept = int(head_mask.sum())

        # Apply head hooks for the entire inner MLP loop
        head_hooks = apply_head_mask_to_model(model, head_mask)

        # Evaluate head-only pruned model (no MLP pruning yet)
        head_only_metrics = _eval_cascade(model, test_loader, device)
        head_only_stats   = compute_cascade_stats(model, head_mask, None)
        print(f"    Head-only: Dice={head_only_metrics['mean_dice']:.4f}  "
              f"IoU={head_only_metrics['mean_iou']:.4f}  "
              f"Heads={n_heads_kept}/144  "
              f"Params↓{head_only_stats['param_reduction_pct']:.1f}%  "
              f"FLOPs={head_only_stats['flops_remaining_G']:.1f}G"
              f"↓{head_only_stats['flop_reduction_pct']:.1f}%")

        all_results["cascade_results"].append({
            "phase":          "head_only",
            "head_method":    phase1_method,
            "head_sparsity":  head_sp,
            "mlp_method":     None,
            "mlp_sparsity":   None,
            **head_only_metrics,
            **head_only_stats,
        })

        # Inner loop: Phase-2 MLP pruning
        for mlp_sp in args.mlp_sparsities:
            print(f"\n    -- MLP sparsity {mlp_sp*100:.0f}% --")

            # ---- Random MLP (averaged over seeds) ----
            rand_mlp_metrics_list = []
            for s in args.random_seeds:
                nm = generate_neuron_mask(mlp_random_scores[s], mlp_sp,
                                          mlp_dim=mlp_dim)
                mlp_hooks = apply_mlp_mask_to_model(model, nm)
                rand_mlp_metrics_list.append(_eval_cascade(model, test_loader, device))
                remove_hooks(mlp_hooks)

            rand_mlp_avg = {
                k: float(np.mean([m[k] for m in rand_mlp_metrics_list]))
                for k in rand_mlp_metrics_list[0]
                if isinstance(rand_mlp_metrics_list[0][k], (int, float, np.integer, np.floating))
            }
            rand_nm    = generate_neuron_mask(mlp_random_scores[args.random_seeds[0]],
                                              mlp_sp, mlp_dim=mlp_dim)
            rand_stats = compute_cascade_stats(model, head_mask, rand_nm)
            rand_entry = {
                "phase": "cascade", "head_method": phase1_method,
                "head_sparsity": head_sp, "mlp_method": "random",
                "mlp_sparsity": mlp_sp, "n_seeds": len(args.random_seeds),
                **rand_mlp_avg, **rand_stats,
            }
            all_results["cascade_results"].append(rand_entry)
            _print_cascade_row(head_sp, "random", mlp_sp, rand_mlp_avg, rand_stats)

            # ---- Magnitude MLP ----
            mag_nm      = generate_neuron_mask(mlp_magnitude_scores, mlp_sp,
                                               mlp_dim=mlp_dim)
            mlp_hooks   = apply_mlp_mask_to_model(model, mag_nm)
            mag_metrics = _eval_cascade(model, test_loader, device)
            remove_hooks(mlp_hooks)
            mag_stats   = compute_cascade_stats(model, head_mask, mag_nm)
            all_results["cascade_results"].append({
                "phase": "cascade", "head_method": phase1_method,
                "head_sparsity": head_sp, "mlp_method": "magnitude",
                "mlp_sparsity": mlp_sp,
                **mag_metrics, **mag_stats,
            })
            _print_cascade_row(head_sp, "magnitude", mlp_sp, mag_metrics, mag_stats)

            # ---- Dual-intervention MLP variants ----
            for alpha in args.alpha_values:
                q_mlp = combine_scores(delta_zero_mlp, delta_reset_mlp, alpha=alpha)
                nm    = generate_neuron_mask(q_mlp, mlp_sp, mlp_dim=mlp_dim)

                mlp_hooks   = apply_mlp_mask_to_model(model, nm)
                dual_metrics = _eval_cascade(model, test_loader, device)
                remove_hooks(mlp_hooks)
                dual_stats = compute_cascade_stats(model, head_mask, nm)

                if alpha == 0.0:
                    mlp_method_name = "reset_only"
                elif alpha == 1.0:
                    mlp_method_name = "zero_only"
                else:
                    mlp_method_name = f"dual_a{alpha:.1f}"

                all_results["cascade_results"].append({
                    "phase": "cascade", "head_method": phase1_method,
                    "head_sparsity": head_sp, "mlp_method": mlp_method_name,
                    "mlp_sparsity": mlp_sp, "alpha": float(alpha),
                    **dual_metrics, **dual_stats,
                })
                _print_cascade_row(head_sp, mlp_method_name, mlp_sp,
                                   dual_metrics, dual_stats)

        # Remove head hooks before next head_sparsity
        remove_hooks(head_hooks)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "cascade_results.json")
    with open(results_path, "w") as f:
        json.dump(json.loads(json.dumps(all_results, default=_json_safe)), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 10. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CASCADE PRUNING — RESULT SUMMARY  (Dice)")
    print("=" * 120)
    header = (f"{'Head-sp':>8} {'MLP-method':<22} {'MLP-sp':>7} "
              f"{'Dice':>8} {'IoU':>7} {'HD95':>8}  "
              f"{'Params(M)':>10} {'Par%↓':>6}  {'FLOPs(G)':>9} {'FL%↓':>6}")
    print(header)
    print("-" * 120)

    print(f"{'unpruned':>8} {'-':<22} {'-':>7} "
          f"{b['mean_dice']:>8.4f} {b['mean_iou']:>7.4f} {b['mean_hd95']:>8.2f}  "
          f"{'89.7':>10} {'0.0%':>6}  {'926.5':>9} {'0.0%':>6}")
    print("-" * 120)

    for head_sp in args.head_sparsities:
        # Head-only row
        head_rows = [r for r in all_results["cascade_results"]
                     if r["head_sparsity"] == head_sp and r["phase"] == "head_only"]
        if head_rows:
            h = head_rows[0]
            print(f"{head_sp*100:>7.0f}% {'[HEAD ONLY]':<22} {'-':>7} "
                  f"{h['mean_dice']:>8.4f} {h['mean_iou']:>7.4f} {h['mean_hd95']:>8.2f}  "
                  f"{h.get('n_params_remaining',0)/1e6:>10.1f} "
                  f"{h.get('param_reduction_pct',0.0):>5.1f}%  "
                  f"{h.get('flops_remaining_G',0.0):>9.1f} "
                  f"{h.get('flop_reduction_pct',0.0):>5.1f}%")

        # Cascade rows
        for mlp_sp in args.mlp_sparsities:
            rows = [r for r in all_results["cascade_results"]
                    if r["head_sparsity"] == head_sp
                    and r.get("mlp_sparsity") == mlp_sp
                    and r["phase"] == "cascade"]
            for r in rows:
                print(f"{head_sp*100:>7.0f}% {r['mlp_method']:<22} {mlp_sp*100:>6.0f}% "
                      f"{r['mean_dice']:>8.4f} {r['mean_iou']:>7.4f} {r['mean_hd95']:>8.2f}  "
                      f"{r.get('n_params_remaining',0)/1e6:>10.1f} "
                      f"{r.get('param_reduction_pct',0.0):>5.1f}%  "
                      f"{r.get('flops_remaining_G',0.0):>9.1f} "
                      f"{r.get('flop_reduction_pct',0.0):>5.1f}%")
        print("-" * 120)

    # ------------------------------------------------------------------
    # 11. Best cascade per head_sparsity
    # ------------------------------------------------------------------
    print("\n  Best cascade combo per head-sparsity (by Dice):")
    for head_sp in args.head_sparsities:
        cascade_rows = [r for r in all_results["cascade_results"]
                        if r["head_sparsity"] == head_sp and r["phase"] == "cascade"]
        if not cascade_rows:
            continue
        best = max(cascade_rows, key=lambda x: x["mean_dice"])
        # Compare to head-only
        head_only_row = next((r for r in all_results["cascade_results"]
                              if r["head_sparsity"] == head_sp
                              and r["phase"] == "head_only"), None)
        head_only_dice = head_only_row["mean_dice"] if head_only_row else float("nan")
        gap = best["mean_dice"] - head_only_dice
        print(f"    Head={head_sp*100:.0f}%  best=[{best['mlp_method']} "
              f"mlp_sp={best['mlp_sparsity']*100:.0f}%]  "
              f"Dice={best['mean_dice']:.4f}  "
              f"vs head-only={head_only_dice:.4f}  "
              f"gap={gap:+.4f}  "
              f"Params↓{best.get('param_reduction_pct',0.0):.1f}%  "
              f"FLOPs↓{best.get('flop_reduction_pct',0.0):.1f}%")

    print("\n" + "=" * 72)
    print(f"Cascade experiment complete. Results: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
