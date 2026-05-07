# -*- coding: utf-8 -*-
"""
Cascade pruning v5: Nonuniform-only with enhanced boundary protection.

Improvements vs v4 (run_cascade_v4.py)
----------------------------------------
1. Nonuniform-only : uniform method dropped; nonuniform is strictly better at
   heavy sparsity and avoids the "multi-objective tug-of-war" problem that
   made v4 uniform worse than v3 at >=90% MLP sparsity.

2. Hard block protection : --protected_blocks 10 11 (default).  Blocks 10-11
   are the closest transformer layers to the mask decoder and have the highest
   Fisher sensitivity.  Never pruning them preserves the spatial boundary
   signals that the decoder relies on.

3. Rebalanced boundary losses :
      boundary_loss_weight  : 2.0 → 1.0  (reduce competition with Dice)
      logit_distill_weight  : 1.0 → 2.0  (stronger direct boundary signal)

4. Frequency-domain prediction loss (new) :
      L_freq = Σ_ω |FFT(σ(student)) − FFT(GT)| · w(ω)
   where w(ω)=1 for ω > low_cutoff (high spatial frequencies) and 0 otherwise.
   Boundary pixels live in high frequencies, so this actively penalises missing
   fine boundary detail.  Complementary to spatial-domain distillation.

5. Improved frequency-weighted sampling :
   Sampling weight changed from raw FFT energy ratio to boundary complexity
      complexity = perimeter / √area
   This better identifies truly hard, complex-boundary samples.

Usage
-----
    cd /volume/med-train/users/jshan/testSAMpruning
    /home/zhangcenwei/miniconda3/envs/medsam/bin/python -m pilot_dual.run_cascade_v5 \\
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \\
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \\
        --data_root   assert/CVC-ColonDB \\
        --device      cuda:1 \\
        --output_dir  results/pilot_cascade_v5 \\
        --recovery_steps 100 --recovery_lr 1e-5 \\
        --feat_distill_weight 0.5 \\
        --boundary_loss_weight 1.0 \\
        --logit_distill_weight 2.0 \\
        --freq_pred_loss_weight 0.5 \\
        --protected_blocks 10 11
"""

import os
import sys
import copy
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset  import build_dataloaders

from pilot_dual.scoring import (
    compute_diagonal_fisher,
    load_sam_encoder_params,
    compute_head_scores,
    compute_mlp_neuron_scores,
    combine_scores,
    score_summary,
    compute_head_costs,
    compute_neuron_costs,
)
from pilot_dual.pruning import (
    generate_head_mask_constrained,
    generate_neuron_mask_constrained,
    apply_head_mask_to_model,
    apply_mlp_mask_to_model,
    remove_hooks,
    compute_cascade_stats,
    compute_block_sensitivity,
    allocate_nonuniform_head_sparsity,
    allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform,
    generate_neuron_mask_nonuniform,
)
from pilot_dual.recovery import recovery_finetune
from pilot_phase1.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Frequency-weighted sampler helper  (v5: boundary complexity)
# ---------------------------------------------------------------------------

def compute_cal_freq_weights(cal_loader):
    """
    Per-sample boundary complexity weight for the calibration set.

    Uses  complexity = perimeter / sqrt(area)  instead of raw FFT energy
    ratio (v4).  This captures the true geometric complexity of the GT mask
    boundary: thin / elongated / jagged contours get higher weights.

    Returns:
        weights : np.ndarray of shape (n_cal,), values in (0, ∞).
    """
    k3 = torch.ones(1, 1, 3, 3)     # 3×3 kernel for dilation/erosion
    ordered_loader = DataLoader(
        cal_loader.dataset,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
    )
    weights = []
    for batch in ordered_loader:
        mask = batch["mask_256"][0, 0].float()       # (256, 256)
        m4d  = mask[None, None]                      # (1,1,256,256)
        dilated  = F.conv2d(m4d, k3, padding=1).clamp(0, 1)
        eroded   = 1.0 - F.conv2d(1.0 - m4d, k3, padding=1).clamp(0, 1)
        boundary = (dilated - eroded).squeeze()      # (256,256) boundary pixels
        perimeter   = boundary.sum().item()
        area        = mask.sum().item()
        complexity  = perimeter / (area ** 0.5 + 1.0)
        weights.append(max(complexity, 1e-4))
    return np.array(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_and_collect(model, test_loader, device, collect_indices=None):
    model.eval()
    collect_indices = set(collect_indices) if collect_indices is not None else set()
    all_metrics, collected = [], {}

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
        images   = batch["image"].to(device)
        masks_gt = batch["mask_1024"].numpy()
        bboxes   = batch["bbox"].numpy()

        image_emb = model.image_encoder(images)
        box_t = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
        if box_t.dim() == 2:
            box_t = box_t[:, None, :]
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_t, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred_1024 = F.interpolate(low_res, size=(1024, 1024),
                                  mode="bilinear", align_corners=False)
        pred_bin  = (torch.sigmoid(pred_1024) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        gt_bin    = masks_gt[0].astype(np.uint8)
        m         = compute_all_metrics(pred_bin, gt_bin)
        m["name"] = batch["name"][0]
        all_metrics.append(m)

        if i in collect_indices:
            img_np = batch["image"][0].permute(1, 2, 0).cpu().numpy()
            collected[i] = {
                "pred": _resize256(pred_bin.astype(np.float32), order=0),
                "gt":   _resize256(gt_bin.astype(np.float32),   order=0),
                "img":  (_resize256(img_np, order=1) * 255).astype(np.uint8),
                "name": batch["name"][0],
                "dice": m["dice"],
                "bf1":  m["boundary_f1"],
            }

    keys = ["dice", "iou", "boundary_f1", "hd95"]
    avg  = {f"mean_{k}": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    avg.update({f"std_{k}": float(np.std([m[k] for m in all_metrics])) for k in keys})
    avg["head_sparsity"] = 0.0
    avg["kept_heads"]    = 144
    avg["total_heads"]   = 144
    return avg, all_metrics, collected


def _resize256(arr, order=0):
    from skimage.transform import resize as sk_resize
    if arr.ndim == 2:
        return sk_resize(arr, (256, 256), order=order,
                         preserve_range=True, anti_aliasing=False)
    return sk_resize(arr, (256, 256, arr.shape[2]), order=order,
                     preserve_range=True, anti_aliasing=(order > 0))


def _json_safe(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def _print_row(head_sp, mlp_sp, mlp_meth, m, stats):
    p  = stats.get("param_reduction_pct", 0.0)
    fl = stats.get("flops_remaining_G",   0.0)
    fr = stats.get("flop_reduction_pct",  0.0)
    print(f"    [nonunif+{mlp_meth}] sp_head={head_sp*100:.0f}% "
          f"sp_mlp={mlp_sp*100:.0f}% | "
          f"Dice={m['mean_dice']:.4f}  BF1={m['mean_boundary_f1']:.4f}  "
          f"HD95={m['mean_hd95']:.2f}  "
          f"Params↓{p:.1f}%  FLOPs={fl:.1f}G↓{fr:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cascade head→MLP pruning v5 (nonuniform-only + enhanced boundary)"
    )
    parser.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",     default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_root",    default="assert/CVC-ColonDB")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_cal",        type=int, default=128)
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output_dir",   default="results/pilot_cascade_v5")

    # Sparsity grid
    parser.add_argument("--head_sparsities", type=float, nargs="+",
                        default=[0.5, 0.7, 0.8])
    parser.add_argument("--mlp_sparsities",  type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 0.9, 0.95])

    # Scoring
    parser.add_argument("--phase1_alpha",     type=float, default=1.0)
    parser.add_argument("--mlp_alpha_values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--tau",              type=float, default=0.0)
    parser.add_argument("--recompute_fisher_phase2", action="store_true")

    # Recovery (base)
    parser.add_argument("--recovery_steps",       type=int,   default=0)
    parser.add_argument("--recovery_lr",          type=float, default=1e-5)
    parser.add_argument("--feat_distill_weight",  type=float, default=0.5)

    # Boundary losses (v4 adjusted)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0,
                        help="λ for boundary-weighted BCE (reduced from v4's 2.0).")
    parser.add_argument("--logit_distill_weight", type=float, default=2.0,
                        help="Boundary-region logit distillation weight (increased from v4's 1.0).")

    # Frequency-domain loss (v5 new)
    parser.add_argument("--freq_pred_loss_weight", type=float, default=0.5,
                        help="Weight for frequency-domain prediction loss L_freq. "
                             "0=disable. Penalises high-frequency (boundary) errors.")
    parser.add_argument("--no_freq_sampling", action="store_true",
                        help="Disable boundary-complexity-weighted sampling.")

    # FT-full baseline
    parser.add_argument("--ft_baseline_steps", type=int, default=0)

    # Structural protection
    parser.add_argument("--protected_blocks",         type=int, nargs="*",
                        default=[10, 11],
                        help="Blocks never pruned. Default: last 2 blocks (10,11).")
    parser.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    parser.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("CASCADE PRUNING v5  (nonuniform-only + enhanced boundary)")
    print(f"  Data          : {args.data_root}   n_cal={args.n_cal}")
    print(f"  Head sp       : {args.head_sparsities}  MLP sp: {args.mlp_sparsities}")
    print(f"  Recovery      : {args.recovery_steps} steps  lr={args.recovery_lr}")
    print(f"  feat_distill  : {args.feat_distill_weight}")
    print(f"  boundary_loss : {args.boundary_loss_weight}")
    print(f"  logit_distill : {args.logit_distill_weight}")
    print(f"  freq_pred_loss: {args.freq_pred_loss_weight}")
    print(f"  freq_sampling : {'off' if args.no_freq_sampling else 'on (complexity-weighted)'}")
    print(f"  protected_blk : {args.protected_blocks}")
    print(f"  Output        : {args.output_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load MedSAM
    # ------------------------------------------------------------------
    print("\n[1] Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model = model.to(device).eval()
    for p in model.prompt_encoder.parameters(): p.requires_grad_(False)
    for p in model.mask_decoder.parameters():   p.requires_grad_(False)
    original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    # 2. SAM reference params
    # ------------------------------------------------------------------
    print("\n[2] Loading SAM params ...")
    sam_params = load_sam_encoder_params(args.sam_ckpt, device="cpu")

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
    print(f"  Cal: {args.n_cal}  Test: {n_test}")

    # ------------------------------------------------------------------
    # 3b. Boundary-complexity sampling weights (v5 improved)
    # ------------------------------------------------------------------
    cal_freq_weights = None
    if args.recovery_steps > 0 and not args.no_freq_sampling:
        print("\n[3b] Computing boundary-complexity sampling weights ...")
        cal_freq_weights = compute_cal_freq_weights(cal_loader)
        print(f"  Complexity: mean={cal_freq_weights.mean():.3f}  "
              f"min={cal_freq_weights.min():.3f}  max={cal_freq_weights.max():.3f}")
        np.save(os.path.join(args.output_dir, "cal_freq_weights.npy"), cal_freq_weights)

    # ------------------------------------------------------------------
    # 4. Baseline evaluation
    # ------------------------------------------------------------------
    print("\n[4] Baseline (unpruned) evaluation ...")
    baseline_metrics, _, _ = _eval_and_collect(model, test_loader, device)
    b = baseline_metrics
    print(f"  Dice={b['mean_dice']:.4f}  BF1={b['mean_boundary_f1']:.4f}  "
          f"IoU={b['mean_iou']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ------------------------------------------------------------------
    # 4b. FT-full baseline
    # ------------------------------------------------------------------
    ft_full_metrics = None
    if args.ft_baseline_steps > 0:
        print(f"\n[4b] FT-full baseline: {args.ft_baseline_steps} steps ...")
        ft_model = copy.deepcopy(model)
        recovery_finetune(
            ft_model, cal_loader, device,
            n_steps=args.ft_baseline_steps, lr=args.recovery_lr,
            boundary_loss_weight=args.boundary_loss_weight,
            freq_pred_loss_weight=args.freq_pred_loss_weight,
            freq_weights=cal_freq_weights,
        )
        ft_full_metrics, _, _ = _eval_and_collect(ft_model, test_loader, device)
        del ft_model; torch.cuda.empty_cache()
        f = ft_full_metrics
        print(f"  FT-full: Dice={f['mean_dice']:.4f}  BF1={f['mean_boundary_f1']:.4f}  "
              f"HD95={f['mean_hd95']:.2f}  "
              f"(Dice boost: {f['mean_dice']-b['mean_dice']:+.4f})")

    # ------------------------------------------------------------------
    # 5. Diagonal Fisher
    # ------------------------------------------------------------------
    print("\n[5] Computing diagonal Fisher ...")
    t0 = time.time()
    fisher = compute_diagonal_fisher(model, cal_loader, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    block_sensitivity = compute_block_sensitivity(fisher, num_blocks=12)
    np.save(os.path.join(args.output_dir, "block_sensitivity.npy"), block_sensitivity)
    print(f"  Block sensitivity: min={block_sensitivity.min():.3e}  "
          f"max={block_sensitivity.max():.3e}  "
          f"argmax={int(block_sensitivity.argmax())}  "
          f"protected={args.protected_blocks}")

    # ------------------------------------------------------------------
    # 6. Head scores (Phase-1)
    # ------------------------------------------------------------------
    print("\n[6] Computing head scores ...")
    dz_head, dr_head = compute_head_scores(model, sam_params, fisher)
    head_summary     = score_summary(dz_head, dr_head)
    head_costs  = compute_head_costs(model) if args.tau > 0 else None
    head_scores = combine_scores(dz_head, dr_head,
                                  alpha=args.phase1_alpha,
                                  cost=head_costs, tau=args.tau)

    # ------------------------------------------------------------------
    # 7. MLP neuron scores
    # ------------------------------------------------------------------
    print("\n[7] Computing MLP neuron scores ...")
    dz_mlp, dr_mlp = compute_mlp_neuron_scores(model, sam_params, fisher)
    mlp_summary     = score_summary(dz_mlp, dr_mlp)
    neuron_costs = compute_neuron_costs(model) if args.tau > 0 else None

    # ------------------------------------------------------------------
    # 8. Feature / logit teacher (CPU-offloaded)
    # ------------------------------------------------------------------
    feature_teacher = None
    need_teacher = (args.recovery_steps > 0 and
                    (args.feat_distill_weight > 0 or args.logit_distill_weight > 0))
    if need_teacher:
        print("\n[8] Building teacher (CPU-offloaded) ...")
        feature_teacher = copy.deepcopy(model).cpu()
        feature_teacher.eval()
        for p in feature_teacher.parameters():
            p.requires_grad_(False)
        print(f"  Teacher: {sum(p.numel() for p in feature_teacher.parameters())/1e6:.1f}M params on CPU")

    np.savez(
        os.path.join(args.output_dir, "scores.npz"),
        delta_zero_head=dz_head, delta_reset_head=dr_head,
        delta_zero_mlp=dz_mlp,  delta_reset_mlp=dr_mlp,
        block_sensitivity=block_sensitivity,
    )

    # ------------------------------------------------------------------
    # 9. Cascade experiment loop  (nonuniform only)
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "cascade_results_v5.json")

    def _save(obj):
        with open(results_path, "w") as _f:
            json.dump(json.loads(json.dumps(obj, default=_json_safe)), _f, indent=2)

    print("\n[9] Running cascade experiments (nonuniform only) ...")
    all_results = {
        "config":             vars(args),
        "baseline":           baseline_metrics,
        "ft_full_baseline":   ft_full_metrics,
        "head_score_summary": head_summary,
        "mlp_score_summary":  mlp_summary,
        "block_sensitivity":  block_sensitivity.tolist(),
        "cascade_results":    [],
    }

    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    for head_sp in args.head_sparsities:
        print(f"\n{'='*60}")
        print(f"  Head sparsity target: {head_sp*100:.0f}%  "
              f"(protected_blocks={args.protected_blocks})")

        per_block_sp_head = allocate_nonuniform_head_sparsity(
            block_sensitivity, head_sp,
            num_heads=12, min_keep=args.nonuniform_min_keep_head,
            protected_blocks=args.protected_blocks,
        )
        head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)

        n_heads_kept   = int(head_mask.sum())
        actual_head_sp = 1.0 - n_heads_kept / 144.0
        print(f"  Nonuniform head mask: kept={n_heads_kept}/144  "
              f"actual_sp={actual_head_sp*100:.1f}%")

        # Restore pristine weights
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        head_hooks = apply_head_mask_to_model(model, head_mask)

        # Phase-2 Fisher re-estimation (optional)
        if args.recompute_fisher_phase2:
            print(f"    [Phase-2 Fisher] Re-estimating ...")
            fisher_p2             = compute_diagonal_fisher(model, cal_loader, device)
            dz_mlp_p2, dr_mlp_p2 = compute_mlp_neuron_scores(model, sam_params, fisher_p2)
            block_sensitivity_p2  = compute_block_sensitivity(fisher_p2)
        else:
            fisher_p2, dz_mlp_p2, dr_mlp_p2 = fisher, dz_mlp, dr_mlp
            block_sensitivity_p2             = block_sensitivity

        # Phase-1 recovery
        if args.recovery_steps > 0:
            print(f"    [Phase-1 recovery] {args.recovery_steps} steps ...")
            recovery_finetune(
                model, cal_loader, device,
                head_mask=head_mask, neuron_mask=None,
                n_steps=args.recovery_steps, lr=args.recovery_lr,
                feature_teacher=feature_teacher,
                feat_distill_weight=args.feat_distill_weight,
                boundary_loss_weight=args.boundary_loss_weight,
                logit_distill_weight=args.logit_distill_weight,
                freq_pred_loss_weight=args.freq_pred_loss_weight,
                freq_weights=cal_freq_weights,
            )

        # Evaluate head-only
        head_only_metrics, _, _ = _eval_and_collect(model, test_loader, device)
        head_only_stats         = compute_cascade_stats(model, head_mask, None)
        print(f"    Head-only: Dice={head_only_metrics['mean_dice']:.4f}  "
              f"BF1={head_only_metrics['mean_boundary_f1']:.4f}  "
              f"HD95={head_only_metrics['mean_hd95']:.2f}  "
              f"Params↓{head_only_stats['param_reduction_pct']:.1f}%")

        all_results["cascade_results"].append({
            "phase":          "head_only",
            "head_sp_target": head_sp,
            "head_sp_actual": actual_head_sp,
            "mlp_method":     None,
            "mlp_sp_target":  None,
            "mlp_sp_actual":  None,
            "mlp_alpha":      None,
            **head_only_metrics,
            **head_only_stats,
        })

        # Save post-Phase-1 state
        post_p1_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # ---- Inner MLP loop (nonuniform only) ----
        for mlp_sp in args.mlp_sparsities:
            print(f"\n    == MLP target sparsity {mlp_sp*100:.0f}% ==")

            per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
                block_sensitivity_p2, mlp_sp,
                mlp_dim=mlp_dim,
                min_frac=args.nonuniform_min_frac_mlp,
                protected_blocks=args.protected_blocks,
            )

            for mlp_alpha in args.mlp_alpha_values:
                if   mlp_alpha == 0.0: mlp_score_name = "reset_only"
                elif mlp_alpha == 1.0: mlp_score_name = "zero_only"
                else:                  mlp_score_name = f"dual_a{mlp_alpha:.1f}"

                q_mlp = combine_scores(
                    dz_mlp_p2, dr_mlp_p2,
                    alpha=mlp_alpha,
                    cost=neuron_costs, tau=args.tau,
                )

                # Restore post-Phase-1 state
                model.load_state_dict(
                    {k: v.to(device) for k, v in post_p1_state.items()}
                )

                nm = generate_neuron_mask_nonuniform(
                    q_mlp, per_block_sp_mlp, mlp_dim=mlp_dim
                )
                actual_mlp_sp = 1.0 - float(nm.sum()) / (12 * mlp_dim)

                mlp_hooks = apply_mlp_mask_to_model(model, nm)

                # Phase-2 recovery
                if args.recovery_steps > 0:
                    recovery_finetune(
                        model, cal_loader, device,
                        head_mask=None, neuron_mask=None,
                        n_steps=args.recovery_steps, lr=args.recovery_lr,
                        feature_teacher=feature_teacher,
                        feat_distill_weight=args.feat_distill_weight,
                        boundary_loss_weight=args.boundary_loss_weight,
                        logit_distill_weight=args.logit_distill_weight,
                        freq_pred_loss_weight=args.freq_pred_loss_weight,
                        freq_weights=cal_freq_weights,
                    )

                metrics, _, _ = _eval_and_collect(model, test_loader, device)
                stats         = compute_cascade_stats(model, head_mask, nm)

                remove_hooks(mlp_hooks)

                all_results["cascade_results"].append({
                    "phase":          "cascade",
                    "head_sp_target": head_sp,
                    "head_sp_actual": actual_head_sp,
                    "mlp_method":     "nonuniform",
                    "mlp_sp_target":  mlp_sp,
                    "mlp_sp_actual":  actual_mlp_sp,
                    "mlp_score":      mlp_score_name,
                    "mlp_alpha":      mlp_alpha,
                    **metrics,
                    **stats,
                })
                _print_row(head_sp, mlp_sp, mlp_score_name, metrics, stats)

        remove_hooks(head_hooks)

        # Flush after each head_sp iteration
        _save(all_results)
        print(f"  [checkpoint] saved → {results_path}")

    # ------------------------------------------------------------------
    # 10. Final save + summary
    # ------------------------------------------------------------------
    _save(all_results)
    print(f"\n[10] Results saved to {results_path}")

    print("\n" + "=" * 120)
    print("v5 RESULT SUMMARY  —  Nonuniform-only + Enhanced Boundary")
    print(f"  protected_blocks={args.protected_blocks}  "
          f"boundary_loss={args.boundary_loss_weight}  "
          f"logit_distill={args.logit_distill_weight}  "
          f"freq_pred={args.freq_pred_loss_weight}")
    print("=" * 120)

    cascade = all_results["cascade_results"]
    print(f"\n  {'Config':<45} {'Dice':>6} {'BF1':>6} {'HD95':>7} {'Par%':>6} {'FL%':>6}")
    print("  " + "-" * 80)
    for r in cascade:
        if r["phase"] == "head_only":
            lbl = f"HEAD_ONLY h={r['head_sp_target']*100:.0f}%"
        else:
            lbl = (f"h={r['head_sp_target']*100:.0f}% "
                   f"m={r['mlp_sp_target']*100:.0f}%_{r.get('mlp_score','?')}")
        print(f"  {lbl:<45} "
              f"{r['mean_dice']:>6.4f} {r['mean_boundary_f1']:>6.4f} "
              f"{r['mean_hd95']:>7.2f} "
              f"{r.get('param_reduction_pct',0):>6.1f} "
              f"{r.get('flop_reduction_pct',0):>6.1f}")

    extreme = [r for r in cascade if r["phase"] == "cascade"
               and r["head_sp_target"] >= 0.7 and r["mlp_sp_target"] >= 0.85]
    if extreme:
        best_bf1  = max(extreme, key=lambda x: x["mean_boundary_f1"])
        best_hd95 = min(extreme, key=lambda x: x["mean_hd95"])
        print(f"\n  Best extreme BF1 : "
              f"h={best_bf1['head_sp_target']*100:.0f}% "
              f"m={best_bf1['mlp_sp_target']*100:.0f}%_{best_bf1.get('mlp_score','')}  "
              f"Dice={best_bf1['mean_dice']:.4f}  BF1={best_bf1['mean_boundary_f1']:.4f}  "
              f"HD95={best_bf1['mean_hd95']:.2f}  "
              f"Params↓{best_bf1.get('param_reduction_pct',0):.1f}%")
        print(f"  Best extreme HD95: "
              f"h={best_hd95['head_sp_target']*100:.0f}% "
              f"m={best_hd95['mlp_sp_target']*100:.0f}%_{best_hd95.get('mlp_score','')}  "
              f"Dice={best_hd95['mean_dice']:.4f}  BF1={best_hd95['mean_boundary_f1']:.4f}  "
              f"HD95={best_hd95['mean_hd95']:.2f}")

    print(f"\n  Baseline:  Dice={b['mean_dice']:.4f}  BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")
    if ft_full_metrics:
        f = ft_full_metrics
        print(f"  FT-full:   Dice={f['mean_dice']:.4f}  BF1={f['mean_boundary_f1']:.4f}  "
              f"HD95={f['mean_hd95']:.2f}")
    print("\n" + "=" * 80)
    print(f"Done. Results: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
