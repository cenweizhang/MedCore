"""
v8.5 pilot: h=0.4 / m=0.85, strict v8 comparison with [C]+[D] recovery only.

Changes vs v8 (strict control):
  protected_blocks = [10, 11]           same as v8 default
  cal_sizes        = [64, 38, 26]       same as v8 default  ← was [64, 32, 32]
  dist_beta        = 0.3                same as v8 default  ← was 0.1
  [C] boundary_freq_gamma   = 1.0      (omega_freq-adaptive lambda_b in recovery)
  [D] contour_loss_weight   = 1.0      (L_contour = MSE(|∇σ(pred)|, boundary_map))

v8 reference at h=0.4, m=0.85 (no recovery): BF1=0.4536  ΔBF1=-0.0785  Par↓62.8%

Usage:
    python -m pilot_dual.run_cascade_v8p5 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
        --data_roots  asserts/kvasir-seg/Kvasir-SEG asserts/CVC-ColonDB asserts/CVC-ClinicDB \
        --device cuda:0 \
        --output_dir results/pilot_cascade_v8p5 \
        --recovery_steps 100
"""
import os, sys, json, copy, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry
from pilot_dual.run_cascade_v8 import (
    build_multi_dataset_loaders,
    compute_diagonal_fisher_boundary_aware,
    compute_sam_fisher_on_medical,
    sum_fisher_dicts,
    compute_scores_per_subset_v7,
    compute_adaptive_alpha_per_block,
    combine_scores_dist_adaptive,
    eval_all_datasets,
)
from pilot_dual.scoring import load_sam_encoder_params
from pilot_dual.pruning import (
    compute_block_sensitivity,
    allocate_nonuniform_head_sparsity, allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform, generate_neuron_mask_nonuniform,
    apply_head_mask_to_model, apply_mlp_mask_to_model, remove_hooks,
)
from pilot_dual.recovery import recovery_finetune

# ── fixed config ─────────────────────────────────────────────────────────────
HEAD_SP     = 0.4
MLP_SP      = 0.85
DATASET_NAMES = ["Kvasir", "ColonDB", "ClinicDB"]
CAL_SIZES   = [64, 38, 26]          # per-dataset calibration sizes (same as v8 default)
PI_R        = [0.5, 0.3, 0.2]       # clinical importance weights

# same as v8 — strict control for [C]+[D] comparison
PROTECTED   = [10, 11]

# [B] removed — keep v8 default to preserve adaptive-alpha discriminability
BOUNDARY_FISHER_WEIGHT = 3.0

# [C] adaptive lambda_b in recovery
BOUNDARY_FREQ_GAMMA = 1.0

# [D] contour consistency loss
CONTOUR_LOSS_WEIGHT = 1.0


def _json_safe(obj):
    if isinstance(obj, dict):   return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt", required=True)
    p.add_argument("--sam_ckpt",    required=True)
    p.add_argument("--data_roots",  nargs=3,
                   default=["asserts/Kvasir-SEG", "asserts/CVC-ColonDB", "asserts/CVC-ClinicDB"])
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--output_dir",  default="results/pilot_cascade_v8p5")
    p.add_argument("--eval_batch",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    # recovery
    p.add_argument("--recovery_steps",         type=int,   default=100)
    p.add_argument("--recovery_lr",            type=float, default=1e-5)
    p.add_argument("--feat_distill_weight",    type=float, default=0.5)
    p.add_argument("--boundary_loss_weight",   type=float, default=1.0)
    p.add_argument("--logit_distill_weight",   type=float, default=2.0)
    p.add_argument("--freq_pred_loss_weight",  type=float, default=0.5)
    p.add_argument("--contour_loss_weight",    type=float, default=CONTOUR_LOSS_WEIGHT)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    pi_r   = np.array(PI_R, dtype=np.float32)
    pi_r  /= pi_r.sum()

    print(f"\n{'='*60}")
    print(f"v8.5 pilot  h={HEAD_SP:.0%}  m={MLP_SP:.0%}  (strict v8 comparison)")
    print(f"  protected_blocks      = {PROTECTED}  (same as v8)")
    print(f"  cal_sizes             = {CAL_SIZES}  (same as v8)")
    print(f"  dist_beta             = 0.3  (same as v8)")
    print(f"  [C] boundary_freq_gamma   = {BOUNDARY_FREQ_GAMMA}")
    print(f"  [D] contour_loss_weight   = {CONTOUR_LOSS_WEIGHT}")
    print(f"  v8 reference (no recovery): BF1=0.4536  ΔBF1=-0.0785  Par↓62.8%")
    print(f"{'='*60}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("[1] Building multi-dataset loaders ...")
    (per_ds_cal_loaders, combined_cal_loader,
     per_ds_test_loaders, _, combined_cal_freq_weights) = build_multi_dataset_loaders(
        data_roots=args.data_roots,
        dataset_names=DATASET_NAMES,
        cal_sizes=CAL_SIZES,
        eval_batch_size=args.eval_batch,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"  cal_freq_weights: mean={combined_cal_freq_weights.mean():.3f}  "
          f"std={combined_cal_freq_weights.std():.3f}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print("[2] Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device)
    original_state = copy.deepcopy(model.state_dict())

    # ── 3. Baseline eval ─────────────────────────────────────────────────────
    print("[3] Baseline evaluation ...")
    model.eval()
    baseline = eval_all_datasets(model, per_ds_test_loaders, DATASET_NAMES, device)
    print(f"  macro BF1={baseline['macro']['mean_boundary_f1']:.4f}  "
          f"Dice={baseline['macro']['mean_dice']:.4f}")
    for ds in DATASET_NAMES:
        m = baseline["per_dataset"][ds]
        print(f"    {ds}: BF1={m['mean_boundary_f1']:.4f}  Dice={m['mean_dice']:.4f}")

    # ── 4. MedSAM Fisher [B: boundary_weight=5.0] ────────────────────────────
    print(f"\n[4] MedSAM boundary-aware Fisher  (boundary_weight={BOUNDARY_FISHER_WEIGHT}) ...")
    t0 = time.time()
    fisher_m_list = []
    for i, (loader, name) in enumerate(zip(per_ds_cal_loaders, DATASET_NAMES)):
        print(f"    F^M [{name}] n={len(loader.dataset)} ...")
        fisher_m_list.append(
            compute_diagonal_fisher_boundary_aware(
                model, loader, device, boundary_weight=BOUNDARY_FISHER_WEIGHT))
    print(f"  F^M done: {time.time()-t0:.1f}s")

    # SAM cross-Fisher
    print("[4b] SAM cross-Fisher ...")
    t0 = time.time()
    fisher_s = compute_sam_fisher_on_medical(
        args.sam_ckpt, combined_cal_loader, device,
        boundary_weight=BOUNDARY_FISHER_WEIGHT)
    cross_fisher_list = [
        {n: torch.sqrt(fm[n].float() * fisher_s.get(n, torch.zeros_like(fm[n])).float() + 1e-30)
         for n in fm}
        for fm in fisher_m_list
    ]
    del fisher_s
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  Cross-Fisher done: {time.time()-t0:.1f}s")

    # ── 5. Scoring ────────────────────────────────────────────────────────────
    print("[5] Computing per-subset scores ...")
    mlp_dim    = 3072
    sam_params = load_sam_encoder_params(args.sam_ckpt, device="cpu")

    dz_head_list, dr_head_list, dz_mlp_list, dr_mlp_list = compute_scores_per_subset_v7(
        model, sam_params, fisher_m_list, cross_fisher_list)

    # pi_r-weighted mean (shape 144) — required by compute_adaptive_alpha_per_block
    R = len(dz_head_list)
    dz_head_mean = sum(float(pi_r[i]) * dz_head_list[i] for i in range(R)).astype(np.float32)
    dr_head_mean = sum(float(pi_r[i]) * dr_head_list[i] for i in range(R)).astype(np.float32)

    alpha_pb = compute_adaptive_alpha_per_block(dz_head_mean, dr_head_mean)
    print(f"  alpha_per_block: {np.round(alpha_pb, 3).tolist()}")

    dist_beta = 0.3
    head_scores = combine_scores_dist_adaptive(
        dz_head_list, dr_head_list, alpha_pb,
        items_per_block=12, pi=pi_r, beta=dist_beta)
    mlp_scores = combine_scores_dist_adaptive(
        dz_mlp_list, dr_mlp_list, alpha_pb,
        items_per_block=mlp_dim, pi=pi_r, beta=dist_beta)

    # block sensitivity (π_r-weighted Fisher)
    fisher_combined = sum_fisher_dicts(fisher_m_list, pi=pi_r)
    block_sens = compute_block_sensitivity(fisher_combined, num_blocks=12)
    print(f"  block_sensitivity: {[f'{s:.2e}' for s in block_sens]}")

    np.savez(os.path.join(args.output_dir, "scores.npz"),
             delta_zero_head=np.stack(dz_head_list).mean(0),
             delta_reset_head=np.stack(dr_head_list).mean(0),
             delta_zero_mlp=np.stack(dz_mlp_list).mean(0),
             delta_reset_mlp=np.stack(dr_mlp_list).mean(0),
             dz_head_per_subset=np.stack(dz_head_list),
             dr_head_per_subset=np.stack(dr_head_list),
             dz_mlp_per_subset=np.stack(dz_mlp_list),
             dr_mlp_per_subset=np.stack(dr_mlp_list),
             block_sensitivity=block_sens,
             alpha_per_block=alpha_pb,
             pi_r=pi_r,
             dist_beta=np.float32(dist_beta))

    # ── 6. Masks (same as v8: protected_blocks=[10,11]) ──────────────────────
    print(f"\n[6] Generating masks  h={HEAD_SP:.0%} m={MLP_SP:.0%}  "
          f"protected={PROTECTED} ...")

    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sens, HEAD_SP, num_heads=12, min_keep=1,
        protected_blocks=PROTECTED)
    head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
    print(f"  head_mask: {int(head_mask.sum())}/144 kept "
          f"(actual sp {1-head_mask.mean():.1%})")

    per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
        block_sens, MLP_SP, mlp_dim=mlp_dim, min_frac=0.05,
        protected_blocks=PROTECTED)
    neuron_mask = generate_neuron_mask_nonuniform(
        mlp_scores, per_block_sp_mlp, mlp_dim=mlp_dim)
    print(f"  mlp_mask:  {int(neuron_mask.sum())}/{12*mlp_dim} kept "
          f"(actual sp {1-neuron_mask.mean():.1%})")

    # ── 7. Apply masks + eval before recovery ────────────────────────────────
    model.load_state_dict(original_state)
    h_hooks = apply_head_mask_to_model(model, head_mask)
    m_hooks = apply_mlp_mask_to_model(model, neuron_mask)

    print("[7] Eval before recovery ...")
    pre_rec = eval_all_datasets(model, per_ds_test_loaders, DATASET_NAMES, device)
    print(f"  macro BF1={pre_rec['macro']['mean_boundary_f1']:.4f}  "
          f"Dice={pre_rec['macro']['mean_dice']:.4f}")
    for ds in DATASET_NAMES:
        m = pre_rec["per_dataset"][ds]
        bl = baseline["per_dataset"][ds]
        db = m["mean_boundary_f1"] - bl["mean_boundary_f1"]
        print(f"    {ds}: BF1={m['mean_boundary_f1']:.4f} ({db:+.4f})")

    # ── 8. Recovery [C: boundary_freq_gamma=1.0, D: contour_loss] ───────────
    print(f"\n[8] Recovery fine-tune  steps={args.recovery_steps}  "
          f"boundary_freq_gamma={BOUNDARY_FREQ_GAMMA}  "
          f"contour_loss_weight={args.contour_loss_weight} ...")
    feature_teacher = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).cpu().eval()
    for p in feature_teacher.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    recovery_finetune(
        model, combined_cal_loader, device,
        n_steps=args.recovery_steps,
        lr=args.recovery_lr,
        feature_teacher=feature_teacher,
        feat_distill_weight=args.feat_distill_weight,
        boundary_loss_weight=args.boundary_loss_weight,
        logit_distill_weight=args.logit_distill_weight,
        freq_pred_loss_weight=args.freq_pred_loss_weight,
        freq_weights=combined_cal_freq_weights,
        boundary_freq_gamma=BOUNDARY_FREQ_GAMMA,   # [C]
        contour_loss_weight=args.contour_loss_weight,  # [D]
    )
    del feature_teacher
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  Recovery done: {time.time()-t0:.1f}s")

    # ── 9. Final eval ─────────────────────────────────────────────────────────
    print("[9] Final evaluation ...")
    final = eval_all_datasets(model, per_ds_test_loaders, DATASET_NAMES, device)

    print(f"\n{'='*60}  SUMMARY")
    print(f"{'Config':>24}   BF1     Dice    HD95")
    print(f"  {'Baseline':>20}  {baseline['macro']['mean_boundary_f1']:.4f}  "
          f"{baseline['macro']['mean_dice']:.4f}  {baseline['macro']['mean_hd95']:.1f}")
    print(f"  {'Before recovery':>20}  {pre_rec['macro']['mean_boundary_f1']:.4f}  "
          f"{pre_rec['macro']['mean_dice']:.4f}  {pre_rec['macro']['mean_hd95']:.1f}")
    print(f"  {'After recovery':>20}  {final['macro']['mean_boundary_f1']:.4f}  "
          f"{final['macro']['mean_dice']:.4f}  {final['macro']['mean_hd95']:.1f}")
    print()
    for ds in DATASET_NAMES:
        bl = baseline["per_dataset"][ds]
        af = final["per_dataset"][ds]
        db = af["mean_boundary_f1"] - bl["mean_boundary_f1"]
        dd = af["mean_dice"]        - bl["mean_dice"]
        print(f"    {ds:>10}: BF1={af['mean_boundary_f1']:.4f}({db:+.4f})  "
              f"Dice={af['mean_dice']:.4f}({dd:+.4f})")
    print(f"{'='*60}\n")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    results = {
        "version": "v8.5",
        "config": {
            "head_sp": HEAD_SP, "mlp_sp": MLP_SP,
            "protected_blocks": PROTECTED,
            "boundary_fisher_weight": BOUNDARY_FISHER_WEIGHT,
            "boundary_freq_gamma": BOUNDARY_FREQ_GAMMA,
            "contour_loss_weight": args.contour_loss_weight,
            "pi_r": PI_R,
        },
        "baseline":         baseline,
        "before_recovery":  pre_rec,
        "after_recovery":   final,
        "delta_bf1_macro":  final["macro"]["mean_boundary_f1"] - baseline["macro"]["mean_boundary_f1"],
        "delta_dice_macro": final["macro"]["mean_dice"]        - baseline["macro"]["mean_dice"],
        "head_mask_kept":   int(head_mask.sum()),
        "mlp_mask_kept":    int(neuron_mask.sum()),
    }
    out_path = os.path.join(args.output_dir, "results_v8p5.json")
    with open(out_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"Results saved → {out_path}")

    remove_hooks(h_hooks + m_hooks)


if __name__ == "__main__":
    main()
