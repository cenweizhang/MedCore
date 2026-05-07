# -*- coding: utf-8 -*-
"""
Cascade pruning v6 : Boundary-aware Fisher + Iterative Prune-and-Recover +
                      Distribution-aware scoring.

Improvements vs v5
------------------
1. **Boundary-aware Fisher** (suggestion #1)
   Fisher estimation now uses a boundary-weighted BCE loss:
       w_ij = 1 + lambda_b * boundary_mask_ij
       L    = sum(w * BCE(logits, gt)) + dice_loss
   This makes Delta_zero and Delta_reset directly sensitive to boundary
   pixels, which BF1 depends on.

2. **Iterative Prune-and-Recover** (suggestion #2, paper's Algorithm 1)
   Instead of one-shot pruning per MLP sparsity level, v6 walks through
   `--mlp_sparsities` as progressive waypoints:
       50% -> recover -> 70% -> recover -> 85% -> recover -> 90% -> ...
   After each stage, optionally re-estimates Fisher on the pruned model
   (--recompute_fisher_between_stages) so that the next prune step sees
   up-to-date group importances.

3. **Distribution-aware scoring** (suggestion #5, paper Eq. 29-31)
   Split the calibration set into R subsets by boundary-complexity quantile
   (simple vs. complex boundaries).  Compute subset-specific Fisher and
   dual-intervention scores, then combine:
       Q_g^dist = sum_r pi_r * Q_g^(r) + beta * Var(Q_g^(1..R))
   A positive variance penalty (beta > 0) discourages pruning neurons that
   are important on complex-boundary samples but unimportant on easy ones.
   Set --n_dist_subsets 1 to disable (reverts to v5-style single-distribution).

Usage
-----
    cd /volume/med-train/users/jshan/testSAMpruning
    python -m pilot_dual.run_cascade_v6 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
        --data_root   asserts/CVC-ColonDB \
        --device      cuda:0 \
        --output_dir  results/v6_abl_all \
        --recovery_steps 100 --recovery_lr 1e-5 \
        --boundary_fisher_weight 3.0 \
        --n_dist_subsets 2 --dist_beta 0.3 \
        --recompute_fisher_between_stages \
        --mlp_sparsities 0.5 0.7 0.85 0.9 \
        --head_sparsities 0.5 0.7 \
        --mlp_alpha_values 1.0 \
        --protected_blocks 10 11

Ablation presets (isolate each improvement)
-------------------------------------------
    v5-equivalent (all 3 off) : --boundary_fisher_weight 0 --n_dist_subsets 1 --oneshot_mlp
    only #1 (boundary Fisher) : --boundary_fisher_weight 3 --n_dist_subsets 1 --oneshot_mlp
    only #2 (iterative)       : --boundary_fisher_weight 0 --n_dist_subsets 1
                                (omit --oneshot_mlp so chaining happens)
    only #5 (dist-aware)      : --boundary_fisher_weight 0 --n_dist_subsets 2 --dist_beta 0.3 --oneshot_mlp
    all three (default v6)    : --boundary_fisher_weight 3 --n_dist_subsets 2 --dist_beta 0.3
                                --recompute_fisher_between_stages
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.metrics import compute_all_metrics

from pilot_dual.scoring import (
    load_sam_encoder_params,
    compute_head_scores,
    compute_mlp_neuron_scores,
    combine_scores,
    score_summary,
    compute_head_costs,
    compute_neuron_costs,
)
from pilot_dual.pruning import (
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


# ---------------------------------------------------------------------------
# Local helpers (reused from v5, kept self-contained)
# ---------------------------------------------------------------------------

def _dice_loss_sum(pred_logits, target):
    pred = torch.sigmoid(pred_logits)
    pred   = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5
    )
    return per.sum()


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


@torch.no_grad()
def _eval_and_collect(model, test_loader, device):
    model.eval()
    all_metrics = []
    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
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
        m["name"] = batch["name"][0] if "name" in batch else ""
        all_metrics.append(m)

    keys = ["dice", "iou", "boundary_f1", "hd95"]
    avg = {f"mean_{k}": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    avg.update({f"std_{k}": float(np.std([m[k] for m in all_metrics])) for k in keys})
    avg["head_sparsity"] = 0.0
    avg["kept_heads"]    = 144
    avg["total_heads"]   = 144
    return avg, all_metrics, {}


def compute_cal_freq_weights(cal_loader):
    """Per-sample boundary complexity = perimeter / sqrt(area)."""
    k3 = torch.ones(1, 1, 3, 3)
    ordered_loader = DataLoader(
        cal_loader.dataset,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
    )
    weights = []
    for batch in ordered_loader:
        mask = batch["mask_256"][0, 0].float()
        m4d  = mask[None, None]
        dilated = F.conv2d(m4d, k3, padding=1).clamp(0, 1)
        eroded  = 1.0 - F.conv2d(1.0 - m4d, k3, padding=1).clamp(0, 1)
        boundary = (dilated - eroded).squeeze()
        perimeter  = boundary.sum().item()
        area       = mask.sum().item()
        complexity = perimeter / (area ** 0.5 + 1.0)
        weights.append(max(complexity, 1e-4))
    return np.array(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# IMPROVEMENT 1: Boundary-aware Fisher
# ---------------------------------------------------------------------------

def compute_diagonal_fisher_boundary_aware(model, dataloader, device,
                                            boundary_weight=3.0,
                                            num_blocks=12):
    """
    Diagonal Fisher with boundary-weighted BCE loss.

    For each sample, build a per-pixel weight map:
        w_ij = 1 + boundary_weight * B_ij
    where B is a 3x3 dilation-minus-erosion boundary mask of the GT.
    Loss becomes:
        L = sum(w * BCE(logits, gt)) + dice_loss
    This amplifies the contribution of boundary-adjacent pixels to the
    gradient, making Fisher directly reflect boundary sensitivity.

    boundary_weight=0 reduces to the standard Fisher in scoring.py.

    Args:
        model            : MedSAM model on `device`.
        dataloader       : calibration DataLoader (batch_size=1 recommended).
        device           : torch device.
        boundary_weight  : lambda_b >= 0. Typical: 3.0.

    Returns:
        fisher : dict {param_name -> tensor on CPU}
    """
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.image_encoder.parameters():
        p.requires_grad_(True)

    fisher = {
        n: torch.zeros_like(p, device="cpu")
        for n, p in model.image_encoder.named_parameters()
    }

    # 3x3 dilate/erode kernel (on device)
    k3 = torch.ones(1, 1, 3, 3, device=device)

    n_processed = 0
    nan_batches = 0
    for batch in tqdm(dataloader, desc="Fisher (boundary-aware)"):
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device).float()      # (B, 1, 256, 256)
        bboxes = batch["bbox"].to(device).float()
        B = images.shape[0]

        # --- boundary mask (B, 1, 256, 256) in {0, 1} ---
        with torch.no_grad():
            dilated = F.conv2d(masks,         k3, padding=1).clamp(0, 1)
            eroded  = 1.0 - F.conv2d(1.0 - masks, k3, padding=1).clamp(0, 1)
            boundary_mask = (dilated - eroded)
            weight_map = 1.0 + boundary_weight * boundary_mask   # (B,1,256,256)

        model.zero_grad()
        image_emb = model.image_encoder(images)

        with torch.no_grad():
            box_t = bboxes
            if box_t.dim() == 2:
                box_t = box_t[:, None, :]
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=box_t, masks=None
            )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # Per-pixel weighted BCE (none reduction then weighted sum)
        bce_pp = F.binary_cross_entropy_with_logits(
            low_res_masks, masks, reduction="none"
        )
        bce_loss = (weight_map * bce_pp).sum()

        loss = bce_loss + _dice_loss_sum(low_res_masks, masks)
        loss.backward()

        has_nan = any(
            p.grad is not None and p.grad.isnan().any().item()
            for _, p in model.image_encoder.named_parameters()
        )
        if has_nan:
            nan_batches += 1
            model.zero_grad()
            continue

        with torch.no_grad():
            for n, p in model.image_encoder.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float().cpu() ** 2

        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_processed += B

    if nan_batches > 0:
        print(f"  WARNING: {nan_batches} batches skipped (NaN grads).")

    for n in fisher:
        fisher[n] /= max(n_processed, 1)

    for p in model.image_encoder.parameters():
        p.requires_grad_(False)

    return fisher


# ---------------------------------------------------------------------------
# IMPROVEMENT 5: Distribution-aware scoring
# ---------------------------------------------------------------------------

def split_cal_by_complexity(cal_loader, n_splits=2):
    """
    Split calibration set into n_splits quantile buckets by boundary complexity.

    Bucket 0 = easiest boundaries, bucket (n_splits-1) = hardest.

    Returns:
        list of DataLoaders (one per split), each with batch_size=1.
    """
    dataset = cal_loader.dataset   # torch Subset or similar
    complexities = compute_cal_freq_weights(cal_loader)   # (n_cal,)

    order = np.argsort(complexities)
    chunks = np.array_split(order, n_splits)

    loaders = []
    for ci, idx in enumerate(chunks):
        sub = Subset(dataset, sorted(int(i) for i in idx))
        loader = DataLoader(sub, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
        loaders.append(loader)
        print(f"    subset {ci}: n={len(idx)} "
              f"complexity range [{complexities[idx].min():.3f}, "
              f"{complexities[idx].max():.3f}]")
    return loaders


def combine_scores_dist(dz_list, dr_list, alpha=1.0, pi=None, beta=0.3):
    """
    Distribution-aware combined score (paper Eq. 30).

        Q_g^(r)   = alpha * dz^(r) + (1-alpha) * dr^(r)
        Q_g^dist  = sum_r pi_r * Q_g^(r) + beta * Var_r(Q_g^(r))

    Args:
        dz_list : list of np.ndarray (K,) — delta_zero per subset
        dr_list : list of np.ndarray (K,) — delta_reset per subset
        alpha   : in [0,1]
        pi      : list of non-negative weights summing to 1. Default: uniform.
        beta    : variance penalty (>= 0). 0 reduces to weighted average.

    Returns:
        Q_dist : np.ndarray (K,)
    """
    R = len(dz_list)
    assert R >= 1 and R == len(dr_list)
    if pi is None:
        pi = np.full(R, 1.0 / R, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()

    Q_r = np.stack([alpha * dz + (1.0 - alpha) * dr
                    for dz, dr in zip(dz_list, dr_list)], axis=0)   # (R, K)
    Q_mean = (pi[:, None] * Q_r).sum(axis=0)
    if beta == 0.0 or R == 1:
        return Q_mean.astype(np.float32)
    Q_var = Q_r.var(axis=0)
    return (Q_mean + beta * Q_var).astype(np.float32)


def sum_fisher_dicts(fisher_list, pi=None):
    """Weighted sum of per-subset Fisher dicts (for block_sensitivity)."""
    R = len(fisher_list)
    if pi is None:
        pi = np.full(R, 1.0 / R, dtype=np.float64)
    combined = {}
    for n in fisher_list[0]:
        acc = torch.zeros_like(fisher_list[0][n])
        for r, f in enumerate(fisher_list):
            acc = acc + float(pi[r]) * f[n]
        combined[n] = acc
    return combined


# ---------------------------------------------------------------------------
# Helper: compute per-subset scores given a list of fisher dicts
# ---------------------------------------------------------------------------

def compute_scores_per_subset(model, sam_params, fisher_list):
    """
    Returns:
        dz_head_list, dr_head_list : list of (144,) arrays per subset
        dz_mlp_list,  dr_mlp_list  : list of (12*mlp_dim,) arrays per subset
    """
    dz_head_list, dr_head_list = [], []
    dz_mlp_list,  dr_mlp_list  = [], []
    for fisher in fisher_list:
        dzh, drh = compute_head_scores(model, sam_params, fisher)
        dzm, drm = compute_mlp_neuron_scores(model, sam_params, fisher)
        dz_head_list.append(dzh);  dr_head_list.append(drh)
        dz_mlp_list.append(dzm);   dr_mlp_list.append(drm)
    return dz_head_list, dr_head_list, dz_mlp_list, dr_mlp_list


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _print_row(head_sp, mlp_sp, stage_tag, m, stats):
    p  = stats.get("param_reduction_pct", 0.0)
    fl = stats.get("flops_remaining_G",   0.0)
    fr = stats.get("flop_reduction_pct",  0.0)
    print(f"    [{stage_tag}] sp_head={head_sp*100:.0f}% "
          f"sp_mlp={mlp_sp*100:.0f}% | "
          f"Dice={m['mean_dice']:.4f}  BF1={m['mean_boundary_f1']:.4f}  "
          f"HD95={m['mean_hd95']:.2f}  "
          f"Params↓{p:.1f}%  FLOPs={fl:.1f}G↓{fr:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cascade pruning v6 — boundary-aware Fisher + iterative "
                    "prune-and-recover + distribution-aware scoring"
    )
    # --- Paths ---
    parser.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",    default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_root",   default="asserts/CVC-ColonDB")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--n_cal",       type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="results/pilot_cascade_v6")

    # --- Sparsity grid ---
    parser.add_argument("--head_sparsities", type=float, nargs="+",
                        default=[0.5, 0.7])
    parser.add_argument("--mlp_sparsities", type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 0.9],
                        help="Sorted ascending; used as iterative waypoints.")
    parser.add_argument("--mlp_alpha_values", type=float, nargs="+",
                        default=[1.0],
                        help="alpha=1.0 recommended (zero_only); v5 showed it "
                             "beats alpha=0 in almost all configs.")

    # --- Scoring ---
    parser.add_argument("--phase1_alpha", type=float, default=1.0)
    parser.add_argument("--tau",          type=float, default=0.0)

    # --- IMPROVEMENT 1: Boundary-aware Fisher ---
    parser.add_argument("--boundary_fisher_weight", type=float, default=3.0,
                        help="Lambda_b for boundary-weighted Fisher (0 disables).")

    # --- IMPROVEMENT 2: Iterative Prune-and-Recover ---
    parser.add_argument("--recompute_fisher_between_stages", action="store_true",
                        help="Re-estimate Fisher + scores after each MLP stage "
                             "recovery. Most faithful to the paper's Algorithm 1 "
                             "but 2x-5x slower.")
    parser.add_argument("--oneshot_mlp", action="store_true",
                        help="Disable iterative chaining: reset to post-head state "
                             "before each MLP sparsity trial (= v5 behavior). Use for "
                             "ablation studies isolating improvement #2.")

    # --- IMPROVEMENT 5: Distribution-aware scoring ---
    parser.add_argument("--n_dist_subsets", type=int, default=2,
                        help="Number of calibration subsets for distribution-"
                             "aware scoring (1 disables).")
    parser.add_argument("--dist_beta", type=float, default=0.3,
                        help="Variance penalty beta (>=0) across subsets.")

    # --- Recovery ---
    parser.add_argument("--recovery_steps",       type=int,   default=100)
    parser.add_argument("--recovery_lr",          type=float, default=1e-5)
    parser.add_argument("--feat_distill_weight",  type=float, default=0.5)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_distill_weight", type=float, default=2.0)
    parser.add_argument("--freq_pred_loss_weight", type=float, default=0.5)
    parser.add_argument("--no_freq_sampling", action="store_true")

    # --- Protection ---
    parser.add_argument("--protected_blocks", type=int, nargs="*",
                        default=[10, 11])
    parser.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    parser.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Sort MLP sparsities ascending (required for iterative chain)
    args.mlp_sparsities = sorted(args.mlp_sparsities)

    print("=" * 80)
    print("CASCADE PRUNING v6 — boundary-aware Fisher + iterative + dist-aware")
    print(f"  Data        : {args.data_root}   n_cal={args.n_cal}")
    print(f"  Head sp     : {args.head_sparsities}")
    print(f"  MLP sp      : {args.mlp_sparsities}  "
          f"({'oneshot (v5 behavior)' if args.oneshot_mlp else 'iterative waypoints'})")
    print(f"  alpha values: {args.mlp_alpha_values}")
    print(f"  Recovery    : {args.recovery_steps} steps  lr={args.recovery_lr}")
    print(f"  [IMPROV 1] boundary_fisher_weight = {args.boundary_fisher_weight}")
    print(f"  [IMPROV 2] oneshot_mlp = {args.oneshot_mlp}  "
          f"recompute_fisher_between_stages = {args.recompute_fisher_between_stages}")
    print(f"  [IMPROV 5] n_dist_subsets = {args.n_dist_subsets}  "
          f"dist_beta = {args.dist_beta}")
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
    # 3b. Sampling weights
    # ------------------------------------------------------------------
    cal_freq_weights = None
    if args.recovery_steps > 0 and not args.no_freq_sampling:
        print("\n[3b] Computing boundary-complexity sampling weights ...")
        cal_freq_weights = compute_cal_freq_weights(cal_loader)
        print(f"  Complexity: mean={cal_freq_weights.mean():.3f}  "
              f"min={cal_freq_weights.min():.3f}  max={cal_freq_weights.max():.3f}")
        np.save(os.path.join(args.output_dir, "cal_freq_weights.npy"),
                cal_freq_weights)

    # ------------------------------------------------------------------
    # 4. Baseline
    # ------------------------------------------------------------------
    print("\n[4] Baseline (unpruned) evaluation ...")
    baseline_metrics, baseline_per_sample, _ = _eval_and_collect(
        model, test_loader, device)
    b = baseline_metrics
    print(f"  Dice={b['mean_dice']:.4f}  BF1={b['mean_boundary_f1']:.4f}  "
          f"IoU={b['mean_iou']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ------------------------------------------------------------------
    # 5. Distribution-aware calibration splits  (IMPROVEMENT 5)
    # ------------------------------------------------------------------
    print(f"\n[5] Building {args.n_dist_subsets} calibration subsets ...")
    if args.n_dist_subsets > 1:
        subset_loaders = split_cal_by_complexity(cal_loader, args.n_dist_subsets)
    else:
        subset_loaders = [cal_loader]
        print("    (distribution-aware disabled: single subset)")

    # ------------------------------------------------------------------
    # 6. Boundary-aware Fisher per subset  (IMPROVEMENTS 1 + 5)
    # ------------------------------------------------------------------
    def _estimate_fisher_list(mdl):
        """Estimate Fisher per subset (boundary-aware) on current model."""
        out = []
        for i, sl in enumerate(subset_loaders):
            print(f"    Fisher subset {i+1}/{len(subset_loaders)} "
                  f"(n={len(sl.dataset)}) ...")
            fi = compute_diagonal_fisher_boundary_aware(
                mdl, sl, device,
                boundary_weight=args.boundary_fisher_weight,
            )
            out.append(fi)
        return out

    print("\n[6] Computing boundary-aware Fisher ...")
    t0 = time.time()
    fisher_list = _estimate_fisher_list(model)
    print(f"  Fisher computation: {time.time()-t0:.1f}s")

    fisher_combined = sum_fisher_dicts(fisher_list)
    block_sensitivity = compute_block_sensitivity(fisher_combined, num_blocks=12)
    np.save(os.path.join(args.output_dir, "block_sensitivity.npy"),
            block_sensitivity)
    print(f"  Block sensitivity: min={block_sensitivity.min():.3e}  "
          f"max={block_sensitivity.max():.3e}  "
          f"argmax={int(block_sensitivity.argmax())}  "
          f"protected={args.protected_blocks}")

    # ------------------------------------------------------------------
    # 7. Per-subset scores and distribution-aware head scores
    # ------------------------------------------------------------------
    print("\n[7] Computing dual-intervention scores per subset ...")
    dz_h_list, dr_h_list, dz_m_list, dr_m_list = compute_scores_per_subset(
        model, sam_params, fisher_list
    )

    head_scores = combine_scores_dist(
        dz_h_list, dr_h_list,
        alpha=args.phase1_alpha, beta=args.dist_beta,
    )
    head_summary = score_summary(
        dz_h_list[0] if len(dz_h_list) == 1 else np.mean(dz_h_list, axis=0),
        dr_h_list[0] if len(dr_h_list) == 1 else np.mean(dr_h_list, axis=0),
    )
    mlp_summary = score_summary(
        dz_m_list[0] if len(dz_m_list) == 1 else np.mean(dz_m_list, axis=0),
        dr_m_list[0] if len(dr_m_list) == 1 else np.mean(dr_m_list, axis=0),
    )

    np.savez(
        os.path.join(args.output_dir, "scores.npz"),
        delta_zero_head=np.mean(dz_h_list, axis=0),
        delta_reset_head=np.mean(dr_h_list, axis=0),
        delta_zero_mlp=np.mean(dz_m_list, axis=0),
        delta_reset_mlp=np.mean(dr_m_list, axis=0),
        # Per-subset score stacks (shape (R, K)) — enable exact reconstruction
        # of distribution-aware masks and post-hoc beta sensitivity analysis.
        dz_head_per_subset=np.stack(dz_h_list, axis=0),
        dr_head_per_subset=np.stack(dr_h_list, axis=0),
        dz_mlp_per_subset=np.stack(dz_m_list, axis=0),
        dr_mlp_per_subset=np.stack(dr_m_list, axis=0),
        block_sensitivity=block_sensitivity,
        n_dist_subsets=np.int32(args.n_dist_subsets),
        dist_beta=np.float32(args.dist_beta),
    )

    # ------------------------------------------------------------------
    # 8. Teacher (for feat/logit distillation during recovery)
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

    # ------------------------------------------------------------------
    # 9. Cascade experiment loop with iterative prune-and-recover
    #    (IMPROVEMENT 2)
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "cascade_results_v6.json")

    def _save(obj):
        with open(results_path, "w") as fp:
            json.dump(json.loads(json.dumps(obj, default=_json_safe)), fp, indent=2)

    all_results = {
        "config":             vars(args),
        "baseline":           baseline_metrics,
        "baseline_per_sample": baseline_per_sample,
        "head_score_summary": head_summary,
        "mlp_score_summary":  mlp_summary,
        "block_sensitivity":  block_sensitivity.tolist(),
        "cascade_results":    [],
    }

    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    print("\n[9] Running cascade experiments (iterative prune-and-recover) ...")

    for head_sp in args.head_sparsities:
        print(f"\n{'='*60}")
        print(f"  Head sparsity target: {head_sp*100:.0f}%")

        # ---- Head allocation + mask ----
        per_block_sp_head = allocate_nonuniform_head_sparsity(
            block_sensitivity, head_sp,
            num_heads=12, min_keep=args.nonuniform_min_keep_head,
            protected_blocks=args.protected_blocks,
        )
        head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
        n_heads_kept   = int(head_mask.sum())
        actual_head_sp = 1.0 - n_heads_kept / 144.0
        print(f"  Head mask: kept={n_heads_kept}/144  "
              f"actual_sp={actual_head_sp*100:.1f}%")

        # Restore pristine weights, apply head mask
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        head_hooks = apply_head_mask_to_model(model, head_mask)

        # ---- Phase-1 recovery (head-only) ----
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
        head_only_metrics, head_only_per_sample, _ = _eval_and_collect(
            model, test_loader, device)
        head_only_stats = compute_cascade_stats(model, head_mask, None)
        _print_row(head_sp, 0.0, "head_only",
                   head_only_metrics, head_only_stats)
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
            "per_sample_metrics": head_only_per_sample,
        })
        _save(all_results)

        # ================================================================
        # Iterative MLP prune-and-recover across waypoints
        # (one chain per alpha value; each alpha gets its own trajectory)
        # ================================================================
        post_head_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        for mlp_alpha in args.mlp_alpha_values:
            tag = (f"reset_only" if mlp_alpha == 0.0
                   else f"zero_only" if mlp_alpha == 1.0
                   else f"dual_a{mlp_alpha:.1f}")
            print(f"\n    ---- alpha = {mlp_alpha} ({tag}) iterative chain ----")

            # Reset to post-head state
            model.load_state_dict(
                {k: v.to(device) for k, v in post_head_state.items()}
            )

            # Iteration-local score state
            cur_fisher_list  = fisher_list
            cur_block_sens   = block_sensitivity
            cur_dz_m_list    = dz_m_list
            cur_dr_m_list    = dr_m_list

            active_mlp_hooks = []   # track current MLP hooks (grow through stages)

            for stage_idx, target_mlp_sp in enumerate(args.mlp_sparsities):
                print(f"\n      == stage {stage_idx+1}/{len(args.mlp_sparsities)}: "
                      f"MLP target {target_mlp_sp*100:.0f}% ==")

                # Remove previous MLP hooks; we'll apply a fresh cumulative mask
                remove_hooks(active_mlp_hooks)
                active_mlp_hooks = []

                # ONESHOT mode: reset to post-head state before each trial.
                # This replays v5 behavior (each mlp_sp is an independent trial)
                # and disables iterative chaining / per-stage Fisher updates.
                if args.oneshot_mlp and stage_idx > 0:
                    model.load_state_dict(
                        {k: v.to(device) for k, v in post_head_state.items()}
                    )
                    cur_fisher_list = fisher_list
                    cur_block_sens  = block_sensitivity
                    cur_dz_m_list   = dz_m_list
                    cur_dr_m_list   = dr_m_list

                # Build MLP mask at current target sparsity using current scores
                q_mlp = combine_scores_dist(
                    cur_dz_m_list, cur_dr_m_list,
                    alpha=mlp_alpha, beta=args.dist_beta,
                )
                per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
                    cur_block_sens, target_mlp_sp,
                    mlp_dim=mlp_dim,
                    min_frac=args.nonuniform_min_frac_mlp,
                    protected_blocks=args.protected_blocks,
                )
                neuron_mask = generate_neuron_mask_nonuniform(
                    q_mlp, per_block_sp_mlp, mlp_dim=mlp_dim
                )
                actual_mlp_sp = 1.0 - float(neuron_mask.sum()) / (12 * mlp_dim)

                active_mlp_hooks = apply_mlp_mask_to_model(model, neuron_mask)

                # Stage recovery
                if args.recovery_steps > 0:
                    print(f"        recovery ({args.recovery_steps} steps) ...")
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

                # Eval + save
                metrics, stage_per_sample, _ = _eval_and_collect(
                    model, test_loader, device)
                stats = compute_cascade_stats(model, head_mask, neuron_mask)
                iter_tag = "oneshot" if args.oneshot_mlp else "iterative"
                _print_row(head_sp, target_mlp_sp, f"{iter_tag}[{tag}]",
                           metrics, stats)

                all_results["cascade_results"].append({
                    "phase":          "cascade",
                    "iteration":      iter_tag,
                    "stage_idx":      stage_idx,
                    "head_sp_target": head_sp,
                    "head_sp_actual": actual_head_sp,
                    "mlp_method":     "nonuniform",
                    "mlp_sp_target":  target_mlp_sp,
                    "mlp_sp_actual":  actual_mlp_sp,
                    "mlp_score":      tag,
                    "mlp_alpha":      mlp_alpha,
                    **metrics,
                    **stats,
                    "per_sample_metrics": stage_per_sample,
                })
                _save(all_results)

                # ---- Optionally re-score for next stage (IMPROVEMENT 2 core) ----
                # Skip in oneshot mode — each stage is independent, no chain.
                if (not args.oneshot_mlp
                        and args.recompute_fisher_between_stages
                        and stage_idx < len(args.mlp_sparsities) - 1):
                    print(f"        [re-score] Fisher + per-subset MLP scores ...")
                    cur_fisher_list = _estimate_fisher_list(model)
                    cur_combined    = sum_fisher_dicts(cur_fisher_list)
                    cur_block_sens  = compute_block_sensitivity(cur_combined, 12)
                    # Only MLP scores change stage-to-stage; head mask is fixed
                    _, _, cur_dz_m_list, cur_dr_m_list = compute_scores_per_subset(
                        model, sam_params, cur_fisher_list
                    )

            # End of iterative chain for this alpha — clean up MLP hooks
            remove_hooks(active_mlp_hooks)

        # End of alpha loop — clean up head hooks before next head_sp
        remove_hooks(head_hooks)
        _save(all_results)
        print(f"  [checkpoint] saved → {results_path}")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    _save(all_results)
    print(f"\n[10] Results saved to {results_path}")

    print("\n" + "=" * 120)
    print("v6 RESULT SUMMARY")
    print(f"  boundary_fisher_w={args.boundary_fisher_weight}  "
          f"n_subsets={args.n_dist_subsets} beta={args.dist_beta}  "
          f"recompute_stages={args.recompute_fisher_between_stages}")
    print("=" * 120)

    print(f"\n  {'Config':<50} {'Dice':>6} {'BF1':>6} {'HD95':>7} "
          f"{'Par%':>6} {'FL%':>6}")
    print("  " + "-" * 90)
    for r in all_results["cascade_results"]:
        if r["phase"] == "head_only":
            lbl = f"HEAD_ONLY h={r['head_sp_target']*100:.0f}%"
        else:
            lbl = (f"h={r['head_sp_target']*100:.0f}% "
                   f"m={r['mlp_sp_target']*100:.0f}%_{r.get('mlp_score','?')}")
        print(f"  {lbl:<50} "
              f"{r['mean_dice']:>6.4f} {r['mean_boundary_f1']:>6.4f} "
              f"{r['mean_hd95']:>7.2f} "
              f"{r.get('param_reduction_pct',0):>6.1f} "
              f"{r.get('flop_reduction_pct',0):>6.1f}")

    print(f"\n  Baseline:  Dice={b['mean_dice']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")


if __name__ == "__main__":
    main()
