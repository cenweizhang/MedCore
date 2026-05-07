# -*- coding: utf-8 -*-
"""
Cascade pruning v7: Cross-Fisher + Per-block adaptive α +
                    Exact group validation + End-of-run visualization.

Improvements vs v6
------------------
V7-1  Cross-Fisher for Δ_reset
      Δ̂_reset_g = 0.5 * Σ sqrt(F_i^M · F_i^S(D_med)) · (θ_i^M - θ_i^S)²
      F^S(D_med) = SAM's boundary-aware Fisher on the medical cal set.
      Geometric mean protects groups that are (a) currently important in
      MedSAM AND (b) were hard for SAM to handle on medical images.
      Δ_zero is unaffected — still uses F^M (current MedSAM curvature).

V7-2  Per-block adaptive α
      α_b = (1 + r_b) / 2,  r_b = Pearson r(Δ_zero, Δ_reset) over heads in block b.
      r_b ≈ 1  → α_b ≈ 1 (Δ_zero sufficient, deeply specialised block)
      r_b < 0.8 → α_b drops below 1 (diverged signals, e.g. Block 3 r=0.63)
      Not a hand-tuned hyper-parameter — derived from the cal data.

V7-3  Exact group-level reset validation  (diagnostic, default OFF)
      For the top-K groups by approximate Δ_reset, actually swap weights to
      SAM values and measure true loss increase on a small cal subset.
      Reports exact/approx ratio to quantify diagonal-Fisher approximation
      error.  Enable with --exact_validate_topk N.

      All v6 improvements are retained:
        - Boundary-aware Fisher        (--boundary_fisher_weight)
        - Iterative prune-and-recover  (--recompute_fisher_between_stages)
        - Distribution-aware scoring   (--n_dist_subsets, --dist_beta)

V7-4  End-of-run visualization
      After all experiments the best-BF1 pruning masks are applied to a
      fresh model and an 8-sample comparison figure is saved:
        col 0: image  col 1: GT  col 2: baseline  col 3: pruned  col 4: diff

Usage
-----
    cd /volume/med-train/users/jshan/testSAMpruning
    python -m pilot_dual.run_cascade_v7 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
        --data_root   asserts/CVC-ColonDB \
        --device      cuda:0 \
        --output_dir  results/pilot_cascade_v7 \
        --recovery_steps 100 --recovery_lr 1e-5 \
        --boundary_fisher_weight 3.0 \
        --n_dist_subsets 2 --dist_beta 0.3 \
        --recompute_fisher_between_stages \
        --mlp_sparsities 0.5 0.7 0.85 0.9 \
        --head_sparsities 0.5 0.7 \
        --protected_blocks 10 11

Ablation: disable cross-fisher / adaptive-alpha individually
    --no_cross_fisher        : revert to standard F^M for Δ_reset (v6 behaviour)
    --no_adaptive_alpha      : use single global --phase1_alpha for all blocks
    --exact_validate_topk 10 : validate top-10 heads with exact reset
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
    score_summary,
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
# Shared helpers  (same as v6)
# ---------------------------------------------------------------------------

def _dice_loss_sum(pred_logits, target):
    pred   = torch.sigmoid(pred_logits)
    pred   = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)
    return per.sum()


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
        bboxes   = batch["bbox"].to(device).float()
        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]

        image_emb = model.image_encoder(images)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=bboxes, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred_1024 = F.interpolate(low_res, (1024, 1024),
                                  mode="bilinear", align_corners=False)
        pred_bin  = (torch.sigmoid(pred_1024) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        gt_bin    = masks_gt[0].astype(np.uint8)
        m         = compute_all_metrics(pred_bin, gt_bin)
        m["name"] = batch["name"][0] if "name" in batch else ""
        all_metrics.append(m)

    keys = ["dice", "iou", "boundary_f1", "hd95"]
    avg  = {f"mean_{k}": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    avg.update({f"std_{k}":  float(np.std( [m[k] for m in all_metrics])) for k in keys})
    avg["head_sparsity"] = 0.0
    avg["kept_heads"]    = 144
    avg["total_heads"]   = 144
    return avg, all_metrics, {}


def compute_cal_freq_weights(cal_loader):
    k3 = torch.ones(1, 1, 3, 3)
    ordered = DataLoader(cal_loader.dataset, batch_size=1,
                         shuffle=False, num_workers=0, pin_memory=False)
    weights = []
    for batch in ordered:
        mask = batch["mask_256"][0, 0].float()
        m4d  = mask[None, None]
        dilated  = F.conv2d(m4d, k3, padding=1).clamp(0, 1)
        eroded   = 1.0 - F.conv2d(1.0 - m4d, k3, padding=1).clamp(0, 1)
        boundary = (dilated - eroded).squeeze()
        perimeter   = boundary.sum().item()
        area        = mask.sum().item()
        complexity  = perimeter / (area ** 0.5 + 1.0)
        weights.append(max(complexity, 1e-4))
    return np.array(weights, dtype=np.float32)


def _print_row(head_sp, mlp_sp, tag, m, stats):
    p  = stats.get("param_reduction_pct", 0.0)
    fl = stats.get("flops_remaining_G",   0.0)
    fr = stats.get("flop_reduction_pct",  0.0)
    print(f"    [{tag}] h={head_sp*100:.0f}% m={mlp_sp*100:.0f}% | "
          f"Dice={m['mean_dice']:.4f}  BF1={m['mean_boundary_f1']:.4f}  "
          f"HD95={m['mean_hd95']:.2f}  Params↓{p:.1f}%  FLOPs={fl:.1f}G↓{fr:.1f}%")


# ---------------------------------------------------------------------------
# Boundary-aware Fisher  (same as v6 — applies to any model)
# ---------------------------------------------------------------------------

def compute_diagonal_fisher_boundary_aware(model, dataloader, device,
                                            boundary_weight=3.0):
    model.eval()
    for p in model.parameters():        p.requires_grad_(False)
    for p in model.image_encoder.parameters(): p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p, device="cpu")
              for n, p in model.image_encoder.named_parameters()}
    k3 = torch.ones(1, 1, 3, 3, device=device)
    n_processed = nan_batches = 0

    for batch in tqdm(dataloader, desc="Fisher (boundary-aware)", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device).float()
        bboxes = batch["bbox"].to(device).float()
        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]
        B = images.shape[0]

        with torch.no_grad():
            dilated      = F.conv2d(masks,           k3, padding=1).clamp(0, 1)
            eroded       = 1.0 - F.conv2d(1.0 - masks, k3, padding=1).clamp(0, 1)
            weight_map   = 1.0 + boundary_weight * (dilated - eroded)

        model.zero_grad()
        image_emb = model.image_encoder(images)
        with torch.no_grad():
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=bboxes, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        bce_pp   = F.binary_cross_entropy_with_logits(low_res, masks, reduction="none")
        loss     = (weight_map * bce_pp).sum() + _dice_loss_sum(low_res, masks)
        loss.backward()

        has_nan = any(p.grad is not None and p.grad.isnan().any().item()
                      for _, p in model.image_encoder.named_parameters())
        if has_nan:
            nan_batches += 1
            model.zero_grad()
            continue

        with torch.no_grad():
            for n, p in model.image_encoder.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float().cpu() ** 2
        model.zero_grad()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        n_processed += B

    if nan_batches: print(f"  WARNING: {nan_batches} batches skipped (NaN grads).")
    for n in fisher: fisher[n] /= max(n_processed, 1)
    for p in model.image_encoder.parameters(): p.requires_grad_(False)
    return fisher


# ---------------------------------------------------------------------------
# V7-1: Cross-Fisher
# ---------------------------------------------------------------------------

def compute_sam_fisher_on_medical(sam_ckpt, dataloader, device,
                                   boundary_weight=3.0):
    """
    Load SAM, compute boundary-aware Fisher on the medical cal set, return
    the Fisher dict, then delete the SAM model to free GPU memory.

    F^S(D_med) captures which parameters SAM finds most sensitive when
    processing out-of-distribution medical images — i.e. what medical
    fine-tuning most needed to update.
    """
    print("  [V7-1] Loading SAM for cross-Fisher estimation ...")
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam_model = sam_model.to(device).eval()
    for p in sam_model.prompt_encoder.parameters(): p.requires_grad_(False)
    for p in sam_model.mask_decoder.parameters():   p.requires_grad_(False)

    fisher_s = compute_diagonal_fisher_boundary_aware(
        sam_model, dataloader, device, boundary_weight=boundary_weight)

    del sam_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("  [V7-1] SAM Fisher done; SAM model freed from GPU.")
    return fisher_s


def compute_cross_fisher(fisher_m, fisher_s):
    """
    Geometric mean Fisher: sqrt(F^M · F^S) element-wise.

    Used as the Fisher weight in Δ_reset so that the reset score reflects
    both (a) current MedSAM curvature and (b) SAM's curvature on medical
    data — i.e. the cross-model distribution shift on the same medical images.

    Returns a new dict with the same keys as fisher_m.
    """
    cross = {}
    for n in fisher_m:
        fm = fisher_m[n].float()
        fs = fisher_s.get(n, torch.zeros_like(fm)).float()
        cross[n] = torch.sqrt(fm * fs + 1e-30)   # +eps avoids sqrt(0) gradient issues
    return cross


def build_cross_fisher_list(fisher_m_list, fisher_s):
    """
    For each per-subset MedSAM Fisher, combine with (global) SAM Fisher.
    Returns a list of cross-Fisher dicts, one per calibration subset.
    """
    return [compute_cross_fisher(fm, fisher_s) for fm in fisher_m_list]


# ---------------------------------------------------------------------------
# Distribution-aware splits  (same as v6)
# ---------------------------------------------------------------------------

def split_cal_by_complexity(cal_loader, n_splits=2):
    dataset      = cal_loader.dataset
    complexities = compute_cal_freq_weights(cal_loader)
    order        = np.argsort(complexities)
    chunks       = np.array_split(order, n_splits)
    loaders = []
    for ci, idx in enumerate(chunks):
        sub    = Subset(dataset, sorted(int(i) for i in idx))
        loader = DataLoader(sub, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
        loaders.append(loader)
        print(f"    subset {ci}: n={len(idx)} "
              f"complexity [{complexities[idx].min():.3f}, "
              f"{complexities[idx].max():.3f}]")
    return loaders


def sum_fisher_dicts(fisher_list, pi=None):
    R  = len(fisher_list)
    if pi is None: pi = np.full(R, 1.0 / R)
    combined = {}
    for n in fisher_list[0]:
        acc = torch.zeros_like(fisher_list[0][n])
        for r, f in enumerate(fisher_list):
            acc = acc + float(pi[r]) * f[n]
        combined[n] = acc
    return combined


# ---------------------------------------------------------------------------
# V7-1 + V7-2: Cross-Fisher scoring + adaptive α
# ---------------------------------------------------------------------------

def compute_scores_per_subset_v7(model, sam_params, fisher_m_list, cross_fisher_list):
    """
    Δ_zero uses F^M  (standard MedSAM curvature).
    Δ_reset uses √(F^M · F^S)  (cross-Fisher, V7-1).

    Calls existing compute_head_scores / compute_mlp_neuron_scores twice:
    once with fisher_m to get dz, once with cross_fisher to get dr.
    """
    dz_head_list, dr_head_list = [], []
    dz_mlp_list,  dr_mlp_list  = [], []

    for fm, fc in zip(fisher_m_list, cross_fisher_list):
        dz_h, _  = compute_head_scores(model, sam_params, fm)
        dz_m, _  = compute_mlp_neuron_scores(model, sam_params, fm)
        _,  dr_h = compute_head_scores(model, sam_params, fc)
        _,  dr_m = compute_mlp_neuron_scores(model, sam_params, fc)
        dz_head_list.append(dz_h);  dr_head_list.append(dr_h)
        dz_mlp_list.append(dz_m);   dr_mlp_list.append(dr_m)

    return dz_head_list, dr_head_list, dz_mlp_list, dr_mlp_list


def compute_adaptive_alpha_per_block(dz_head, dr_head,
                                      num_blocks=12, num_heads=12):
    """
    V7-2: α_b = (1 + r_b) / 2  where r_b is the Pearson correlation
    between Δ_zero and Δ_reset across the 12 heads of block b.

    r_b ≈ 1 → α_b ≈ 1  (Δ_zero sufficient, scores carry same info)
    r_b ≈ 0 → α_b ≈ 0.5 (equal weight to both)

    Returns np.ndarray shape (num_blocks,), dtype float32.
    """
    alpha_b = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        dz_b = dz_head[b * num_heads: (b + 1) * num_heads]
        dr_b = dr_head[b * num_heads: (b + 1) * num_heads]
        if dz_b.std() > 1e-12 and dr_b.std() > 1e-12:
            r = float(np.corrcoef(dz_b, dr_b)[0, 1])
            r = float(np.clip(r, -1.0, 1.0))
        else:
            r = 1.0
        alpha_b[b] = (1.0 + r) / 2.0
    return alpha_b


def combine_scores_dist_adaptive(dz_list, dr_list, alpha_per_block,
                                  items_per_block, pi=None, beta=0.3):
    """
    V7-2 + distribution-aware (Eq. 30):
        Q_g^(r) = α_b(g) · dz_g^(r) + (1−α_b(g)) · dr_g^(r)
        Q_g^dist = Σ_r π_r Q_g^(r) + β · Var_r(Q_g^(r))

    alpha_per_block: (num_blocks,) — per-block α derived from correlation.
    items_per_block: 12 for heads, mlp_dim for MLP neurons.
    """
    R = len(dz_list)
    K = len(dz_list[0])

    # Broadcast per-block α to per-group α vector  (K,)
    alpha_vec = np.repeat(alpha_per_block, items_per_block).astype(np.float32)
    if len(alpha_vec) != K:
        # Fallback: single global alpha (mean of per-block values)
        alpha_vec = np.full(K, float(alpha_per_block.mean()), dtype=np.float32)

    Q_r = np.stack(
        [alpha_vec * dz + (1.0 - alpha_vec) * dr
         for dz, dr in zip(dz_list, dr_list)],
        axis=0,
    )   # (R, K)

    if pi is None:
        pi = np.full(R, 1.0 / R, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64) / np.sum(pi)

    Q_mean = (pi[:, None] * Q_r).sum(axis=0)
    if beta == 0.0 or R == 1:
        return Q_mean.astype(np.float32)
    return (Q_mean + beta * Q_r.var(axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# V7-3: Exact group-level reset validation  (diagnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _loss_on_cal_subset(model, cal_loader, device, n_samples=16):
    """Sum of Dice+BCE losses on the first n_samples batches of cal_loader."""
    model.eval()
    total, count = 0.0, 0
    for batch in cal_loader:
        if count >= n_samples: break
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device).float()
        bboxes = batch["bbox"].to(device).float()
        if bboxes.dim() == 2: bboxes = bboxes[:, None, :]
        emb = model.image_encoder(images)
        sp, dp = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
        low, _ = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
        )
        total += (F.binary_cross_entropy_with_logits(low, masks, reduction="sum")
                  + _dice_loss_sum(low, masks)).item()
        count += images.shape[0]
    return total / max(count, 1)


def exact_group_reset_validation(model, sam_params, dr_head, cal_loader,
                                   device, topk=5, n_validate=16):
    """
    V7-3: For the top-K heads by approximate Δ_reset, actually swap
    weights to SAM values and measure the true loss increase.

    Returns a dict {group_id: {approx, exact, ratio}}.
    ratio = exact / approx;  ratio ≈ 1 ⟹ diagonal approximation is good.
    """
    print(f"  [V7-3] Exact validation: top-{topk} heads, "
          f"n_validate={n_validate} samples ...")

    baseline = _loss_on_cal_subset(model, cal_loader, device, n_validate)
    top_ids  = np.argsort(dr_head)[-topk:]
    results  = {}

    for head_id in top_ids:
        b, h     = int(head_id // 12), int(head_id % 12)
        attn     = model.image_encoder.blocks[b].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // 12
        has_bias = attn.qkv.bias is not None

        # Slice definitions
        q_rows = slice(h * head_dim,          (h + 1) * head_dim)
        k_rows = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
        v_rows = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
        p_cols = slice(h * head_dim,          (h + 1) * head_dim)

        # Save originals
        saved_qkv_w  = attn.qkv.weight.data.clone()
        saved_proj_w = attn.proj.weight.data.clone()
        if has_bias:
            saved_qkv_b = attn.qkv.bias.data.clone()

        # Swap to SAM
        with torch.no_grad():
            bk = f"blocks.{b}.attn"
            sam_qkv_w  = sam_params[f"{bk}.qkv.weight"].to(device)
            sam_proj_w = sam_params[f"{bk}.proj.weight"].to(device)
            attn.qkv.weight.data[q_rows]    = sam_qkv_w[q_rows]
            attn.qkv.weight.data[k_rows]    = sam_qkv_w[k_rows]
            attn.qkv.weight.data[v_rows]    = sam_qkv_w[v_rows]
            attn.proj.weight.data[:, p_cols] = sam_proj_w[:, p_cols]
            if has_bias:
                sam_qkv_b = sam_params[f"{bk}.qkv.bias"].to(device)
                attn.qkv.bias.data[q_rows] = sam_qkv_b[q_rows]
                attn.qkv.bias.data[k_rows] = sam_qkv_b[k_rows]
                attn.qkv.bias.data[v_rows] = sam_qkv_b[v_rows]

        exact_delta = _loss_on_cal_subset(model, cal_loader, device, n_validate) - baseline
        approx      = float(dr_head[head_id])
        ratio       = float(exact_delta / (approx + 1e-12))

        print(f"    head b{b}_h{h}: approx={approx:.5f}  "
              f"exact={exact_delta:.5f}  ratio={ratio:.3f}")
        results[f"b{b}_h{h}"] = {"approx": approx,
                                   "exact":  float(exact_delta),
                                   "ratio":  ratio}

        # Restore originals
        with torch.no_grad():
            attn.qkv.weight.data.copy_(saved_qkv_w)
            attn.proj.weight.data.copy_(saved_proj_w)
            if has_bias:
                attn.qkv.bias.data.copy_(saved_qkv_b)

    return results


# ---------------------------------------------------------------------------
# V7-4: End-of-run visualization  (8 samples, pruned-without-recovery)
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_best_result(original_state, head_mask, neuron_mask,
                           test_loader, device, output_path, n_samples=8):
    """
    Apply head_mask + neuron_mask to a fresh model (no recovery), run
    inference on n_samples evenly-spaced test images, save figure with
    5 columns: image | GT | baseline | pruned | diff (|baseline - pruned|).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.transform import resize as sk_resize

    def _load_model():
        m = sam_model_registry["vit_b"]()
        m.load_state_dict({k: v.clone() for k, v in original_state.items()})
        return m.to(device).eval()

    def _infer(mdl, batch):
        images = batch["image"].to(device)
        bboxes = batch["bbox"].to(device).float()
        if bboxes.dim() == 2: bboxes = bboxes[:, None, :]
        emb     = mdl.image_encoder(images)
        sp, dp  = mdl.prompt_encoder(points=None, boxes=bboxes, masks=None)
        low, _  = mdl.mask_decoder(
            image_embeddings=emb,
            image_pe=mdl.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
        )
        pred = F.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)
        return (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    def _d256(arr, order=0):
        return sk_resize(arr, (256, 256), order=order,
                         preserve_range=True, anti_aliasing=False)

    # Collect all batches; pick evenly-spaced indices
    all_batches = list(test_loader)
    n_total  = len(all_batches)
    indices  = np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)
    n_rows   = len(indices)

    model_b = _load_model()
    model_p = _load_model()
    hooks_h = apply_head_mask_to_model(model_p, head_mask)
    hooks_m = apply_mlp_mask_to_model(model_p, neuron_mask) if neuron_mask is not None else []

    n_pruned_heads = int((1 - head_mask).sum())
    n_pruned_mlp   = int((1 - neuron_mask).sum()) if neuron_mask is not None else 0
    subtitle = (f"Best pruned: {n_pruned_heads}/144 heads pruned, "
                f"{n_pruned_mlp} MLP neurons pruned  "
                f"(masks applied without recovery)")

    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 3.5))
    if n_rows == 1: axes = axes[None, :]
    fig.suptitle(subtitle, fontsize=10, y=1.01)

    for c, title in enumerate(["Image", "GT Mask", "Baseline", "Pruned (no recovery)", "Diff"]):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(indices):
        batch   = all_batches[int(idx)]
        name    = batch["name"][0] if "name" in batch else str(idx)
        img_np  = batch["image"][0].cpu().numpy().transpose(1, 2, 0)
        img_np  = (img_np - img_np.min()) / (img_np.ptp() + 1e-8)
        gt_np   = batch["mask_1024"].numpy()[0].squeeze().astype(np.uint8)

        pred_b = _infer(model_b, batch)
        pred_p = _infer(model_p, batch)
        diff   = np.abs(pred_b.astype(np.float32) - pred_p.astype(np.float32))

        axes[row, 0].imshow(_d256(img_np, order=1).clip(0, 1))
        axes[row, 1].imshow(_d256(gt_np),   cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(_d256(pred_b),  cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(_d256(pred_p),  cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(_d256(diff),    cmap="hot",  vmin=0, vmax=1)

        for c in range(5): axes[row, c].axis("off")
        axes[row, 0].set_ylabel(name[:25], fontsize=7, rotation=0,
                                 labelpad=60, va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [V7-4] Visualization saved → {output_path}")

    remove_hooks(hooks_h + hooks_m)
    del model_b, model_p
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cascade pruning v7 — cross-Fisher + adaptive α + exact validation")

    # --- Paths ---
    parser.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",    default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_root",   default="asserts/CVC-ColonDB")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--n_cal",       type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="results/pilot_cascade_v7")

    # --- Sparsity grid ---
    parser.add_argument("--head_sparsities",   type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--mlp_sparsities",    type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 0.9])
    parser.add_argument("--mlp_alpha_values",  type=float, nargs="+", default=[1.0])

    # --- V7 flags ---
    parser.add_argument("--no_cross_fisher",   action="store_true",
                        help="Disable V7-1: revert Δ_reset to use F^M only (v6).")
    parser.add_argument("--no_adaptive_alpha", action="store_true",
                        help="Disable V7-2: use single global --phase1_alpha.")
    parser.add_argument("--exact_validate_topk", type=int, default=0,
                        help="V7-3: validate top-K heads with exact reset (0=off).")
    parser.add_argument("--n_visualize_samples", type=int, default=8,
                        help="V7-4: number of samples in end-of-run figure.")

    # --- Scoring (fallbacks when adaptive-α disabled) ---
    parser.add_argument("--phase1_alpha", type=float, default=1.0)
    parser.add_argument("--tau",          type=float, default=0.0)

    # --- v6 improvements retained ---
    parser.add_argument("--boundary_fisher_weight",         type=float, default=3.0)
    parser.add_argument("--recompute_fisher_between_stages", action="store_true")
    parser.add_argument("--oneshot_mlp",                    action="store_true")
    parser.add_argument("--n_dist_subsets",  type=int,   default=2)
    parser.add_argument("--dist_beta",       type=float, default=0.3)

    # --- Recovery ---
    parser.add_argument("--recovery_steps",        type=int,   default=100)
    parser.add_argument("--recovery_lr",           type=float, default=1e-5)
    parser.add_argument("--feat_distill_weight",   type=float, default=0.5)
    parser.add_argument("--boundary_loss_weight",  type=float, default=1.0)
    parser.add_argument("--logit_distill_weight",  type=float, default=2.0)
    parser.add_argument("--freq_pred_loss_weight", type=float, default=0.5)
    parser.add_argument("--no_freq_sampling",      action="store_true")

    # --- Protection ---
    parser.add_argument("--protected_blocks",           type=int, nargs="*", default=[10, 11])
    parser.add_argument("--nonuniform_min_keep_head",   type=int,   default=1)
    parser.add_argument("--nonuniform_min_frac_mlp",    type=float, default=0.05)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.mlp_sparsities = sorted(args.mlp_sparsities)

    print("=" * 80)
    print("CASCADE PRUNING v7 — cross-Fisher + adaptive α + exact validation")
    print(f"  [V7-1] cross_fisher   : {'OFF (v6 mode)' if args.no_cross_fisher else 'ON'}")
    print(f"  [V7-2] adaptive_alpha : {'OFF (global alpha=' + str(args.phase1_alpha) + ')' if args.no_adaptive_alpha else 'ON (data-derived per block)'}")
    print(f"  [V7-3] exact_validate : {args.exact_validate_topk} heads")
    print(f"  [V7-4] visualize      : {args.n_visualize_samples} samples")
    print(f"  Boundary fisher λ={args.boundary_fisher_weight}  "
          f"n_subsets={args.n_dist_subsets}  β={args.dist_beta}")
    print(f"  Head sp: {args.head_sparsities}   MLP sp: {args.mlp_sparsities}")
    print(f"  Protected blocks: {args.protected_blocks}")
    print(f"  Output: {args.output_dir}")
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
    print(f"  Cal: {args.n_cal}  Test: {sum(1 for _ in test_loader)}")

    # Sampling weights for recovery
    cal_freq_weights = None
    if args.recovery_steps > 0 and not args.no_freq_sampling:
        print("\n[3b] Computing boundary-complexity sampling weights ...")
        cal_freq_weights = compute_cal_freq_weights(cal_loader)
        np.save(os.path.join(args.output_dir, "cal_freq_weights.npy"), cal_freq_weights)

    # ------------------------------------------------------------------
    # 4. Baseline
    # ------------------------------------------------------------------
    print("\n[4] Baseline evaluation ...")
    baseline_metrics, baseline_per_sample, _ = _eval_and_collect(
        model, test_loader, device)
    b = baseline_metrics
    print(f"  Dice={b['mean_dice']:.4f}  BF1={b['mean_boundary_f1']:.4f}  "
          f"IoU={b['mean_iou']:.4f}  HD95={b['mean_hd95']:.2f}")

    # ------------------------------------------------------------------
    # 5. Calibration subsets by boundary complexity
    # ------------------------------------------------------------------
    print(f"\n[5] Building {args.n_dist_subsets} calibration subsets ...")
    if args.n_dist_subsets > 1:
        subset_loaders = split_cal_by_complexity(cal_loader, args.n_dist_subsets)
    else:
        subset_loaders = [cal_loader]
        print("    (single subset — distribution-aware disabled)")

    # ------------------------------------------------------------------
    # 6. MedSAM boundary-aware Fisher per subset  (F^M_r)
    # ------------------------------------------------------------------
    def _estimate_fisher_list(mdl):
        out = []
        for i, sl in enumerate(subset_loaders):
            print(f"    Fisher[M] subset {i+1}/{len(subset_loaders)} (n={len(sl.dataset)}) ...")
            out.append(compute_diagonal_fisher_boundary_aware(
                mdl, sl, device, boundary_weight=args.boundary_fisher_weight))
        return out

    print("\n[6] Computing MedSAM boundary-aware Fisher (F^M) ...")
    t0 = time.time()
    fisher_m_list = _estimate_fisher_list(model)
    print(f"  F^M done: {time.time()-t0:.1f}s")

    fisher_m_combined = sum_fisher_dicts(fisher_m_list)
    block_sensitivity  = compute_block_sensitivity(fisher_m_combined, num_blocks=12)
    np.save(os.path.join(args.output_dir, "block_sensitivity.npy"), block_sensitivity)

    # ------------------------------------------------------------------
    # 6b. V7-1: SAM Fisher on medical cal set (F^S) + cross-Fisher
    # ------------------------------------------------------------------
    if not args.no_cross_fisher:
        print("\n[6b] Computing SAM Fisher on medical data (F^S) for cross-Fisher ...")
        t0 = time.time()
        fisher_s = compute_sam_fisher_on_medical(
            args.sam_ckpt, cal_loader, device,
            boundary_weight=args.boundary_fisher_weight)
        print(f"  F^S done: {time.time()-t0:.1f}s")
        cross_fisher_list = build_cross_fisher_list(fisher_m_list, fisher_s)
        del fisher_s
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("  Cross-Fisher list built.")
    else:
        # V7-1 off: use F^M as fisher for Δ_reset (identical to v6)
        cross_fisher_list = fisher_m_list
        print("\n[6b] cross-Fisher DISABLED — using F^M for Δ_reset (v6 behaviour).")

    # ------------------------------------------------------------------
    # 7. Per-subset scores using cross-Fisher for Δ_reset
    # ------------------------------------------------------------------
    print("\n[7] Computing dual-intervention scores per subset ...")
    dz_h_list, dr_h_list, dz_m_list, dr_m_list = compute_scores_per_subset_v7(
        model, sam_params, fisher_m_list, cross_fisher_list)

    # Mean scores for summary / fallback
    dz_head_mean = np.mean(dz_h_list, axis=0)
    dr_head_mean = np.mean(dr_h_list, axis=0)

    # ------------------------------------------------------------------
    # 7b. V7-2: Per-block adaptive α
    # ------------------------------------------------------------------
    if not args.no_adaptive_alpha:
        alpha_per_block = compute_adaptive_alpha_per_block(
            dz_head_mean, dr_head_mean)
        print("\n[7b] Per-block adaptive α (V7-2):")
        for bi, ab in enumerate(alpha_per_block):
            mark = " ←" if abs(ab - 1.0) > 0.05 else ""
            print(f"  Block {bi:2d}: α={ab:.3f}{mark}")
    else:
        alpha_per_block = np.full(12, args.phase1_alpha, dtype=np.float32)
        print(f"\n[7b] Adaptive α DISABLED — using global α={args.phase1_alpha}.")

    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    # Final head scores (distribution-aware + adaptive α)
    head_scores = combine_scores_dist_adaptive(
        dz_h_list, dr_h_list, alpha_per_block,
        items_per_block=12, beta=args.dist_beta)

    head_summary = score_summary(dz_head_mean, dr_head_mean)

    # ------------------------------------------------------------------
    # 7c. V7-3: Exact validation (diagnostic)
    # ------------------------------------------------------------------
    exact_val_results = {}
    if args.exact_validate_topk > 0:
        print(f"\n[7c] Exact group-level validation (V7-3) ...")
        exact_val_results = exact_group_reset_validation(
            model, sam_params, dr_head_mean, cal_loader, device,
            topk=args.exact_validate_topk, n_validate=16)

    # Save scores
    np.savez(
        os.path.join(args.output_dir, "scores.npz"),
        delta_zero_head=dz_head_mean,
        delta_reset_head=dr_head_mean,
        delta_zero_mlp=np.mean(dz_m_list, axis=0),
        delta_reset_mlp=np.mean(dr_m_list, axis=0),
        dz_head_per_subset=np.stack(dz_h_list, axis=0),
        dr_head_per_subset=np.stack(dr_h_list, axis=0),
        dz_mlp_per_subset=np.stack(dz_m_list, axis=0),
        dr_mlp_per_subset=np.stack(dr_m_list, axis=0),
        block_sensitivity=block_sensitivity,
        alpha_per_block=alpha_per_block,
        n_dist_subsets=np.int32(args.n_dist_subsets),
        dist_beta=np.float32(args.dist_beta),
    )

    # ------------------------------------------------------------------
    # 8. Teacher for distillation
    # ------------------------------------------------------------------
    feature_teacher = None
    if args.recovery_steps > 0 and (args.feat_distill_weight > 0
                                     or args.logit_distill_weight > 0):
        print("\n[8] Building teacher model (CPU) ...")
        feature_teacher = copy.deepcopy(model).cpu().eval()
        for p in feature_teacher.parameters(): p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 9. Cascade experiment loop
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "cascade_results_v7.json")

    def _save(obj):
        with open(results_path, "w") as fp:
            json.dump(json.loads(json.dumps(obj, default=_json_safe)), fp, indent=2)

    all_results = {
        "config":              vars(args),
        "baseline":            baseline_metrics,
        "baseline_per_sample": baseline_per_sample,
        "head_score_summary":  head_summary,
        "block_sensitivity":   block_sensitivity.tolist(),
        "alpha_per_block":     alpha_per_block.tolist(),
        "exact_validation":    exact_val_results,
        "cascade_results":     [],
    }

    # Track best BF1 for end-of-run visualization
    best_bf1         = -1.0
    best_head_mask   = None
    best_neuron_mask = None

    print("\n[9] Running cascade experiments ...")

    for head_sp in args.head_sparsities:
        print(f"\n{'='*60}")
        print(f"  Head sparsity target: {head_sp*100:.0f}%")

        per_block_sp_head = allocate_nonuniform_head_sparsity(
            block_sensitivity, head_sp,
            num_heads=12, min_keep=args.nonuniform_min_keep_head,
            protected_blocks=args.protected_blocks)
        head_mask       = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
        n_heads_kept    = int(head_mask.sum())
        actual_head_sp  = 1.0 - n_heads_kept / 144.0
        print(f"  Head mask: kept={n_heads_kept}/144  actual_sp={actual_head_sp*100:.1f}%")

        # Restore + apply head mask
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        head_hooks = apply_head_mask_to_model(model, head_mask)

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

        head_only_metrics, head_only_per_sample, _ = _eval_and_collect(
            model, test_loader, device)
        head_only_stats = compute_cascade_stats(model, head_mask, None)
        _print_row(head_sp, 0.0, "head_only", head_only_metrics, head_only_stats)

        if head_only_metrics["mean_boundary_f1"] > best_bf1:
            best_bf1       = head_only_metrics["mean_boundary_f1"]
            best_head_mask = head_mask.copy()
            best_neuron_mask = None

        all_results["cascade_results"].append({
            "phase":          "head_only",
            "head_sp_target": head_sp,
            "head_sp_actual": actual_head_sp,
            **head_only_metrics,
            **head_only_stats,
            "per_sample_metrics": head_only_per_sample,
        })
        _save(all_results)

        # ================================================================
        # Iterative MLP chain (one per alpha)
        # ================================================================
        post_head_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        for mlp_alpha_global in args.mlp_alpha_values:
            # When adaptive alpha is ON, mlp_alpha_global is overridden per block;
            # it still controls the fallback value shown in logs.
            if args.no_adaptive_alpha:
                cur_alpha_b = np.full(12, mlp_alpha_global, dtype=np.float32)
            else:
                cur_alpha_b = alpha_per_block   # data-derived

            tag = (f"cross_adaptive_a{mlp_alpha_global:.1f}"
                   if not (args.no_cross_fisher or args.no_adaptive_alpha)
                   else f"a{mlp_alpha_global:.1f}")
            print(f"\n    ---- alpha≈{mlp_alpha_global:.1f} ({tag}) iterative chain ----")

            model.load_state_dict({k: v.to(device) for k, v in post_head_state.items()})

            cur_fisher_m_list = fisher_m_list
            cur_cross_f_list  = cross_fisher_list
            cur_block_sens    = block_sensitivity
            cur_dz_m_list     = dz_m_list
            cur_dr_m_list     = dr_m_list
            active_mlp_hooks  = []

            for stage_idx, target_mlp_sp in enumerate(args.mlp_sparsities):
                print(f"\n      == stage {stage_idx+1}/{len(args.mlp_sparsities)}: "
                      f"MLP {target_mlp_sp*100:.0f}% ==")

                remove_hooks(active_mlp_hooks)
                active_mlp_hooks = []

                if args.oneshot_mlp and stage_idx > 0:
                    model.load_state_dict(
                        {k: v.to(device) for k, v in post_head_state.items()})
                    cur_fisher_m_list = fisher_m_list
                    cur_cross_f_list  = cross_fisher_list
                    cur_block_sens    = block_sensitivity
                    cur_dz_m_list     = dz_m_list
                    cur_dr_m_list     = dr_m_list

                # MLP scores with adaptive α per block
                q_mlp = combine_scores_dist_adaptive(
                    cur_dz_m_list, cur_dr_m_list, cur_alpha_b,
                    items_per_block=mlp_dim, beta=args.dist_beta)

                per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
                    cur_block_sens, target_mlp_sp,
                    mlp_dim=mlp_dim, min_frac=args.nonuniform_min_frac_mlp,
                    protected_blocks=args.protected_blocks)
                neuron_mask     = generate_neuron_mask_nonuniform(
                    q_mlp, per_block_sp_mlp, mlp_dim=mlp_dim)
                actual_mlp_sp   = 1.0 - float(neuron_mask.sum()) / (12 * mlp_dim)
                active_mlp_hooks = apply_mlp_mask_to_model(model, neuron_mask)

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

                metrics, stage_per_sample, _ = _eval_and_collect(
                    model, test_loader, device)
                stats    = compute_cascade_stats(model, head_mask, neuron_mask)
                iter_tag = "oneshot" if args.oneshot_mlp else "iterative"
                _print_row(head_sp, target_mlp_sp, f"{iter_tag}[{tag}]",
                           metrics, stats)

                if metrics["mean_boundary_f1"] > best_bf1:
                    best_bf1         = metrics["mean_boundary_f1"]
                    best_head_mask   = head_mask.copy()
                    best_neuron_mask = neuron_mask.copy()

                all_results["cascade_results"].append({
                    "phase":          "cascade",
                    "iteration":      iter_tag,
                    "stage_idx":      stage_idx,
                    "head_sp_target": head_sp,
                    "head_sp_actual": actual_head_sp,
                    "mlp_sp_target":  target_mlp_sp,
                    "mlp_sp_actual":  actual_mlp_sp,
                    "mlp_score":      tag,
                    "mlp_alpha":      mlp_alpha_global,
                    "v7_cross_fisher": not args.no_cross_fisher,
                    "v7_adaptive_alpha": not args.no_adaptive_alpha,
                    **metrics,
                    **stats,
                    "per_sample_metrics": stage_per_sample,
                })
                _save(all_results)

                if (not args.oneshot_mlp
                        and args.recompute_fisher_between_stages
                        and stage_idx < len(args.mlp_sparsities) - 1):
                    print(f"        [re-score] recomputing Fisher + MLP scores ...")
                    cur_fisher_m_list = _estimate_fisher_list(model)
                    cur_combined      = sum_fisher_dicts(cur_fisher_m_list)
                    cur_block_sens    = compute_block_sensitivity(cur_combined, 12)
                    if not args.no_cross_fisher:
                        # Recompute cross-Fisher with updated F^M; F^S is global (fixed)
                        # We don't recompute F^S here — it's a fixed reference.
                        # Reload from saved scores.npz fisher_s approximation:
                        # Instead, use a simpler fallback: cross = updated F^M only
                        # (full re-estimation of F^S is too expensive between stages).
                        cur_cross_f_list = cur_fisher_m_list   # fallback between stages
                    else:
                        cur_cross_f_list = cur_fisher_m_list
                    _, _, cur_dz_m_list, cur_dr_m_list = compute_scores_per_subset_v7(
                        model, sam_params, cur_fisher_m_list, cur_cross_f_list)

            remove_hooks(active_mlp_hooks)

        remove_hooks(head_hooks)
        _save(all_results)
        print(f"  [checkpoint] saved → {results_path}")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    _save(all_results)
    print(f"\n[10] Results saved → {results_path}")

    print("\n" + "=" * 120)
    print("v7 RESULT SUMMARY")
    print(f"  cross_fisher={'ON' if not args.no_cross_fisher else 'OFF'}  "
          f"adaptive_alpha={'ON' if not args.no_adaptive_alpha else 'OFF'}  "
          f"n_subsets={args.n_dist_subsets}  β={args.dist_beta}")
    print("=" * 120)
    print(f"\n  {'Config':<55} {'Dice':>6} {'BF1':>6} {'HD95':>7} {'Par%':>6} {'FL%':>6}")
    print("  " + "-" * 95)
    for r in all_results["cascade_results"]:
        if r["phase"] == "head_only":
            lbl = f"HEAD_ONLY h={r['head_sp_target']*100:.0f}%"
        else:
            lbl = (f"h={r['head_sp_target']*100:.0f}% "
                   f"m={r['mlp_sp_target']*100:.0f}%_{r.get('mlp_score','?')}")
        print(f"  {lbl:<55} "
              f"{r['mean_dice']:>6.4f} {r['mean_boundary_f1']:>6.4f} "
              f"{r['mean_hd95']:>7.2f} "
              f"{r.get('param_reduction_pct',0):>6.1f} "
              f"{r.get('flop_reduction_pct',0):>6.1f}")
    print(f"\n  Baseline: Dice={b['mean_dice']:.4f}  "
          f"BF1={b['mean_boundary_f1']:.4f}  HD95={b['mean_hd95']:.2f}")
    print(f"\n  Best BF1 achieved: {best_bf1:.4f}")

    # ------------------------------------------------------------------
    # 11. V7-4: End-of-run visualization
    # ------------------------------------------------------------------
    if best_head_mask is not None and args.n_visualize_samples > 0:
        print(f"\n[11] End-of-run visualization ({args.n_visualize_samples} samples) ...")
        vis_path = os.path.join(args.output_dir, "visualization_v7.png")
        visualize_best_result(
            original_state,
            best_head_mask,
            best_neuron_mask,
            test_loader,
            device,
            vis_path,
            n_samples=args.n_visualize_samples,
        )
    else:
        print("\n[11] No pruning result to visualize (no cascade experiments ran).")


if __name__ == "__main__":
    main()
