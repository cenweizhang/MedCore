# -*- coding: utf-8 -*-
"""
Cascade pruning v8: three-dataset calibration + fixed π_r + macro-avg ranking.

Builds on v7 (Cross-Fisher + adaptive α + β·Var + boundary-aware Fisher +
iterative prune-and-recover + freq-weighted recovery sampling).  The R=3
distribution-aware subsets are now three real polyp datasets rather than
boundary-complexity splits of a single dataset.

Improvements vs v7
------------------
V8-1  Multi-dataset calibration pool
      R=3 natural subsets: Kvasir-SEG + CVC-ColonDB + CVC-ClinicDB.
      Each contributes `cal_sizes[i]` samples; combined pool feeds recovery.

V8-2  User-specified clinical π_r weights (default 0.5/0.3/0.2)
      π_r is now fixed by CLI rather than uniform, so the weighted mean in
      Eq. 30 biases toward clinically dominant datasets.

V8-3  Macro-avg ranking metric
      Each config is evaluated on every dataset individually; the headline
      ranking metric is the macro-avg over the three per-dataset means.
      Pool-level and worst-dataset metrics are reported as diagnostics.

V8-4  Batched test evaluation (eval_bs ≥ 1)
      Fisher estimation remains batch_size=1 (unbiased per-sample gradients);
      evaluation runs at configurable batch size to utilise 140 GB of VRAM.

Usage
-----
    cd /volume/med-train/users/jshan/testSAMpruning
    python -m pilot_dual.run_cascade_v8 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
        --device      cuda:0 \
        --output_dir  results/pilot_cascade_v8 \
        --head_sparsities 0.5 \
        --mlp_sparsities  0.5 \
        --eval_batch_size 16 \
        --recovery_steps  100 \
        --num_workers     4 \
        --n_visualize_per_dataset 2

CLI additions vs v7
-------------------
    --data_roots      path for Kvasir / ColonDB / ClinicDB roots
    --dataset_names   Kvasir ColonDB ClinicDB
    --cal_sizes       64 38 26     (sum = total cal pool)
    --pi_r            0.5 0.3 0.2  (clinical subset weights)
    --eval_batch_size 16           (test-time batch size)
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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import PolypDataset
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
# Shared helpers (unchanged from v7 except _eval_and_collect now batches)
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
def _eval_and_collect(model, test_loader, device, desc="Evaluating"):
    """
    Batched evaluation.  Unlike v7 which assumed batch_size=1, this version
    correctly iterates over the batch dimension and collects per-sample
    metrics.  Fisher estimation still uses batch_size=1 loaders elsewhere.
    """
    model.eval()
    all_metrics = []
    for batch in tqdm(test_loader, desc=desc, leave=False):
        images   = batch["image"].to(device)                # (B, 3, 1024, 1024)
        masks_gt = batch["mask_1024"].numpy()               # (B, 1024, 1024)
        bboxes   = batch["bbox"].to(device).float()         # (B, 4)
        names    = batch.get("name", [""] * images.shape[0])

        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]                     # (B, 1, 4)

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
        # low_res output is single-channel → pred_1024 shape (B, 1, 1024, 1024)
        pred_bin = (torch.sigmoid(pred_1024) > 0.5).cpu().numpy().astype(np.uint8)
        if pred_bin.ndim == 4:
            pred_bin = pred_bin[:, 0]                       # (B, 1024, 1024)

        B = images.shape[0]
        for i in range(B):
            m = compute_all_metrics(pred_bin[i], masks_gt[i].astype(np.uint8))
            m["name"] = names[i] if i < len(names) else ""
            all_metrics.append(m)

    keys = ["dice", "iou", "boundary_f1", "hd95"]
    avg  = {f"mean_{k}": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    avg.update({f"std_{k}":  float(np.std( [m[k] for m in all_metrics])) for k in keys})
    avg["head_sparsity"] = 0.0
    avg["kept_heads"]    = 144
    avg["total_heads"]   = 144
    return avg, all_metrics, {}


def eval_all_datasets(model, per_dataset_test_loaders, dataset_names, device):
    """
    Run evaluation on each dataset independently; return:
      per_dataset : dict {name: {mean_dice, mean_bf1, ...}}
      pool        : sample-count-weighted mean over all test samples
      macro       : mean over dataset-level means (each dataset weight = 1)
      worst       : worst dataset value per metric
      all_metrics : flat list of all per-sample metrics (for downstream use)
    """
    per_dataset = {}
    all_metrics = []
    for loader, name in zip(per_dataset_test_loaders, dataset_names):
        avg, per_sample, _ = _eval_and_collect(
            model, loader, device, desc=f"Eval[{name}]")
        per_dataset[name] = avg
        all_metrics.extend(per_sample)

    keys = ["dice", "iou", "boundary_f1", "hd95"]
    pool  = {f"mean_{k}": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    macro = {f"mean_{k}": float(np.mean([per_dataset[n][f"mean_{k}"]
                                         for n in dataset_names])) for k in keys}
    worst = {f"mean_{k}": float(min(per_dataset[n][f"mean_{k}"]
                                    for n in dataset_names)) for k in keys}
    pool["n_total"]  = len(all_metrics)
    macro["n_sets"]  = len(dataset_names)

    return {
        "per_dataset": per_dataset,
        "pool":        pool,
        "macro":       macro,
        "worst":       worst,
        "all_metrics": all_metrics,
    }


def compute_cal_freq_weights(cal_loader):
    """Boundary-complexity weights for calibration samples (unchanged from v7)."""
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


def _print_row(head_sp, mlp_sp, tag, eval_result, stats):
    """Single-line summary printing macro-avg + per-dataset BF1 breakdown."""
    p  = stats.get("param_reduction_pct", 0.0)
    fl = stats.get("flops_remaining_G",   0.0)
    fr = stats.get("flop_reduction_pct",  0.0)
    m  = eval_result["macro"]
    pd = eval_result["per_dataset"]

    per_bf1 = "  ".join(f"{n[:4]}={pd[n]['mean_boundary_f1']:.3f}" for n in pd)

    print(f"    [{tag}] h={head_sp*100:.0f}% m={mlp_sp*100:.0f}% | "
          f"macro Dice={m['mean_dice']:.4f} BF1={m['mean_boundary_f1']:.4f}  "
          f"Par↓{p:.1f}% FL↓{fr:.1f}% | {per_bf1}")


# ---------------------------------------------------------------------------
# Boundary-aware Fisher (identical to v7)
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
        bce_pp = F.binary_cross_entropy_with_logits(low_res, masks, reduction="none")
        loss   = (weight_map * bce_pp).sum() + _dice_loss_sum(low_res, masks)
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
# Cross-Fisher (V7-1, unchanged)
# ---------------------------------------------------------------------------

def compute_sam_fisher_on_medical(sam_ckpt, dataloader, device,
                                   boundary_weight=3.0):
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
    cross = {}
    for n in fisher_m:
        fm = fisher_m[n].float()
        fs = fisher_s.get(n, torch.zeros_like(fm)).float()
        cross[n] = torch.sqrt(fm * fs + 1e-30)
    return cross


def build_cross_fisher_list(fisher_m_list, fisher_s):
    return [compute_cross_fisher(fm, fisher_s) for fm in fisher_m_list]


# ---------------------------------------------------------------------------
# V8-1: multi-dataset loader builder
# ---------------------------------------------------------------------------

def build_multi_dataset_loaders(data_roots, dataset_names, cal_sizes,
                                 eval_batch_size, seed, num_workers):
    """
    Build calibration and test loaders over N datasets.

    Returns
    -------
    per_dataset_cal_loaders : list[DataLoader], each batch_size=1
        For per-subset Fisher estimation (F^M_r).
    combined_cal_loader     : DataLoader, batch_size=1
        ConcatDataset of all cal subsets; used for SAM cross-Fisher,
        recovery fine-tuning, and exact-reset validation.
    per_dataset_test_loaders : list[DataLoader], each batch_size=eval_batch_size
        For macro-avg and per-dataset diagnostic metrics.
    combined_cal_dataset    : ConcatDataset
        Underlying dataset for recovery sampling.
    combined_cal_freq_weights : np.ndarray shape (sum(cal_sizes),)
        Boundary-complexity sampling weights, in ConcatDataset order.
    """
    assert len(data_roots) == len(dataset_names) == len(cal_sizes), \
        "data_roots, dataset_names, cal_sizes must have equal length."

    per_dataset_cal_loaders  = []
    per_dataset_test_loaders = []
    cal_subsets              = []

    print(f"\n  Building {len(data_roots)} dataset loaders ...")
    for root, name, n_cal in zip(data_roots, dataset_names, cal_sizes):
        cal_full  = PolypDataset(root, bbox_shift=0)
        test_full = PolypDataset(root, bbox_shift=5)
        n_total   = len(cal_full)
        assert n_cal < n_total, \
            f"{name}: cal_size {n_cal} >= dataset size {n_total}"

        gen      = torch.Generator().manual_seed(seed)
        indices  = torch.randperm(n_total, generator=gen).tolist()
        cal_idx  = indices[:n_cal]
        test_idx = indices[n_cal:]

        cal_subset  = Subset(cal_full,  cal_idx)
        test_subset = Subset(test_full, test_idx)
        cal_subsets.append(cal_subset)

        per_dataset_cal_loaders.append(
            DataLoader(cal_subset, batch_size=1, shuffle=False,
                       num_workers=num_workers, pin_memory=True))
        per_dataset_test_loaders.append(
            DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True))

        print(f"    [{name}] root={root}  cal={n_cal}  test={len(test_idx)}")

    combined_cal_dataset = ConcatDataset(cal_subsets)
    combined_cal_loader  = DataLoader(
        combined_cal_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"    [combined] total_cal={len(combined_cal_dataset)}")

    # Compute boundary complexity on the combined pool in ConcatDataset order
    combined_cal_freq_weights = compute_cal_freq_weights(combined_cal_loader)

    return (per_dataset_cal_loaders, combined_cal_loader,
            per_dataset_test_loaders, combined_cal_dataset,
            combined_cal_freq_weights)


# ---------------------------------------------------------------------------
# π_r aware fisher combination + scoring (same helpers as v7, called w/ π_r)
# ---------------------------------------------------------------------------

def sum_fisher_dicts(fisher_list, pi=None):
    R  = len(fisher_list)
    if pi is None:
        pi = np.full(R, 1.0 / R)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    combined = {}
    for n in fisher_list[0]:
        acc = torch.zeros_like(fisher_list[0][n])
        for r, f in enumerate(fisher_list):
            acc = acc + float(pi[r]) * f[n]
        combined[n] = acc
    return combined


def compute_scores_per_subset_v7(model, sam_params, fisher_m_list, cross_fisher_list):
    """Δ_zero uses F^M; Δ_reset uses √(F^M·F^S)."""
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
    R = len(dz_list)
    K = len(dz_list[0])

    alpha_vec = np.repeat(alpha_per_block, items_per_block).astype(np.float32)
    if len(alpha_vec) != K:
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
# V7-3 exact validation (diagnostic, unchanged)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _loss_on_cal_subset(model, cal_loader, device, n_samples=16):
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

        q_rows = slice(h * head_dim,          (h + 1) * head_dim)
        k_rows = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
        v_rows = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
        p_cols = slice(h * head_dim,          (h + 1) * head_dim)

        saved_qkv_w  = attn.qkv.weight.data.clone()
        saved_proj_w = attn.proj.weight.data.clone()
        if has_bias:
            saved_qkv_b = attn.qkv.bias.data.clone()

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

        with torch.no_grad():
            attn.qkv.weight.data.copy_(saved_qkv_w)
            attn.proj.weight.data.copy_(saved_proj_w)
            if has_bias:
                attn.qkv.bias.data.copy_(saved_qkv_b)

    return results


# ---------------------------------------------------------------------------
# V8 visualization: samples from each dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_best_result_multi(original_state, head_mask, neuron_mask,
                                 per_dataset_test_loaders, dataset_names,
                                 medsam_ckpt, device, output_path,
                                 n_per_dataset=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.transform import resize as sk_resize

    def _load_model():
        m = sam_model_registry["vit_b"](checkpoint=medsam_ckpt)
        m.load_state_dict({k: v.clone() for k, v in original_state.items()})
        return m.to(device).eval()

    def _infer(mdl, image, bbox):
        image = image.unsqueeze(0).to(device)
        bbox  = bbox.unsqueeze(0).unsqueeze(0).to(device).float()
        emb     = mdl.image_encoder(image)
        sp, dp  = mdl.prompt_encoder(points=None, boxes=bbox, masks=None)
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

    # Pick evenly-spaced samples per dataset
    picked = []  # list of (dataset_name, sample_dict)
    for loader, name in zip(per_dataset_test_loaders, dataset_names):
        ds    = loader.dataset
        n_tot = len(ds)
        idx   = np.linspace(0, n_tot - 1, min(n_per_dataset, n_tot), dtype=int)
        for i in idx:
            picked.append((name, ds[int(i)]))

    model_b = _load_model()
    model_p = _load_model()
    hooks_h = apply_head_mask_to_model(model_p, head_mask)
    hooks_m = apply_mlp_mask_to_model(model_p, neuron_mask) if neuron_mask is not None else []

    n_pruned_heads = int((1 - head_mask).sum())
    n_pruned_mlp   = int((1 - neuron_mask).sum()) if neuron_mask is not None else 0
    subtitle = (f"Best config: {n_pruned_heads}/144 heads pruned, "
                f"{n_pruned_mlp} MLP neurons pruned  (masks applied without recovery)")

    n_rows = len(picked)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 3.0))
    if n_rows == 1: axes = axes[None, :]
    fig.suptitle(subtitle, fontsize=10, y=1.00)

    for c, title in enumerate(["Image", "GT Mask", "Baseline", "Pruned (no recovery)", "Diff"]):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    for row, (ds_name, s) in enumerate(picked):
        img_np = s["image"].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.ptp() + 1e-8)
        gt_np  = s["mask_1024"].numpy().squeeze().astype(np.uint8)

        pred_b = _infer(model_b, s["image"], s["bbox"])
        pred_p = _infer(model_p, s["image"], s["bbox"])
        diff   = np.abs(pred_b.astype(np.float32) - pred_p.astype(np.float32))

        axes[row, 0].imshow(_d256(img_np, order=1).clip(0, 1))
        axes[row, 1].imshow(_d256(gt_np),   cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(_d256(pred_b),  cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(_d256(pred_p),  cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(_d256(diff),    cmap="hot",  vmin=0, vmax=1)

        for c in range(5): axes[row, c].axis("off")
        label = f"[{ds_name}] {s.get('name','')[:20]}"
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0, labelpad=70, va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [V8-viz] saved → {output_path}")

    remove_hooks(hooks_h + hooks_m)
    del model_b, model_p
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cascade pruning v8 — multi-dataset calibration + clinical π_r + macro-avg ranking")

    # --- Paths and models ---
    parser.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",    default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--output_dir",  default="results/pilot_cascade_v8")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # --- V8: multi-dataset configuration ---
    parser.add_argument("--data_roots", nargs="+",
                        default=["asserts/kvasir-seg/Kvasir-SEG",
                                 "asserts/CVC-ColonDB",
                                 "asserts/CVC-ClinicDB"])
    parser.add_argument("--dataset_names", nargs="+",
                        default=["Kvasir", "ColonDB", "ClinicDB"])
    parser.add_argument("--cal_sizes", nargs="+", type=int,
                        default=[64, 38, 26])
    parser.add_argument("--pi_r", nargs="+", type=float,
                        default=[0.5, 0.3, 0.2],
                        help="Clinical subset weights for π_r in Eq. 30. "
                             "Normalised to sum=1 internally.")
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # --- Sparsity grid ---
    parser.add_argument("--head_sparsities", type=float, nargs="+", default=[0.5])
    parser.add_argument("--mlp_sparsities",  type=float, nargs="+",
                        default=[0.5])
    parser.add_argument("--mlp_alpha_values", type=float, nargs="+", default=[1.0])

    # --- V7 flags (retained) ---
    parser.add_argument("--no_cross_fisher",   action="store_true")
    parser.add_argument("--no_adaptive_alpha", action="store_true")
    parser.add_argument("--exact_validate_topk", type=int, default=0)
    parser.add_argument("--n_visualize_per_dataset", type=int, default=3)

    # --- Scoring fallbacks ---
    parser.add_argument("--phase1_alpha", type=float, default=1.0)
    parser.add_argument("--tau",          type=float, default=0.0)

    # --- Boundary / distribution ---
    parser.add_argument("--boundary_fisher_weight",         type=float, default=3.0)
    parser.add_argument("--recompute_fisher_between_stages", action="store_true")
    parser.add_argument("--oneshot_mlp",                    action="store_true")
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
    parser.add_argument("--uniform_allocation",         action="store_true",
                        help="Ablation: replace inverse-sensitivity nonuniform allocation "
                             "with uniform per-block sparsity (equal sensitivity for all blocks).")

    # --- Phase-separated execution (for parallel sweep orchestration) ---
    parser.add_argument("--phase", choices=["all", "b_only", "c_only"], default="all",
                        help="all: full pipeline (default). "
                             "b_only: Phase B (Fisher+scoring+baseline+cache), exit before sweep. "
                             "c_only: load cached Phase B artifacts, run cascade sweep only.")
    parser.add_argument("--worker_idx", type=int, default=-1,
                        help="When >=0, cascade JSON output uses this suffix. "
                             "Intended for parallel launcher to avoid concurrent writes.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for Phase B artifacts. "
                             "Defaults to output_dir/_phase_b_cache.")

    # --- Phase B parallel fan-out (used by the launcher) ---
    parser.add_argument("--fisher_task", type=str, default=None,
                        help="Internal: compute one Fisher tensor dict and exit. "
                             "Format 'medsam:<subset_idx>' or 'sam:combined'. "
                             "Output is saved as cache_dir/fisher_<tag>.pt.")
    parser.add_argument("--load_precomputed_fisher", action="store_true",
                        help="In Phase B, load fisher_m_<i>.pt and fisher_s.pt from "
                             "cache_dir (produced by --fisher_task workers) instead "
                             "of computing inline.")

    args = parser.parse_args()

    # Input validation
    assert len(args.data_roots) == len(args.dataset_names) == \
           len(args.cal_sizes)   == len(args.pi_r), \
           "data_roots, dataset_names, cal_sizes, pi_r must all have same length."

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.mlp_sparsities = sorted(args.mlp_sparsities)

    # Resolve cache dir (Phase B artifacts for b_only/c_only handoff)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "_phase_b_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Normalise π_r
    pi_r = np.asarray(args.pi_r, dtype=np.float64)
    pi_r = pi_r / pi_r.sum()
    R    = len(pi_r)

    print("=" * 80)
    print("CASCADE PRUNING v8 — multi-dataset + π_r + macro-avg")
    print(f"  Phase    : {args.phase}" +
          (f"  worker_idx={args.worker_idx}" if args.worker_idx >= 0 else ""))
    print(f"  cache_dir: {cache_dir}")
    print(f"  Datasets : {args.dataset_names}")
    print(f"  cal_sizes: {args.cal_sizes}  (total={sum(args.cal_sizes)})")
    print(f"  π_r (raw): {args.pi_r}  →  normalised: {pi_r.tolist()}")
    print(f"  eval_bs  : {args.eval_batch_size}")
    print(f"  [V7-1] cross_fisher   : {'OFF' if args.no_cross_fisher else 'ON'}")
    print(f"  [V7-2] adaptive_alpha : {'OFF' if args.no_adaptive_alpha else 'ON (per-block)'}")
    print(f"  [V7-3] exact_validate : {args.exact_validate_topk} heads")
    print(f"  Boundary fisher λ={args.boundary_fisher_weight}  β={args.dist_beta}")
    print(f"  Head sp: {args.head_sparsities}   MLP sp: {args.mlp_sparsities}")
    print(f"  Protected blocks: {args.protected_blocks}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    # ==================================================================
    # Parallel Fisher fan-out handler
    #   When --fisher_task is set, compute a single Fisher and exit.
    #   The launcher spawns 4 such processes (3 MedSAM subsets + 1 SAM).
    # ==================================================================
    if args.fisher_task is not None:
        task = args.fisher_task
        print(f"\n[fisher_task] Running single task: {task}")
        if task.startswith("medsam:"):
            subset_idx = int(task.split(":", 1)[1])
            assert 0 <= subset_idx < len(args.data_roots), \
                f"fisher_task subset_idx {subset_idx} out of range"
            # Build just one dataset's cal loader
            root = args.data_roots[subset_idx]
            name = args.dataset_names[subset_idx]
            n_cal = args.cal_sizes[subset_idx]
            cal_full = PolypDataset(root, bbox_shift=0)
            gen = torch.Generator().manual_seed(args.seed)
            idx = torch.randperm(len(cal_full), generator=gen).tolist()[:n_cal]
            cal_loader = DataLoader(
                Subset(cal_full, idx), batch_size=1, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            print(f"  Loading MedSAM on {device} for F^M[{name}] (n={n_cal}) ...")
            m = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
            m = m.to(device).eval()
            for p in m.prompt_encoder.parameters(): p.requires_grad_(False)
            for p in m.mask_decoder.parameters():   p.requires_grad_(False)
            fisher = compute_diagonal_fisher_boundary_aware(
                m, cal_loader, device,
                boundary_weight=args.boundary_fisher_weight)
            out_path = os.path.join(cache_dir, f"fisher_m_{subset_idx}.pt")
            torch.save(fisher, out_path)
            print(f"  [fisher_task] F^M[{name}] saved → {out_path}")
        elif task == "sam:combined":
            # Build combined cal loader across all datasets
            subsets = []
            for root, n_cal in zip(args.data_roots, args.cal_sizes):
                cal_full = PolypDataset(root, bbox_shift=0)
                gen = torch.Generator().manual_seed(args.seed)
                idx = torch.randperm(len(cal_full), generator=gen).tolist()[:n_cal]
                subsets.append(Subset(cal_full, idx))
            combined = ConcatDataset(subsets)
            cal_loader = DataLoader(
                combined, batch_size=1, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            print(f"  Loading SAM on {device} for F^S (n={len(combined)}) ...")
            fisher_s = compute_sam_fisher_on_medical(
                args.sam_ckpt, cal_loader, device,
                boundary_weight=args.boundary_fisher_weight)
            out_path = os.path.join(cache_dir, "fisher_s.pt")
            torch.save(fisher_s, out_path)
            print(f"  [fisher_task] F^S saved → {out_path}")
        else:
            raise ValueError(f"Unknown fisher_task spec: {task}. "
                             f"Expected 'medsam:<idx>' or 'sam:combined'.")
        return

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
    # 3. Multi-dataset loaders
    # ------------------------------------------------------------------
    print("\n[3] Building three-dataset loaders ...")
    (per_dataset_cal_loaders, combined_cal_loader,
     per_dataset_test_loaders, combined_cal_dataset,
     combined_cal_freq_weights) = build_multi_dataset_loaders(
        data_roots=args.data_roots,
        dataset_names=args.dataset_names,
        cal_sizes=args.cal_sizes,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    np.save(os.path.join(args.output_dir, "cal_freq_weights.npy"),
            combined_cal_freq_weights)
    if args.recovery_steps == 0 or args.no_freq_sampling:
        combined_cal_freq_weights = None

    # ==================================================================
    # Phase B: baseline + Fisher + scoring
    #   phase=="all"/"b_only" → run it; phase=="c_only" → load from cache
    # ==================================================================
    if args.phase in ("all", "b_only"):
        # ------------------------------------------------------------------
        # 4. Baseline evaluation — macro-avg + per-dataset + pool
        # ------------------------------------------------------------------
        print("\n[4] Baseline evaluation (per-dataset + macro) ...")
        baseline_eval = eval_all_datasets(
            model, per_dataset_test_loaders, args.dataset_names, device)
        bm = baseline_eval["macro"]; bp = baseline_eval["pool"]
        print(f"  macro Dice={bm['mean_dice']:.4f}  BF1={bm['mean_boundary_f1']:.4f}  "
              f"IoU={bm['mean_iou']:.4f}  HD95={bm['mean_hd95']:.2f}")
        print(f"  pool  Dice={bp['mean_dice']:.4f}  BF1={bp['mean_boundary_f1']:.4f}  "
              f"(N={bp['n_total']})")
        for n in args.dataset_names:
            pd = baseline_eval["per_dataset"][n]
            print(f"  [{n}] Dice={pd['mean_dice']:.4f}  BF1={pd['mean_boundary_f1']:.4f}  "
                  f"IoU={pd['mean_iou']:.4f}  HD95={pd['mean_hd95']:.2f}")

        # ------------------------------------------------------------------
        # 5. Per-dataset MedSAM Fisher (F^M_r)  →  each dataset IS a subset
        # ------------------------------------------------------------------
        if args.load_precomputed_fisher:
            print(f"\n[5] Loading precomputed F^M from {cache_dir} ...")
            fisher_m_list = []
            for i, name in enumerate(args.dataset_names):
                p = os.path.join(cache_dir, f"fisher_m_{i}.pt")
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Precomputed Fisher missing: {p}. "
                        f"Run `--fisher_task medsam:{i}` worker first.")
                print(f"    F^M subset {i+1}/{R} [{name}] ← {p}")
                fisher_m_list.append(torch.load(p, map_location="cpu"))
        else:
            print(f"\n[5] Computing MedSAM boundary-aware Fisher for {R} subsets ...")
            t0 = time.time()
            fisher_m_list = []
            for i, (loader, name) in enumerate(zip(per_dataset_cal_loaders, args.dataset_names)):
                print(f"    F^M subset {i+1}/{R} [{name}] (n={len(loader.dataset)}) ...")
                fisher_m_list.append(compute_diagonal_fisher_boundary_aware(
                    model, loader, device, boundary_weight=args.boundary_fisher_weight))
            print(f"  F^M done: {time.time()-t0:.1f}s")

        # π_r-weighted Fisher combination for block sensitivity
        fisher_m_combined = sum_fisher_dicts(fisher_m_list, pi=pi_r)
        block_sensitivity  = compute_block_sensitivity(fisher_m_combined, num_blocks=12)
        np.save(os.path.join(args.output_dir, "block_sensitivity.npy"), block_sensitivity)
        print(f"  block_sensitivity (π_r-weighted): "
              f"[{', '.join(f'{s:.3f}' for s in block_sensitivity)}]")

        # ------------------------------------------------------------------
        # 5b. SAM Fisher on combined medical cal set  →  cross-Fisher
        # ------------------------------------------------------------------
        if not args.no_cross_fisher:
            if args.load_precomputed_fisher:
                p = os.path.join(cache_dir, "fisher_s.pt")
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Precomputed SAM Fisher missing: {p}. "
                        f"Run `--fisher_task sam:combined` worker first.")
                print(f"\n[5b] Loading precomputed F^S ← {p}")
                fisher_s = torch.load(p, map_location="cpu")
            else:
                print("\n[5b] Computing SAM Fisher on combined medical cal (F^S) ...")
                t0 = time.time()
                fisher_s = compute_sam_fisher_on_medical(
                    args.sam_ckpt, combined_cal_loader, device,
                    boundary_weight=args.boundary_fisher_weight)
                print(f"  F^S done: {time.time()-t0:.1f}s")
            cross_fisher_list = build_cross_fisher_list(fisher_m_list, fisher_s)
            del fisher_s
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print("  Cross-Fisher list built.")
        else:
            cross_fisher_list = fisher_m_list
            print("\n[5b] cross-Fisher DISABLED — using F^M for Δ_reset.")

        # ------------------------------------------------------------------
        # 6. Per-subset scores
        # ------------------------------------------------------------------
        print("\n[6] Computing dual-intervention scores per subset ...")
        dz_h_list, dr_h_list, dz_m_list, dr_m_list = compute_scores_per_subset_v7(
            model, sam_params, fisher_m_list, cross_fisher_list)

        # Mean scores (using π_r) for adaptive α + diagnostics
        dz_head_mean = sum(float(pi_r[i]) * dz_h_list[i] for i in range(R)).astype(np.float32)
        dr_head_mean = sum(float(pi_r[i]) * dr_h_list[i] for i in range(R)).astype(np.float32)

        # ------------------------------------------------------------------
        # 6b. Per-block adaptive α  (from π_r-weighted mean)
        # ------------------------------------------------------------------
        if not args.no_adaptive_alpha:
            alpha_per_block = compute_adaptive_alpha_per_block(
                dz_head_mean, dr_head_mean)
            print("\n[6b] Per-block adaptive α (from π_r-weighted Δ̄):")
            for bi, ab in enumerate(alpha_per_block):
                mark = " ←" if abs(ab - 1.0) > 0.05 else ""
                print(f"  Block {bi:2d}: α={ab:.3f}{mark}")
        else:
            alpha_per_block = np.full(12, args.phase1_alpha, dtype=np.float32)
            print(f"\n[6b] Adaptive α DISABLED — using global α={args.phase1_alpha}.")

        mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

        # Final head scores: π_r-weighted + β·Var
        head_scores = combine_scores_dist_adaptive(
            dz_h_list, dr_h_list, alpha_per_block,
            items_per_block=12, pi=pi_r, beta=args.dist_beta)

        head_summary = score_summary(dz_head_mean, dr_head_mean)

        # ------------------------------------------------------------------
        # 6c. Exact validation (diagnostic, off by default)
        # ------------------------------------------------------------------
        exact_val_results = {}
        if args.exact_validate_topk > 0:
            print(f"\n[6c] Exact group-level validation ...")
            exact_val_results = exact_group_reset_validation(
                model, sam_params, dr_head_mean, combined_cal_loader, device,
                topk=args.exact_validate_topk, n_validate=16)

        # Save scores
        np.savez(
            os.path.join(args.output_dir, "scores.npz"),
            delta_zero_head=dz_head_mean,
            delta_reset_head=dr_head_mean,
            delta_zero_mlp=sum(float(pi_r[i]) * dz_m_list[i] for i in range(R)).astype(np.float32),
            delta_reset_mlp=sum(float(pi_r[i]) * dr_m_list[i] for i in range(R)).astype(np.float32),
            dz_head_per_subset=np.stack(dz_h_list, axis=0),
            dr_head_per_subset=np.stack(dr_h_list, axis=0),
            dz_mlp_per_subset=np.stack(dz_m_list, axis=0),
            dr_mlp_per_subset=np.stack(dr_m_list, axis=0),
            block_sensitivity=block_sensitivity,
            alpha_per_block=alpha_per_block,
            pi_r=pi_r.astype(np.float32),
            dist_beta=np.float32(args.dist_beta),
        )

        # ------------------------------------------------------------------
        # Save Phase B cache (for c_only workers)
        # ------------------------------------------------------------------
        print(f"\n[Phase B cache] Saving artifacts to {cache_dir} ...")
        # scores.npz duplicated into cache_dir so workers only need cache_dir
        import shutil as _shutil
        _shutil.copy(os.path.join(args.output_dir, "scores.npz"),
                     os.path.join(cache_dir, "scores.npz"))
        torch.save(original_state, os.path.join(cache_dir, "original_state.pt"))
        with open(os.path.join(cache_dir, "baseline_eval.json"), "w") as _fp:
            json.dump(json.loads(json.dumps(baseline_eval, default=_json_safe)),
                      _fp, indent=2)
        with open(os.path.join(cache_dir, "exact_val_results.json"), "w") as _fp:
            json.dump(json.loads(json.dumps(exact_val_results, default=_json_safe)),
                      _fp, indent=2)
        print(f"[Phase B cache] done.")

        if args.phase == "b_only":
            print(f"\n[exit] --phase b_only: cache ready at {cache_dir}. "
                  f"Launch workers with --phase c_only.")
            return

    else:
        # args.phase == "c_only": load cached Phase B artifacts
        print(f"\n[Phase C worker {args.worker_idx}] Loading cached artifacts from {cache_dir} ...")
        sc = np.load(os.path.join(cache_dir, "scores.npz"))
        dz_head_mean    = sc["delta_zero_head"]
        dr_head_mean    = sc["delta_reset_head"]
        block_sensitivity = sc["block_sensitivity"]
        alpha_per_block = sc["alpha_per_block"]
        pi_r_cached     = sc["pi_r"].astype(np.float64)
        dz_h_list = [sc["dz_head_per_subset"][i] for i in range(sc["dz_head_per_subset"].shape[0])]
        dr_h_list = [sc["dr_head_per_subset"][i] for i in range(sc["dr_head_per_subset"].shape[0])]
        dz_m_list = [sc["dz_mlp_per_subset"][i]  for i in range(sc["dz_mlp_per_subset"].shape[0])]
        dr_m_list = [sc["dr_mlp_per_subset"][i]  for i in range(sc["dr_mlp_per_subset"].shape[0])]

        # Sanity: π_r from CLI must match cache (the cascade recombination uses CLI π_r)
        if not np.allclose(pi_r, pi_r_cached, atol=1e-6):
            print(f"  WARNING: CLI π_r {pi_r.tolist()} differs from cached "
                  f"{pi_r_cached.tolist()}. Using CLI value for combination.")

        head_scores = combine_scores_dist_adaptive(
            dz_h_list, dr_h_list, alpha_per_block,
            items_per_block=12, pi=pi_r, beta=args.dist_beta)

        head_summary = score_summary(dz_head_mean, dr_head_mean)
        mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

        # Load cached baseline + exact val
        with open(os.path.join(cache_dir, "baseline_eval.json")) as _fp:
            baseline_eval = json.load(_fp)
        try:
            with open(os.path.join(cache_dir, "exact_val_results.json")) as _fp:
                exact_val_results = json.load(_fp)
        except FileNotFoundError:
            exact_val_results = {}

        # Load original_state (model weights at pre-pruning baseline)
        loaded_state = torch.load(os.path.join(cache_dir, "original_state.pt"),
                                  map_location="cpu")
        original_state = {k: v.clone() for k, v in loaded_state.items()}
        # Also push into GPU model, so evaluation matches b_only baseline
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})

        # Fisher lists are not available in c_only — disable --recompute_fisher_between_stages
        if args.recompute_fisher_between_stages:
            print("  NOTE: --recompute_fisher_between_stages disabled in --phase c_only "
                  "(Fisher tensors not cached).")
            args.recompute_fisher_between_stages = False
        fisher_m_list     = None
        cross_fisher_list = None

        bm = baseline_eval["macro"]; bp = baseline_eval["pool"]
        print(f"  Cached baseline: macro Dice={bm['mean_dice']:.4f}  "
              f"BF1={bm['mean_boundary_f1']:.4f}")

    # ------------------------------------------------------------------
    # 7. Teacher for distillation
    # ------------------------------------------------------------------
    feature_teacher = None
    if args.recovery_steps > 0 and (args.feat_distill_weight > 0
                                     or args.logit_distill_weight > 0):
        print("\n[7] Building teacher model (CPU) ...")
        feature_teacher = copy.deepcopy(model).cpu().eval()
        for p in feature_teacher.parameters(): p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 8. Cascade experiment loop
    # ------------------------------------------------------------------
    if args.worker_idx >= 0:
        results_path = os.path.join(
            args.output_dir, f"cascade_results_v8_w{args.worker_idx:02d}.json")
    else:
        results_path = os.path.join(args.output_dir, "cascade_results_v8.json")

    def _save(obj):
        with open(results_path, "w") as fp:
            json.dump(json.loads(json.dumps(obj, default=_json_safe)), fp, indent=2)

    all_results = {
        "config":              vars(args),
        "baseline":            baseline_eval["macro"],
        "baseline_per_dataset": baseline_eval["per_dataset"],
        "baseline_pool":       baseline_eval["pool"],
        "head_score_summary":  head_summary,
        "block_sensitivity":   block_sensitivity.tolist(),
        "alpha_per_block":     alpha_per_block.tolist(),
        "pi_r":                pi_r.tolist(),
        "exact_validation":    exact_val_results,
        "cascade_results":     [],
    }

    # Track best macro-BF1
    best_macro_bf1   = -1.0
    best_head_mask   = None
    best_neuron_mask = None
    best_tag         = None

    print("\n[8] Running cascade experiments ...")

    for head_sp in args.head_sparsities:
        print(f"\n{'='*60}")
        print(f"  Head sparsity target: {head_sp*100:.0f}%")

        _blk_sens_head = (np.ones_like(block_sensitivity)
                          if args.uniform_allocation else block_sensitivity)
        per_block_sp_head = allocate_nonuniform_head_sparsity(
            _blk_sens_head, head_sp,
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
            print(f"    [Phase-1 recovery] {args.recovery_steps} steps on combined cal ...")
            recovery_finetune(
                model, combined_cal_loader, device,
                head_mask=head_mask, neuron_mask=None,
                n_steps=args.recovery_steps, lr=args.recovery_lr,
                feature_teacher=feature_teacher,
                feat_distill_weight=args.feat_distill_weight,
                boundary_loss_weight=args.boundary_loss_weight,
                logit_distill_weight=args.logit_distill_weight,
                freq_pred_loss_weight=args.freq_pred_loss_weight,
                freq_weights=combined_cal_freq_weights,
            )

        head_only_eval  = eval_all_datasets(
            model, per_dataset_test_loaders, args.dataset_names, device)
        head_only_stats = compute_cascade_stats(model, head_mask, None)
        _print_row(head_sp, 0.0, "head_only", head_only_eval, head_only_stats)

        macro_bf1 = head_only_eval["macro"]["mean_boundary_f1"]
        if macro_bf1 > best_macro_bf1:
            best_macro_bf1   = macro_bf1
            best_head_mask   = head_mask.copy()
            best_neuron_mask = None
            best_tag         = f"head_only_h{int(head_sp*100)}"

        all_results["cascade_results"].append({
            "phase":          "head_only",
            "head_sp_target": head_sp,
            "head_sp_actual": actual_head_sp,
            "macro":          head_only_eval["macro"],
            "pool":           head_only_eval["pool"],
            "worst":          head_only_eval["worst"],
            "per_dataset":    head_only_eval["per_dataset"],
            **head_only_stats,
        })
        _save(all_results)

        # ================================================================
        # Iterative MLP chain (one per alpha)
        # ================================================================
        post_head_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        for mlp_alpha_global in args.mlp_alpha_values:
            if args.no_adaptive_alpha:
                cur_alpha_b = np.full(12, mlp_alpha_global, dtype=np.float32)
            else:
                cur_alpha_b = alpha_per_block

            tag = (f"v8_pi_a{mlp_alpha_global:.1f}"
                   if not (args.no_cross_fisher or args.no_adaptive_alpha)
                   else f"a{mlp_alpha_global:.1f}")
            print(f"\n    ---- α≈{mlp_alpha_global:.1f} ({tag}) iterative chain ----")

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

                q_mlp = combine_scores_dist_adaptive(
                    cur_dz_m_list, cur_dr_m_list, cur_alpha_b,
                    items_per_block=mlp_dim, pi=pi_r, beta=args.dist_beta)

                _blk_sens_mlp = (np.ones_like(cur_block_sens)
                                 if args.uniform_allocation else cur_block_sens)
                per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
                    _blk_sens_mlp, target_mlp_sp,
                    mlp_dim=mlp_dim, min_frac=args.nonuniform_min_frac_mlp,
                    protected_blocks=args.protected_blocks)
                neuron_mask     = generate_neuron_mask_nonuniform(
                    q_mlp, per_block_sp_mlp, mlp_dim=mlp_dim)
                actual_mlp_sp   = 1.0 - float(neuron_mask.sum()) / (12 * mlp_dim)
                active_mlp_hooks = apply_mlp_mask_to_model(model, neuron_mask)

                if args.recovery_steps > 0:
                    print(f"        recovery ({args.recovery_steps} steps) ...")
                    recovery_finetune(
                        model, combined_cal_loader, device,
                        head_mask=None, neuron_mask=None,
                        n_steps=args.recovery_steps, lr=args.recovery_lr,
                        feature_teacher=feature_teacher,
                        feat_distill_weight=args.feat_distill_weight,
                        boundary_loss_weight=args.boundary_loss_weight,
                        logit_distill_weight=args.logit_distill_weight,
                        freq_pred_loss_weight=args.freq_pred_loss_weight,
                        freq_weights=combined_cal_freq_weights,
                    )

                stage_eval = eval_all_datasets(
                    model, per_dataset_test_loaders, args.dataset_names, device)
                stats    = compute_cascade_stats(model, head_mask, neuron_mask)
                iter_tag = "oneshot" if args.oneshot_mlp else "iterative"
                _print_row(head_sp, target_mlp_sp, f"{iter_tag}[{tag}]",
                           stage_eval, stats)

                macro_bf1 = stage_eval["macro"]["mean_boundary_f1"]
                if macro_bf1 > best_macro_bf1:
                    best_macro_bf1   = macro_bf1
                    best_head_mask   = head_mask.copy()
                    best_neuron_mask = neuron_mask.copy()
                    best_tag         = f"{iter_tag}_h{int(head_sp*100)}_m{int(target_mlp_sp*100)}"

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
                    "macro":          stage_eval["macro"],
                    "pool":           stage_eval["pool"],
                    "worst":          stage_eval["worst"],
                    "per_dataset":    stage_eval["per_dataset"],
                    **stats,
                })
                _save(all_results)

                if (not args.oneshot_mlp
                        and args.recompute_fisher_between_stages
                        and stage_idx < len(args.mlp_sparsities) - 1):
                    print(f"        [re-score] recomputing F^M per subset + MLP scores ...")
                    cur_fisher_m_list = []
                    for loader in per_dataset_cal_loaders:
                        cur_fisher_m_list.append(compute_diagonal_fisher_boundary_aware(
                            model, loader, device,
                            boundary_weight=args.boundary_fisher_weight))
                    cur_combined  = sum_fisher_dicts(cur_fisher_m_list, pi=pi_r)
                    cur_block_sens = compute_block_sensitivity(cur_combined, 12)
                    if not args.no_cross_fisher:
                        # F^S fixed; skip re-estimating SAM to save compute
                        cur_cross_f_list = cur_fisher_m_list
                    else:
                        cur_cross_f_list = cur_fisher_m_list
                    _, _, cur_dz_m_list, cur_dr_m_list = compute_scores_per_subset_v7(
                        model, sam_params, cur_fisher_m_list, cur_cross_f_list)

            remove_hooks(active_mlp_hooks)

        remove_hooks(head_hooks)
        _save(all_results)
        print(f"  [checkpoint] saved → {results_path}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    _save(all_results)
    print(f"\n[9] Results saved → {results_path}")

    print("\n" + "=" * 130)
    print("v8 RESULT SUMMARY")
    print(f"  π_r={pi_r.tolist()}  β={args.dist_beta}  "
          f"cross_fisher={'ON' if not args.no_cross_fisher else 'OFF'}  "
          f"adaptive_alpha={'ON' if not args.no_adaptive_alpha else 'OFF'}")
    print("=" * 130)
    print(f"\n  {'Config':<40} {'macro_D':>8} {'macro_BF1':>10} {'worst_BF1':>10} "
          f"{'pool_BF1':>10} {'Par%':>6} {'FL%':>6}")
    print("  " + "-" * 110)
    for r in all_results["cascade_results"]:
        if r["phase"] == "head_only":
            lbl = f"HEAD_ONLY h={r['head_sp_target']*100:.0f}%"
        else:
            lbl = (f"h={r['head_sp_target']*100:.0f}% "
                   f"m={r['mlp_sp_target']*100:.0f}% {r.get('mlp_score','?')}")
        ma = r["macro"]; wo = r["worst"]; po = r["pool"]
        print(f"  {lbl:<40} {ma['mean_dice']:>8.4f} {ma['mean_boundary_f1']:>10.4f} "
              f"{wo['mean_boundary_f1']:>10.4f} {po['mean_boundary_f1']:>10.4f} "
              f"{r.get('param_reduction_pct',0):>6.1f} "
              f"{r.get('flop_reduction_pct',0):>6.1f}")
    print(f"\n  Baseline: macro Dice={bm['mean_dice']:.4f}  "
          f"BF1={bm['mean_boundary_f1']:.4f}  HD95={bm['mean_hd95']:.2f}")
    print(f"\n  Best macro-BF1: {best_macro_bf1:.4f}  (config: {best_tag})")

    # ------------------------------------------------------------------
    # 10. Visualization: samples from each dataset
    #   Skipped for parallel workers (worker_idx >= 0) — the launcher runs
    #   a final visualization after merging per-worker results.
    # ------------------------------------------------------------------
    if args.worker_idx >= 0:
        print(f"\n[10] Skipping visualization (worker {args.worker_idx}); "
              f"launcher will render after merge.")
    elif best_head_mask is not None and args.n_visualize_per_dataset > 0:
        print(f"\n[10] Visualization ({args.n_visualize_per_dataset} samples/dataset) ...")
        vis_path = os.path.join(args.output_dir, "visualization_v8.png")
        visualize_best_result_multi(
            original_state, best_head_mask, best_neuron_mask,
            per_dataset_test_loaders, args.dataset_names,
            args.medsam_ckpt, device, vis_path,
            n_per_dataset=args.n_visualize_per_dataset,
        )
    else:
        print("\n[10] No pruning result to visualize.")


if __name__ == "__main__":
    main()
