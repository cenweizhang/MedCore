# -*- coding: utf-8 -*-
"""
Diagonal Fisher estimation and dual-intervention scoring.

Core equations (see paper Appendix B):
  F_i    = (1/N) Σ_n (∂ℓ/∂θ_i)²                        Eq. 24
  Δ_zero_g  = 0.5 Σ_{i∈g} F_i · θ_i²                   Eq. 25
  Δ_reset_g = 0.5 Σ_{i∈g} F_i · (θ_i − θ_i^S)²        Eq. 26
  Q_g   = α · Δ_zero_g + (1−α) · Δ_reset_g              Eq. 27

Attention head group g^att_{l,h} (Eq. 32):
  qkv.weight  rows for Q/K/V of head h
  qkv.bias    rows for Q/K/V of head h  (if bias exists)
  proj.weight cols for head h

MLP neuron group g^mlp_{l,n}:
  lin1.weight[n, :]
  lin1.bias[n]  (if bias exists)
  lin2.weight[:, n]
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dice_loss_sum(pred_logits, target):
    pred   = torch.sigmoid(pred_logits)
    pred   = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)
    return per.sum()


# ---------------------------------------------------------------------------
# Step 1: Boundary-aware diagonal Fisher estimation
# ---------------------------------------------------------------------------

def compute_diagonal_fisher(model, dataloader, device, boundary_weight=3.0):
    """
    Estimate the diagonal Fisher for all image-encoder parameters with
    boundary-aware loss weighting.

    Uses batch_size=1 for exact per-sample gradients.  Larger batches
    introduce bias because autograd returns gradient of the mean loss.

    Args:
        model           : MedSAM (Sam) model on `device`.
        dataloader      : calibration DataLoader (batch_size=1).
        device          : torch.device.
        boundary_weight : weight multiplier on boundary pixels (λ_b, default 3).

    Returns:
        fisher : dict {param_name → tensor}  (relative to image_encoder)
    """
    model.eval()
    for p in model.parameters():             p.requires_grad_(False)
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
            dilated    = F.conv2d(masks,         k3, padding=1).clamp(0, 1)
            eroded     = 1.0 - F.conv2d(1.0 - masks, k3, padding=1).clamp(0, 1)
            weight_map = 1.0 + boundary_weight * (dilated - eroded)

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

    if nan_batches:
        print(f"  WARNING: {nan_batches} batches skipped (NaN grads).")
    for n in fisher:
        fisher[n] /= max(n_processed, 1)
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)
    return fisher


# ---------------------------------------------------------------------------
# Step 2: SAM reference parameters θ^S
# ---------------------------------------------------------------------------

def load_sam_encoder_params(sam_checkpoint_path, device="cpu"):
    """
    Load the original SAM ViT-B checkpoint and return the image-encoder
    parameter dict {param_name → float tensor on CPU}.
    """
    from segment_anything import sam_model_registry
    print(f"  Loading SAM checkpoint: {sam_checkpoint_path}")
    sam_model  = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
    sam_params = {n: p.data.clone().float().cpu()
                  for n, p in sam_model.image_encoder.named_parameters()}
    del sam_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return sam_params


# ---------------------------------------------------------------------------
# Step 3: Dual-intervention scores
# ---------------------------------------------------------------------------

def compute_head_scores(model, sam_params, fisher, num_blocks=12, num_heads=12):
    """
    Compute Δ_zero_g and Δ_reset_g (Eq. 25–26) for every attention head.

    Returns:
        delta_zero  : np.ndarray (num_blocks * num_heads,) float32
        delta_reset : np.ndarray (num_blocks * num_heads,) float32
    """
    total = num_blocks * num_heads
    delta_zero  = np.zeros(total, dtype=np.float32)
    delta_reset = np.zeros(total, dtype=np.float32)

    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads

        qkv_w_key  = f"blocks.{l}.attn.qkv.weight"
        qkv_b_key  = f"blocks.{l}.attn.qkv.bias"
        proj_w_key = f"blocks.{l}.attn.proj.weight"

        theta_qkv_w  = attn.qkv.weight.data.float().cpu()
        theta_proj_w = attn.proj.weight.data.float().cpu()
        has_bias     = attn.qkv.bias is not None
        if has_bias:
            theta_qkv_b = attn.qkv.bias.data.float().cpu()

        sam_qkv_w  = sam_params.get(qkv_w_key)
        sam_proj_w = sam_params.get(proj_w_key)
        if has_bias:
            sam_qkv_b = sam_params.get(qkv_b_key)

        F_qkv_w  = fisher.get(qkv_w_key,  torch.zeros_like(theta_qkv_w))
        F_proj_w = fisher.get(proj_w_key, torch.zeros_like(theta_proj_w))
        if has_bias:
            F_qkv_b = fisher.get(qkv_b_key, torch.zeros_like(theta_qkv_b))

        for h in range(num_heads):
            head_id = l * num_heads + h
            q = slice(h * head_dim,           (h + 1) * head_dim)
            k = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
            v = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
            p = slice(h * head_dim,           (h + 1) * head_dim)

            theta_g = torch.cat([theta_qkv_w[q].reshape(-1), theta_qkv_w[k].reshape(-1),
                                  theta_qkv_w[v].reshape(-1), theta_proj_w[:, p].reshape(-1)])
            F_g     = torch.cat([F_qkv_w[q].reshape(-1), F_qkv_w[k].reshape(-1),
                                  F_qkv_w[v].reshape(-1), F_proj_w[:, p].reshape(-1)])
            if has_bias:
                theta_g = torch.cat([theta_g, theta_qkv_b[q], theta_qkv_b[k], theta_qkv_b[v]])
                F_g     = torch.cat([F_g,     F_qkv_b[q],     F_qkv_b[k],     F_qkv_b[v]])

            if sam_qkv_w is not None and sam_proj_w is not None:
                sam_g = torch.cat([sam_qkv_w[q].reshape(-1), sam_qkv_w[k].reshape(-1),
                                    sam_qkv_w[v].reshape(-1), sam_proj_w[:, p].reshape(-1)])
                if has_bias and sam_qkv_b is not None:
                    sam_g = torch.cat([sam_g, sam_qkv_b[q], sam_qkv_b[k], sam_qkv_b[v]])
            else:
                sam_g = torch.zeros_like(theta_g)

            dg = theta_g - sam_g
            delta_zero[head_id]  = float(0.5 * (F_g * theta_g ** 2).sum())
            delta_reset[head_id] = float(0.5 * (F_g * dg       ** 2).sum())

    return delta_zero, delta_reset


def compute_mlp_neuron_scores(model, sam_params, fisher, num_blocks=12):
    """
    Compute Δ_zero_g and Δ_reset_g (Eq. 25–26) for every MLP neuron.

    Group g^mlp_{l,n}: lin1.weight[n,:], lin1.bias[n], lin2.weight[:,n].

    Returns:
        delta_zero  : np.ndarray (num_blocks * mlp_dim,) float32
        delta_reset : np.ndarray (num_blocks * mlp_dim,) float32
    """
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]
    total   = num_blocks * mlp_dim
    delta_zero  = np.zeros(total, dtype=np.float32)
    delta_reset = np.zeros(total, dtype=np.float32)

    for l in range(num_blocks):
        mlp = model.image_encoder.blocks[l].mlp
        l1w_key = f"blocks.{l}.mlp.lin1.weight"
        l1b_key = f"blocks.{l}.mlp.lin1.bias"
        l2w_key = f"blocks.{l}.mlp.lin2.weight"

        theta_l1w = mlp.lin1.weight.data.float().cpu()
        theta_l2w = mlp.lin2.weight.data.float().cpu()
        has_bias  = mlp.lin1.bias is not None

        theta_g = torch.cat([theta_l1w, theta_l2w.t()], dim=1)
        if has_bias:
            theta_g = torch.cat([theta_g, mlp.lin1.bias.data.float().cpu().unsqueeze(1)], dim=1)

        F_l1w = fisher.get(l1w_key, torch.zeros_like(theta_l1w))
        F_l2w = fisher.get(l2w_key, torch.zeros_like(theta_l2w))
        F_g   = torch.cat([F_l1w, F_l2w.t()], dim=1)
        if has_bias:
            F_lb = fisher.get(l1b_key, torch.zeros(mlp_dim))
            F_g  = torch.cat([F_g, F_lb.float().unsqueeze(1)], dim=1)

        sam_l1w = sam_params.get(l1w_key)
        sam_l2w = sam_params.get(l2w_key)
        if sam_l1w is not None and sam_l2w is not None:
            sam_g = torch.cat([sam_l1w.float(), sam_l2w.t().float()], dim=1)
            if has_bias:
                sam_lb = sam_params.get(l1b_key)
                col = (sam_lb.float().unsqueeze(1) if sam_lb is not None
                       else torch.zeros(mlp_dim, 1))
                sam_g = torch.cat([sam_g, col], dim=1)
        else:
            sam_g = torch.zeros_like(theta_g)

        dg = theta_g - sam_g
        sl = slice(l * mlp_dim, (l + 1) * mlp_dim)
        delta_zero[sl]  = (0.5 * (F_g * theta_g ** 2).sum(dim=1)).numpy()
        delta_reset[sl] = (0.5 * (F_g * dg       ** 2).sum(dim=1)).numpy()

    return delta_zero, delta_reset


# ---------------------------------------------------------------------------
# Step 4: Multi-subset combination with distribution-shift penalty
# ---------------------------------------------------------------------------

def combine_scores_dist_adaptive(dz_list, dr_list, alpha_per_block,
                                  items_per_block, pi=None, beta=0.3):
    """
    Q = π-weighted mean of per-subset Q_r  +  β · Var_r(Q_r)

    Eq. 30: Q_g = Σ_r π_r · Q_r_g  +  β · Var_r(Q_r_g)

    The β·Var term penalises groups whose importance is inconsistent across
    datasets (distribution shift), making pruning decisions more conservative
    for such groups.

    Args:
        dz_list, dr_list : list of R arrays (num_blocks * items_per_block,)
        alpha_per_block  : (num_blocks,) per-block α
        items_per_block  : 12 for heads, mlp_dim for neurons
        pi               : (R,) clinical weights (normalised internally)
        beta             : distribution-shift penalty weight

    Returns:
        scores : (num_blocks * items_per_block,) float32, lower = prune first
    """
    R = len(dz_list)
    K = len(dz_list[0])

    alpha_vec = np.repeat(alpha_per_block, items_per_block).astype(np.float32)
    if len(alpha_vec) != K:
        alpha_vec = np.full(K, float(alpha_per_block.mean()), dtype=np.float32)

    Q_r = np.stack([alpha_vec * dz + (1.0 - alpha_vec) * dr
                    for dz, dr in zip(dz_list, dr_list)], axis=0)   # (R, K)

    if pi is None:
        pi = np.full(R, 1.0 / R, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64) / np.sum(pi)

    Q_mean = (pi[:, None] * Q_r).sum(axis=0)
    if beta == 0.0 or R == 1:
        return Q_mean.astype(np.float32)
    return (Q_mean + beta * Q_r.var(axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Per-block adaptive α  (V7-2)
# ---------------------------------------------------------------------------

def compute_adaptive_alpha_per_block(dz_head, dr_head, num_blocks=12, num_heads=12):
    """
    Per-block α = (1 + corr(Δ_zero, Δ_reset)) / 2  within each block.

    α≈1  → Δ_zero dominates (Δ_zero and Δ_reset are nearly identical).
    α≈0  → Δ_reset dominates (structural gap between MedSAM and SAM).
    """
    alpha_b = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        dz_b = dz_head[b * num_heads: (b + 1) * num_heads]
        dr_b = dr_head[b * num_heads: (b + 1) * num_heads]
        if dz_b.std() > 1e-12 and dr_b.std() > 1e-12:
            r = float(np.clip(np.corrcoef(dz_b, dr_b)[0, 1], -1.0, 1.0))
        else:
            r = 1.0
        alpha_b[b] = (1.0 + r) / 2.0
    return alpha_b


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def score_summary(delta_zero, delta_reset):
    corr = float(np.dot(delta_zero - delta_zero.mean(),
                        delta_reset - delta_reset.mean()) /
                 (np.linalg.norm(delta_zero - delta_zero.mean()) *
                  np.linalg.norm(delta_reset - delta_reset.mean()) + 1e-12))
    return {
        "delta_zero_mean":        float(delta_zero.mean()),
        "delta_zero_std":         float(delta_zero.std()),
        "delta_reset_mean":       float(delta_reset.mean()),
        "delta_reset_std":        float(delta_reset.std()),
        "correlation_zero_reset": corr,
        "reset_zero_ratio":       float(delta_reset.mean() / (delta_zero.mean() + 1e-12)),
    }
