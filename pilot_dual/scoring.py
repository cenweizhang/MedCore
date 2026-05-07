# -*- coding: utf-8 -*-
"""
Diagonal Fisher computation and dual-intervention head scoring.

Implements the core equations from the paper:

  F_i = (1/N) sum_n (d ell / d theta_i)^2                    (Eq. 24)

  Delta_zero_g  = 0.5 * sum_{i in g} F_i * theta_i^2         (Eq. 25)
  Delta_reset_g = 0.5 * sum_{i in g} F_i * (theta_i-theta_i^S)^2  (Eq. 26)

  Q_g   = alpha * Delta_zero_g + (1 - alpha) * Delta_reset_g  (Eq. 27)
  P_g   = Q_g / (c_g + eps)^tau                               (Eq. 28)

Group definition (Appendix B, Eq. 32):
  g^att_{l,h} = { qkv.weight rows for Q/K/V of head h,
                  qkv.bias   rows for Q/K/V of head h,
                  proj.weight cols for head h }
  (proj.bias is shared across heads and excluded)

Scope: image encoder attention heads only (12 blocks x 12 heads = 144 groups).
       Decoder and MLP groups are left for a future phase.

Usage (from run_dual.py):
    fisher = compute_diagonal_fisher(medsam_model, cal_loader, device)
    sam_params = load_sam_encoder_params(sam_checkpoint_path, device)
    delta_zero, delta_reset = compute_head_scores(medsam_model, sam_params, fisher)
    scores = combine_scores(delta_zero, delta_reset, alpha=0.5)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal loss helpers
# ---------------------------------------------------------------------------

def _dice_loss_sum(pred_logits, target):
    """Soft Dice loss with SUM reduction (preserves per-sample gradients)."""
    pred = torch.sigmoid(pred_logits)
    pred   = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5
    )
    return per.sum()


# ---------------------------------------------------------------------------
# Step 1: Diagonal Fisher estimation
# ---------------------------------------------------------------------------

def compute_diagonal_fisher(model, dataloader, device, num_blocks=12):
    """
    Estimate the diagonal Fisher for all image-encoder parameters.

    For batch_size=1 (recommended), each forward-backward pass gives the
    exact per-sample gradient, so:
        F_i = (1/N) sum_n (d ell_n / d theta_i)^2

    For larger batch sizes this becomes an approximation because autograd
    returns the gradient of the mean loss (not the sum), so per-sample
    contributions are averaged before squaring.  Keeping batch_size=1
    avoids this bias.

    Args:
        model      : MedSAM (Sam) model, on `device`.
        dataloader : calibration DataLoader (batch_size=1 recommended).
        device     : torch device.
        num_blocks : number of transformer blocks in the image encoder.

    Returns:
        fisher : dict {param_name (relative to image_encoder) -> tensor}
                 Same shape as the corresponding parameter.
    """
    model.eval()
    dev_type = device.type if isinstance(device, torch.device) else device.split(":")[0]

    # Freeze everything; unfreeze encoder only
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.image_encoder.parameters():
        p.requires_grad_(True)

    # Initialise accumulator on CPU to save GPU memory
    fisher = {
        n: torch.zeros_like(p, device="cpu")
        for n, p in model.image_encoder.named_parameters()
    }

    n_processed = 0
    nan_batches  = 0
    for batch in tqdm(dataloader, desc="Computing diagonal Fisher"):
        images = batch["image"].to(device)           # (B, 3, 1024, 1024)
        masks  = batch["mask_256"].to(device).float()  # (B, 1, 256, 256)
        bboxes = batch["bbox"].to(device).float()    # (B, 4)
        B      = images.shape[0]

        model.zero_grad()

        # Run in fp32 to avoid fp16 gradient overflow that produces NaN Fisher.
        image_emb = model.image_encoder(images)

        with torch.no_grad():
            box_t = bboxes
            if box_t.dim() == 2:
                box_t = box_t[:, None, :]            # (B, 1, 4)
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

        loss = (
            F.binary_cross_entropy_with_logits(
                low_res_masks, masks, reduction="sum"
            )
            + _dice_loss_sum(low_res_masks, masks)
        )
        loss.backward()

        # NaN guard: skip batches where any gradient is NaN (env/driver issue)
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
        print(f"  WARNING: {nan_batches} batches skipped due to NaN gradients.")

    # Normalise by number of samples -> (1/N) sum_n (...)^2
    for n in fisher:
        fisher[n] /= max(n_processed, 1)

    # Disable encoder gradients again
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)

    return fisher


# ---------------------------------------------------------------------------
# Step 2: Load SAM encoder parameters (theta^S)
# ---------------------------------------------------------------------------

def load_sam_encoder_params(sam_checkpoint_path, device="cpu"):
    """
    Load the original SAM ViT-B checkpoint and return the image-encoder
    parameter dict {param_name -> float tensor on CPU}.

    We load SAM independently so that the MedSAM model is not modified.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segment_anything import sam_model_registry

    print(f"  Loading SAM checkpoint from {sam_checkpoint_path} ...")
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
    sam_model.eval()

    sam_params = {
        n: p.data.clone().float().cpu()
        for n, p in sam_model.image_encoder.named_parameters()
    }
    del sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Loaded {len(sam_params)} SAM encoder parameter tensors.")
    return sam_params


# ---------------------------------------------------------------------------
# Step 3: Compute per-head dual-intervention scores
# ---------------------------------------------------------------------------

def compute_head_scores(model, sam_params, fisher, num_blocks=12, num_heads=12):
    """
    Compute Delta_zero_g and Delta_reset_g for every attention head group.

    Group g^att_{l,h} (Appendix B, Eq. 32) covers:
        qkv.weight  rows for Q, K, V of head h
        qkv.bias    rows for Q, K, V of head h  (if bias exists)
        proj.weight cols for head h

    Args:
        model      : MedSAM model (parameters = theta^M).
        sam_params : dict from load_sam_encoder_params() (theta^S).
        fisher     : dict from compute_diagonal_fisher().
        num_blocks : number of transformer blocks.
        num_heads  : number of attention heads per block.

    Returns:
        delta_zero  : np.ndarray shape (num_blocks * num_heads,), float32
        delta_reset : np.ndarray shape (num_blocks * num_heads,), float32
    """
    total_heads = num_blocks * num_heads
    delta_zero  = np.zeros(total_heads, dtype=np.float32)
    delta_reset = np.zeros(total_heads, dtype=np.float32)

    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        dim      = attn.qkv.weight.shape[1]   # embed_dim (768 for ViT-B)
        head_dim = dim // num_heads            # 64

        qkv_w_key  = f"blocks.{l}.attn.qkv.weight"
        qkv_b_key  = f"blocks.{l}.attn.qkv.bias"
        proj_w_key = f"blocks.{l}.attn.proj.weight"

        # Current MedSAM parameters (theta^M)
        theta_qkv_w  = attn.qkv.weight.data.float().cpu()   # (3*dim, dim)
        theta_proj_w = attn.proj.weight.data.float().cpu()  # (dim, dim)
        has_qkv_bias = (attn.qkv.bias is not None)
        if has_qkv_bias:
            theta_qkv_b = attn.qkv.bias.data.float().cpu()  # (3*dim,)

        # SAM parameters (theta^S)
        sam_qkv_w  = sam_params.get(qkv_w_key)
        sam_proj_w = sam_params.get(proj_w_key)
        if has_qkv_bias:
            sam_qkv_b = sam_params.get(qkv_b_key)

        # Diagonal Fisher values
        F_qkv_w  = fisher.get(qkv_w_key,  torch.zeros_like(theta_qkv_w))
        F_proj_w = fisher.get(proj_w_key, torch.zeros_like(theta_proj_w))
        if has_qkv_bias:
            F_qkv_b = fisher.get(qkv_b_key, torch.zeros_like(theta_qkv_b))

        for h in range(num_heads):
            head_id = l * num_heads + h

            # Index slices for this head
            q_rows = slice(h * head_dim,           (h + 1) * head_dim)
            k_rows = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
            v_rows = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
            p_cols = slice(h * head_dim,           (h + 1) * head_dim)

            # ---- Gather theta_g (current MedSAM params for this head) ----
            theta_parts = [
                theta_qkv_w[q_rows, :].reshape(-1),
                theta_qkv_w[k_rows, :].reshape(-1),
                theta_qkv_w[v_rows, :].reshape(-1),
                theta_proj_w[:, p_cols].reshape(-1),
            ]
            if has_qkv_bias:
                theta_parts += [
                    theta_qkv_b[q_rows],
                    theta_qkv_b[k_rows],
                    theta_qkv_b[v_rows],
                ]
            theta_g = torch.cat(theta_parts)

            # ---- Gather F_g (Fisher for this head) ----
            F_parts = [
                F_qkv_w[q_rows, :].reshape(-1),
                F_qkv_w[k_rows, :].reshape(-1),
                F_qkv_w[v_rows, :].reshape(-1),
                F_proj_w[:, p_cols].reshape(-1),
            ]
            if has_qkv_bias:
                F_parts += [
                    F_qkv_b[q_rows],
                    F_qkv_b[k_rows],
                    F_qkv_b[v_rows],
                ]
            F_g = torch.cat(F_parts)

            # ---- Gather theta_g^S (SAM params for this head) ----
            if sam_qkv_w is not None and sam_proj_w is not None:
                sam_parts = [
                    sam_qkv_w[q_rows, :].reshape(-1),
                    sam_qkv_w[k_rows, :].reshape(-1),
                    sam_qkv_w[v_rows, :].reshape(-1),
                    sam_proj_w[:, p_cols].reshape(-1),
                ]
                if has_qkv_bias and sam_qkv_b is not None:
                    sam_parts += [
                        sam_qkv_b[q_rows],
                        sam_qkv_b[k_rows],
                        sam_qkv_b[v_rows],
                    ]
                theta_g_sam = torch.cat(sam_parts)
            else:
                # Fallback: treat SAM as zero (should not happen in normal use)
                theta_g_sam = torch.zeros_like(theta_g)

            # ---- Compute Eq. 25, 26 ----
            delta_g = theta_g - theta_g_sam

            dz = 0.5 * (F_g * theta_g  ** 2).sum().item()  # Eq. 25
            dr = 0.5 * (F_g * delta_g  ** 2).sum().item()  # Eq. 26

            delta_zero[head_id]  = float(dz)
            delta_reset[head_id] = float(dr)

    return delta_zero, delta_reset


# ---------------------------------------------------------------------------
# Step 4: Combine into final priority score
# ---------------------------------------------------------------------------

def combine_scores(delta_zero, delta_reset, alpha=0.5, cost=None, tau=0.0, eps=1e-8):
    """
    Q_g = alpha * Delta_zero_g + (1 - alpha) * Delta_reset_g    (Eq. 27)
    P_g = Q_g / (c_g + eps)^tau                                  (Eq. 28)

    When tau=0 (default) all groups have the same cost and P_g = Q_g.

    Args:
        delta_zero  : (K,) float array
        delta_reset : (K,) float array
        alpha       : float in [0, 1]
        cost        : (K,) float array or None (uniform cost -> tau ignored)
        tau         : float >= 0
        eps         : numerical stability constant

    Returns:
        scores : (K,) float array (lower = prune first)
    """
    q = alpha * delta_zero + (1.0 - alpha) * delta_reset
    if cost is not None and tau > 0.0:
        q = q / (cost + eps) ** tau
    return q


# ---------------------------------------------------------------------------
# Diagnostic: correlation between zero and reset scores
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2b: Compute per-neuron MLP dual-intervention scores
# ---------------------------------------------------------------------------

def compute_mlp_neuron_scores(model, sam_params, fisher, num_blocks=12):
    """
    Compute Delta_zero_g and Delta_reset_g for every MLP neuron group.

    Group g^mlp_{l,n} (analogous to attention head group in Appendix B):
        lin1.weight[n, :]   embed_dim input weights for neuron n
        lin1.bias[n]        scalar bias (if exists)
        lin2.weight[:, n]   embed_dim output weights for neuron n
    lin2.bias is shared across neurons → excluded.

    Vectorised: all mlp_dim neurons in a block are scored in one shot.

    Args:
        model      : MedSAM model (theta^M).
        sam_params : dict from load_sam_encoder_params() (theta^S).
        fisher     : dict from compute_diagonal_fisher().
        num_blocks : number of transformer blocks.

    Returns:
        delta_zero  : np.ndarray  (num_blocks * mlp_dim,)  float32
        delta_reset : np.ndarray  (num_blocks * mlp_dim,)  float32
    """
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]   # 3072
    total   = num_blocks * mlp_dim

    delta_zero  = np.zeros(total, dtype=np.float32)
    delta_reset = np.zeros(total, dtype=np.float32)

    for l in range(num_blocks):
        mlp = model.image_encoder.blocks[l].mlp

        l1w_key = f"blocks.{l}.mlp.lin1.weight"
        l1b_key = f"blocks.{l}.mlp.lin1.bias"
        l2w_key = f"blocks.{l}.mlp.lin2.weight"

        theta_l1w = mlp.lin1.weight.data.float().cpu()   # (mlp_dim, embed_dim)
        theta_l2w = mlp.lin2.weight.data.float().cpu()   # (embed_dim, mlp_dim)
        has_bias  = mlp.lin1.bias is not None

        # Stack neuron features row-wise → (mlp_dim, 2*embed_dim [+1])
        theta_g = torch.cat([theta_l1w, theta_l2w.t()], dim=1)
        if has_bias:
            theta_g = torch.cat(
                [theta_g, mlp.lin1.bias.data.float().cpu().unsqueeze(1)], dim=1
            )

        # Fisher
        F_l1w = fisher.get(l1w_key, torch.zeros_like(theta_l1w))
        F_l2w = fisher.get(l2w_key, torch.zeros_like(theta_l2w))
        F_g   = torch.cat([F_l1w, F_l2w.t()], dim=1)
        if has_bias:
            F_lb = fisher.get(l1b_key, torch.zeros(mlp_dim))
            F_g  = torch.cat([F_g, F_lb.float().unsqueeze(1)], dim=1)

        # SAM params
        sam_l1w = sam_params.get(l1w_key)
        sam_l2w = sam_params.get(l2w_key)
        if sam_l1w is not None and sam_l2w is not None:
            sam_g = torch.cat([sam_l1w.float(), sam_l2w.t().float()], dim=1)
            if has_bias:
                sam_lb = sam_params.get(l1b_key)
                col   = (sam_lb.float().unsqueeze(1) if sam_lb is not None
                         else torch.zeros(mlp_dim, 1))
                sam_g = torch.cat([sam_g, col], dim=1)
        else:
            sam_g = torch.zeros_like(theta_g)

        delta_g = theta_g - sam_g

        dz = 0.5 * (F_g * theta_g ** 2).sum(dim=1).numpy()   # (mlp_dim,)
        dr = 0.5 * (F_g * delta_g ** 2).sum(dim=1).numpy()

        sl = slice(l * mlp_dim, (l + 1) * mlp_dim)
        delta_zero[sl]  = dz
        delta_reset[sl] = dr

    return delta_zero, delta_reset


# ---------------------------------------------------------------------------
# Diagnostic: correlation between zero and reset scores
# ---------------------------------------------------------------------------

def score_correlation(delta_zero, delta_reset):
    """
    Pearson correlation between Delta_zero and Delta_reset vectors.
    High correlation => reset term adds little new information.
    """
    z = delta_zero - delta_zero.mean()
    r = delta_reset - delta_reset.mean()
    denom = (np.linalg.norm(z) * np.linalg.norm(r)) + 1e-12
    return float(np.dot(z, r) / denom)


def score_summary(delta_zero, delta_reset, num_blocks=12, num_heads=12):
    """Return a dict of summary statistics for logging."""
    corr = score_correlation(delta_zero, delta_reset)
    return {
        "delta_zero_mean":   float(delta_zero.mean()),
        "delta_zero_std":    float(delta_zero.std()),
        "delta_reset_mean":  float(delta_reset.mean()),
        "delta_reset_std":   float(delta_reset.std()),
        "correlation_zero_reset": corr,
        "reset_zero_ratio":  float(delta_reset.mean() / (delta_zero.mean() + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Step 5: Cost vectors  (c_g from Eq. 11 / Eq. 28)
# ---------------------------------------------------------------------------

def compute_head_costs(model, num_blocks=12, num_heads=12):
    """
    Per-head parameter count used as the pruning cost c_g in Eq. (28).

    Group g^att_{l,h} (Appendix B, Eq. 32) covers:
        qkv.weight  rows for Q, K, V of head h  →  3 * head_dim * embed_dim
        proj.weight cols for head h              →      head_dim * embed_dim
        qkv.bias    rows for Q, K, V of head h  →  3 * head_dim  (if bias exists)

    Returns:
        costs : (num_blocks * num_heads,) float32 — parameter count per head.
    """
    costs = np.zeros(num_blocks * num_heads, dtype=np.float32)
    for l in range(num_blocks):
        attn      = model.image_encoder.blocks[l].attn
        embed_dim = attn.qkv.weight.shape[1]
        head_dim  = embed_dim // num_heads
        per_head  = 4 * head_dim * embed_dim          # 3×head_dim×embed (QKV) + 1×head_dim×embed (proj)
        if attn.qkv.bias is not None:
            per_head += 3 * head_dim
        costs[l * num_heads: (l + 1) * num_heads] = per_head
    return costs


def compute_neuron_costs(model, num_blocks=12):
    """
    Per-neuron parameter count used as the pruning cost c_g in Eq. (28).

    Group g^mlp_{l,n} covers:
        lin1.weight[n, :]  →  embed_dim
        lin2.weight[:, n]  →  embed_dim
        lin1.bias[n]       →  1  (if bias exists)

    Returns:
        costs : (num_blocks * mlp_dim,) float32 — parameter count per neuron.
    """
    mlp   = model.image_encoder.blocks[0].mlp
    mlp_dim   = mlp.lin1.weight.shape[0]
    embed_dim = mlp.lin1.weight.shape[1]
    per_neuron = 2 * embed_dim + (1 if mlp.lin1.bias is not None else 0)
    return np.full(num_blocks * mlp_dim, per_neuron, dtype=np.float32)
