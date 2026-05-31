# -*- coding: utf-8 -*-
"""
Head-level structured pruning for MedSAM Image Encoder.
Version 2.1: Parallelized.

V2 changes (vs v1):
  V2.1: compute_head_gradient_projections_fast — output-space Taylor proxy
        c_{i,l,h} = <δ_{i,l,h}, a_{i,l,h}> via hooks on attn.proj.
  V2.2: score_heads_pointwise / score_heads_ewr — per-block distributions.
  V2.3: generate_head_mask_blockwise — exhaustive C(H,k) search per block.
  V2.4: Relative epsilon ε_l = α × median_pairwise_dist.
  V2.5: diagnose_epsilon_sensitivity — Check 3 diagnostic.

V2.1 parallelism improvements:
  P-A: Batched Taylor proxy — removes inner per-sample loop; processes full
       batch in one forward+backward pass. Works for both global attention
       (pre shape: B,H,W,dim) and window attention (pre shape: B*nW,ws,ws,dim)
       by detecting B via pre.shape[0] > B.
       Use --batch_size 32 or higher to saturate GPU.

  P-B: Vectorized exhaustive search —
       - Pointwise: replaces 924 per-candidate calls with one numpy matmul
         (_eval_all_candidates_pointwise).
       - EWR: replaces 924 serial CPU Sinkhorn calls with one batched GPU
         Sinkhorn kernel (_sinkhorn_distance_batch_gpu / _eval_all_candidates_ewr_gpu).

  Expected speedup (128 samples, batch_size=32, 50% sparsity, GPU A100-40G):
    Taylor proxy:   ~8-16× faster (4 batches vs 128 passes)
    Pointwise search: ~500× faster per block (matmul vs loop)
    EWR search:     ~100-200× faster per block (GPU Sinkhorn vs CPU loop)
"""

import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from tqdm import tqdm


# ==============================================================================
# Utility
# ==============================================================================

def dice_loss(pred, target):
    """Soft Dice loss (sigmoid, MEAN reduction)."""
    pred   = torch.sigmoid(pred)
    pred   = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter  = (pred * target).sum(dim=1)
    return (1.0 - (2.0 * inter + 1e-5) /
            (pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)).mean()


def _dice_loss_sum(pred, target):
    """
    Soft Dice loss with SUM reduction.

    Used in batched Taylor proxy so that:
        ∂ (Σ_b loss_b) / ∂ activation[b,...] = ∂ loss_b / ∂ activation[b,...]

    i.e. per-sample gradients are preserved (not averaged away by /B).
    """
    pred   = torch.sigmoid(pred)
    pred   = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)
    return per.sum()


# ==============================================================================
# V2.1 — Output-space Taylor proxy (BATCHED, P-A)
# ==============================================================================

def compute_head_gradient_projections_fast(
    model, dataloader, device,
    num_blocks=12, num_heads=12,
    freq_weights=None,
):
    """
    Compute per-sample output-space Taylor proxy c_{i,l,h}.

        c_{i,l,h} = <a_{i,l,h}, δ_{i,l,h}>

    where a_{i,l,h} = pre-proj activation slice for head h (captured via
    register_forward_hook on attn.proj) and δ_{i,l,h} = ∂ℓ/∂a_{i,l,h}
    (captured via register_full_backward_hook on attn.proj).

    P-A parallelism: processes the FULL BATCH in a single forward+backward.
    Loss uses SUM reduction so grad[b,...] = ∂loss_b/∂activation[b,...].

    Handles both attention variants:
      Global attn  → pre shape (B, H, W, dim)
      Window attn  → pre shape (B*nW, ws, ws, dim)
    Detected automatically: if pre.shape[0] == B → global, else window.

    Freq weighting (P1.1):
        per_sample_projections[i, l*H+h] = sqrt(ω_i) · c_{i,l,h}

    Returns:
        head_importance        : (total_heads,) — mean |c_{i,l,h}|
        per_sample_projections : (n_samples, total_heads) — freq-weighted c values
    """
    model.eval()

    n_samples   = len(dataloader.dataset)
    total_heads = num_blocks * num_heads

    if freq_weights is None:
        freq_weights = np.ones(n_samples, dtype=np.float32)
    assert len(freq_weights) == n_samples

    per_sample_projections = np.zeros((n_samples, total_heads), dtype=np.float32)
    head_dim = model.image_encoder.blocks[0].attn.qkv.weight.shape[1] // num_heads

    sample_idx = 0

    for batch in tqdm(dataloader, desc="Taylor proxy c_{i,l,h} [batched]"):
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device)
        bboxes = batch["bbox"]      # kept on CPU; moved inside loop
        B      = images.shape[0]

        if sample_idx >= n_samples:
            break

        # ------------------------------------------------------------------ #
        # Enable only attn param gradients (anchor computation graph)
        # ------------------------------------------------------------------ #
        for p in model.parameters():
            p.requires_grad = False
        for blk_idx in range(num_blocks):
            for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                p.requires_grad = True

        model.zero_grad()

        # ------------------------------------------------------------------ #
        # Register hooks once per batch
        # ------------------------------------------------------------------ #
        saved_pre  = {}   # blk_idx → (*, dim) pre-proj tensor (detached values)
        saved_grad = {}   # blk_idx → (*, dim) gradient at pre-proj

        def make_fwd_hook(idx):
            def hook(module, inp, out):
                saved_pre[idx] = inp[0].detach()
            return hook

        def make_bwd_hook(idx):
            def hook(module, grad_inp, grad_out):
                if grad_inp[0] is not None:
                    saved_grad[idx] = grad_inp[0].detach()
            return hook

        fwd_hooks, bwd_hooks = [], []
        for blk_idx in range(num_blocks):
            proj = model.image_encoder.blocks[blk_idx].attn.proj
            fwd_hooks.append(proj.register_forward_hook(make_fwd_hook(blk_idx)))
            bwd_hooks.append(proj.register_full_backward_hook(make_bwd_hook(blk_idx)))

        # ------------------------------------------------------------------ #
        # Single forward+backward for the ENTIRE BATCH (P-A)
        # ------------------------------------------------------------------ #
        image_embedding = model.image_encoder(images)

        with torch.no_grad():
            box_t = bboxes.to(device).float()
            if box_t.dim() == 2:
                box_t = box_t[:, None, :]
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=box_t, masks=None
            )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # SUM reduction → grad[b,...] = ∂loss_b/∂activation[b,...]
        loss = (
            F.binary_cross_entropy_with_logits(
                low_res_masks, masks.float(), reduction='sum'
            )
            + _dice_loss_sum(low_res_masks, masks.float())
        )
        loss.backward()

        for hk in fwd_hooks + bwd_hooks:
            hk.remove()

        # ------------------------------------------------------------------ #
        # Extract per-sample c_{i,l,h} from batch tensors
        # ------------------------------------------------------------------ #
        sqrt_omegas = np.sqrt(
            freq_weights[sample_idx: sample_idx + B]
        ).astype(np.float32)   # (B,)

        with torch.no_grad():
            for blk_idx in range(num_blocks):
                if blk_idx not in saved_pre or blk_idx not in saved_grad:
                    continue

                pre    = saved_pre[blk_idx].float()    # (B or B*nW, *, dim)
                grad_t = saved_grad[blk_idx].float()

                first_dim = pre.shape[0]

                for h in range(num_heads):
                    s = slice(h * head_dim, (h + 1) * head_dim)

                    # Element-wise product in head-slice, then flatten and sum
                    # per original-batch sample.
                    #   Global: (B, H, W, head_dim) → reshape (B, -1) → sum → (B,)
                    #   Window: (B*nW, ws, ws, head_dim) → reshape (B, -1) → sum → (B,)
                    # In both cases reshape(B, -1).sum(1) works because window_partition
                    # groups windows by sample: indices [0..nW-1]=sample0, etc.
                    prod = (pre[..., s] * grad_t[..., s])   # (first_dim, *, head_dim)
                    c_ilh = prod.reshape(B, -1).sum(dim=1)  # (B,)

                    per_sample_projections[
                        sample_idx: sample_idx + B,
                        blk_idx * num_heads + h,
                    ] = c_ilh.cpu().numpy() * sqrt_omegas

        model.zero_grad()
        for blk_idx in range(num_blocks):
            for p in model.image_encoder.blocks[blk_idx].attn.parameters():
                p.requires_grad = False

        sample_idx += B

    head_importance = np.abs(per_sample_projections).mean(axis=0)
    return head_importance, per_sample_projections


# ==============================================================================
# Baseline scoring (random, magnitude) — unchanged
# ==============================================================================

def score_heads_random(num_blocks=12, num_heads=12, seed=42):
    """Random importance scores."""
    return np.random.RandomState(seed).rand(num_blocks * num_heads)


def score_heads_magnitude(model, num_blocks=12, num_heads=12):
    """L2 norm of head qkv + proj weights as importance proxy."""
    scores = np.zeros(num_blocks * num_heads)
    for blk_idx in range(num_blocks):
        attn     = model.image_encoder.blocks[blk_idx].attn
        dim      = attn.qkv.weight.shape[1]
        head_dim = dim // num_heads
        with torch.no_grad():
            qkv_w  = attn.qkv.weight.data
            proj_w = attn.proj.weight.data
            for h in range(num_heads):
                head_id = blk_idx * num_heads + h
                q_s = slice(h * head_dim,           (h + 1) * head_dim)
                k_s = slice(dim + h * head_dim,     dim + (h + 1) * head_dim)
                v_s = slice(2 * dim + h * head_dim, 2 * dim + (h + 1) * head_dim)
                p_s = slice(h * head_dim,           (h + 1) * head_dim)
                norm_sq = sum(qkv_w[s, :].norm().item() ** 2 for s in [q_s, k_s, v_s])
                norm_sq += proj_w[:, p_s].norm().item() ** 2
                scores[head_id] = np.sqrt(norm_sq)
    return scores


def score_heads_pointwise(per_sample_projections):
    """Legacy global-ranking Pointwise: mean |c_{i,l,h}| per head."""
    return np.abs(per_sample_projections).mean(axis=0)


# ==============================================================================
# Sinkhorn — CPU single-pair (kept for reference / fallback)
# ==============================================================================

def _logsumexp(a, axis=None):
    a_max  = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return result.squeeze(axis=axis)


def _sinkhorn_distance(x, y, epsilon=0.05, n_iter=200):
    """Entropy-regularised W_{2,ε}² between two 1D empirical distributions (CPU)."""
    n, m  = len(x), len(y)
    C     = (x[:, None] - y[None, :]) ** 2
    log_M = -C / epsilon
    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))
    log_u = np.zeros(n)
    log_v = np.zeros(m)
    for _ in range(n_iter):
        log_u = log_a - _logsumexp(log_M + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_M + log_u[:, None], axis=0)
    log_pi = log_u[:, None] + log_M + log_v[None, :]
    pi     = np.exp(log_pi)
    return float((C * pi).sum())


# ==============================================================================
# GPU-batched Sinkhorn (P-B)
# ==============================================================================

def _sinkhorn_distance_batch_gpu(X, y, epsilon, n_iter=200, device='cuda'):
    """
    Batched W_{2,ε}² between K candidate distributions and one teacher (GPU).

    Args:
        X      : (K, n) numpy or torch — K candidate pruned distributions
        y      : (n,)  numpy or torch — teacher distribution
        epsilon: Sinkhorn regularisation
        n_iter : Sinkhorn iterations
        device : torch device string or object

    Returns:
        (K,) numpy array of W_{2,ε}² values

    Memory: (K=924, n=128) → cost matrix ~60 MB on GPU. Scales linearly in K and n².
    """
    dev = torch.device(device)

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=dev)
    else:
        X = X.to(device=dev, dtype=torch.float32)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=dev)
    else:
        y = y.to(device=dev, dtype=torch.float32)

    K, n = X.shape

    # Cost matrix: C[k, i, j] = (X[k,i] - y[j])²
    C = (X.unsqueeze(2) - y.unsqueeze(0).unsqueeze(0)) ** 2   # (K, n, n)

    log_M = -C / epsilon   # (K, n, n)
    log_a = torch.full((K, n), -np.log(n), device=dev, dtype=torch.float32)
    log_b = torch.full((K, n), -np.log(n), device=dev, dtype=torch.float32)
    log_u = torch.zeros(K, n, device=dev, dtype=torch.float32)
    log_v = torch.zeros(K, n, device=dev, dtype=torch.float32)

    for _ in range(n_iter):
        # log_u[k,i] = log_a[k,i] - logsumexp_j(log_M[k,i,j] + log_v[k,j])
        log_u = log_a - torch.logsumexp(log_M + log_v.unsqueeze(1), dim=2)
        # log_v[k,j] = log_b[k,j] - logsumexp_i(log_M[k,i,j] + log_u[k,i])
        log_v = log_b - torch.logsumexp(log_M + log_u.unsqueeze(2), dim=1)

    log_pi = log_u.unsqueeze(2) + log_M + log_v.unsqueeze(1)  # (K, n, n)
    pi     = torch.exp(log_pi)
    return (C * pi).sum(dim=(1, 2)).cpu().numpy()              # (K,)


# ==============================================================================
# V2.2 — Block-level evaluation (single-candidate, kept for reference)
# ==============================================================================

def _median_pairwise_dist(vals):
    """Median |T_i - T_j| over all i<j pairs (for per-block relative ε)."""
    n    = len(vals)
    diff = np.abs(vals[:, None] - vals[None, :])
    idx  = np.triu_indices(n, k=1)
    return float(np.median(diff[idx]))


def _eval_block_pointwise(c_block, mask_z):
    """Q_pw(z) = mean_i (T_{i,l}(z) - T_{i,l})²  [single candidate, reference]."""
    T_teacher = c_block.sum(axis=1)
    T_pruned  = (c_block * mask_z).sum(axis=1)
    return float(((T_pruned - T_teacher) ** 2).mean())


def _eval_block_ewr(c_block, mask_z, epsilon, n_iter=200):
    """Q_ewr(z) = W_{2,ε}²({T(z)}, {T})  [single candidate, CPU, reference]."""
    T_teacher = c_block.sum(axis=1)
    T_pruned  = (c_block * mask_z).sum(axis=1)
    return _sinkhorn_distance(T_pruned, T_teacher, epsilon=epsilon, n_iter=n_iter)


# ==============================================================================
# Vectorized block evaluators (P-B)
# ==============================================================================

def _eval_all_candidates_pointwise(c_block, candidates):
    """
    Q_pw for ALL K candidates in one numpy matmul (P-B).

    Args:
        c_block    : (n_samples, num_heads) float32
        candidates : list of K tuples (keep_idx)

    Returns:
        (K,) float32 — Q_pw values
    """
    n_samples, num_heads = c_block.shape
    T_teacher = c_block.sum(axis=1)         # (n_samples,)

    K = len(candidates)
    # Build (K, num_heads) mask matrix
    mask_mat = np.zeros((K, num_heads), dtype=np.float32)
    for k, keep_idx in enumerate(candidates):
        mask_mat[k, list(keep_idx)] = 1.0

    # T_pruned: (K, n_samples)  via  (K, H) @ (H, N)
    T_pruned = mask_mat @ c_block.T         # (K, n_samples)

    diff = T_pruned - T_teacher[None, :]    # (K, n_samples)
    return (diff ** 2).mean(axis=1)         # (K,)


def _eval_all_candidates_ewr_gpu(c_block, candidates, epsilon, n_iter, device):
    """
    Q_ewr for ALL K candidates via one batched GPU Sinkhorn call (P-B).

    Args:
        c_block    : (n_samples, num_heads) float32
        candidates : list of K tuples (keep_idx)
        epsilon    : per-block Sinkhorn regularisation
        n_iter     : Sinkhorn iterations
        device     : torch device

    Returns:
        (K,) float32 — W_{2,ε}² values
    """
    n_samples, num_heads = c_block.shape
    T_teacher = c_block.sum(axis=1).astype(np.float32)   # (n_samples,)

    K = len(candidates)
    mask_mat  = np.zeros((K, num_heads), dtype=np.float32)
    for k, keep_idx in enumerate(candidates):
        mask_mat[k, list(keep_idx)] = 1.0

    T_pruned = (mask_mat @ c_block.T).astype(np.float32)  # (K, n_samples)

    if epsilon < 1e-10:
        # Degenerate ε: fall back to MSE (same as pointwise)
        diff = T_pruned - T_teacher[None, :]
        return (diff ** 2).mean(axis=1)

    return _sinkhorn_distance_batch_gpu(T_pruned, T_teacher, epsilon,
                                        n_iter=n_iter, device=device)


# ==============================================================================
# V2.3 — Block-wise exhaustive search (parallelized, P-B)
# ==============================================================================

def generate_head_mask_blockwise(
    per_sample_proj,
    sparsity,
    method,
    epsilon=None,
    alpha=None,
    n_iter=200,
    num_blocks=12,
    num_heads=12,
    device='cpu',
    verbose=True,
):
    """
    Generate head mask via per-block exhaustive search over C(H, k) candidates.

    V2.1 parallelism (P-B):
      - Pointwise: single numpy matmul evaluates all K candidates at once.
      - EWR: single batched GPU Sinkhorn evaluates all K candidates at once.

    Candidate counts per block:
        30% sparsity → C(12,4) = 495
        50% sparsity → C(12,6) = 924
        70% sparsity → C(12,8) = 495

    Args:
        per_sample_proj : (n_samples, total_heads) — c_{i,l,h} values
        sparsity        : fraction of heads to remove per block
        method          : 'pointwise' or 'ewr'
        epsilon         : absolute epsilon (overrides alpha if set)
        alpha           : relative epsilon factor (default 0.5 for EWR)
        n_iter          : Sinkhorn iterations (EWR only)
        device          : torch device for GPU Sinkhorn ('cuda:0', 'cpu', etc.)
        verbose         : show tqdm progress bar

    Returns:
        head_mask : (total_heads,) float32 — 1=keep, 0=remove
    """
    assert method in ('pointwise', 'ewr'), f"Unknown method: {method!r}"
    assert 0.0 < sparsity < 1.0

    if method == 'ewr' and alpha is None and epsilon is None:
        alpha = 0.5

    k_remove   = int(num_heads * sparsity)
    n_keep     = num_heads - k_remove
    candidates = list(combinations(range(num_heads), n_keep))

    head_mask  = np.ones(num_blocks * num_heads, dtype=np.float32)

    block_iter = (
        tqdm(range(num_blocks), desc=f"Blockwise exhaustive [{method}] (P-B vectorized)")
        if verbose else range(num_blocks)
    )

    for blk_idx in block_iter:
        c_block   = per_sample_proj[
            :, blk_idx * num_heads: (blk_idx + 1) * num_heads
        ]                                       # (n_samples, num_heads)
        T_teacher = c_block.sum(axis=1)         # (n_samples,)

        if method == 'pointwise':
            # Vectorized numpy matmul — no GPU needed
            q_vals = _eval_all_candidates_pointwise(c_block, candidates)  # (K,)

        else:  # 'ewr'
            if epsilon is not None:
                eps_l = float(epsilon)
            else:
                med   = _median_pairwise_dist(T_teacher)
                eps_l = alpha * med
                if eps_l < 1e-10:
                    eps_l = max(alpha * float(np.std(T_teacher)), 1e-8)

            q_vals = _eval_all_candidates_ewr_gpu(
                c_block, candidates, eps_l, n_iter, device
            )  # (K,)

        best_k    = int(np.argmin(q_vals))
        block_mask = np.zeros(num_heads, dtype=np.float32)
        block_mask[list(candidates[best_k])] = 1.0
        head_mask[blk_idx * num_heads: (blk_idx + 1) * num_heads] = block_mask

    return head_mask


# ==============================================================================
# V2.5 — Epsilon sensitivity diagnostic (Check 3)
# ==============================================================================

def diagnose_epsilon_sensitivity(
    per_sample_proj,
    sparsity,
    alpha_values=(0.1, 0.5, 1.0),
    num_blocks=12,
    num_heads=12,
    n_iter=200,
    device='cpu',
):
    """
    Check 3: Compare block-level optimal masks across different α values.

    Returns:
        masks_by_alpha : dict {alpha: head_mask (total_heads,)}
        overlap_matrix : (len(alpha_values), len(alpha_values)) — fraction of
                         blocks where both alphas select identical masks
    """
    masks_by_alpha = {}
    for alpha in tqdm(alpha_values, desc="Check 3: ε sensitivity"):
        masks_by_alpha[alpha] = generate_head_mask_blockwise(
            per_sample_proj, sparsity, method='ewr', alpha=alpha,
            n_iter=n_iter, num_blocks=num_blocks, num_heads=num_heads,
            device=device, verbose=False,
        )

    n             = len(alpha_values)
    overlap_matrix = np.zeros((n, n))
    for i, a1 in enumerate(alpha_values):
        for j, a2 in enumerate(alpha_values):
            m1   = masks_by_alpha[a1].reshape(num_blocks, num_heads)
            m2   = masks_by_alpha[a2].reshape(num_blocks, num_heads)
            same = [np.array_equal(m1[b], m2[b]) for b in range(num_blocks)]
            overlap_matrix[i, j] = np.mean(same)

    return masks_by_alpha, overlap_matrix


# ==============================================================================
# Global-ranking mask generation (random / magnitude)
# ==============================================================================

def generate_head_mask(scores, sparsity, num_blocks=12, num_heads=12):
    """Remove the (sparsity × total_heads) lowest-scored heads globally."""
    total_heads = num_blocks * num_heads
    n_remove    = int(total_heads * sparsity)
    sorted_idx  = np.argsort(scores)
    mask        = np.ones(total_heads, dtype=np.float32)
    mask[sorted_idx[:n_remove]] = 0.0
    return mask


# ==============================================================================
# P0.1 — Head mask application (pre-hook on attn.proj, unchanged)
# ==============================================================================

def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Apply head mask via forward_pre_hook on attn.proj (BEFORE W_proj).

    Zeros out channel slice [h*head_dim : (h+1)*head_dim] for each pruned head h.
    """
    hooks = []
    for blk_idx in range(num_blocks):
        attn_module  = model.image_encoder.blocks[blk_idx].attn
        block_mask   = head_mask[blk_idx * num_heads: (blk_idx + 1) * num_heads]
        head_dim     = attn_module.qkv.weight.shape[1] // num_heads
        channel_mask = np.repeat(block_mask, head_dim).astype(np.float32)

        def make_pre_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hook = attn_module.proj.register_forward_pre_hook(
            make_pre_hook(channel_mask)
        )
        hooks.append(hook)
    return hooks


def remove_hooks(hooks):
    """Remove all registered hook handles."""
    for h in hooks:
        h.remove()
