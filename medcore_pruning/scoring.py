"""Dual-intervention group scores and distribution-aware dataset aggregation."""

import numpy as np
import torch


def load_sam_encoder_params(sam_checkpoint_path):
    """Return pretrained SAM image-encoder parameters as CPU tensors."""
    from segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
    return {name: parameter.detach().float().cpu().clone()
            for name, parameter in model.image_encoder.named_parameters()}


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
        fisher     : dict from boundary-aware Fisher estimation.
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
                if has_qkv_bias:
                    if sam_qkv_b is None:
                        raise ValueError(f"SAM checkpoint is missing {qkv_b_key}.")
                    sam_parts += [
                        sam_qkv_b[q_rows],
                        sam_qkv_b[k_rows],
                        sam_qkv_b[v_rows],
                    ]
                theta_g_sam = torch.cat(sam_parts)
            else:
                raise ValueError(f"SAM checkpoint lacks attention parameters for block {l}.")

            # ---- Compute Eq. 25, 26 ----
            delta_g = theta_g - theta_g_sam

            dz = 0.5 * (F_g * theta_g  ** 2).sum().item()  # Eq. 25
            dr = 0.5 * (F_g * delta_g  ** 2).sum().item()  # Eq. 26

            delta_zero[head_id]  = float(dz)
            delta_reset[head_id] = float(dr)

    return delta_zero, delta_reset


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
        fisher     : dict from boundary-aware Fisher estimation.
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
                if sam_lb is None:
                    raise ValueError(f"SAM checkpoint is missing {l1b_key}.")
                col = sam_lb.float().unsqueeze(1)
                sam_g = torch.cat([sam_g, col], dim=1)
        else:
            raise ValueError(f"SAM checkpoint lacks MLP parameters for block {l}.")

        delta_g = theta_g - sam_g

        dz = 0.5 * (F_g * theta_g ** 2).sum(dim=1).numpy()   # (mlp_dim,)
        dr = 0.5 * (F_g * delta_g ** 2).sum(dim=1).numpy()

        sl = slice(l * mlp_dim, (l + 1) * mlp_dim)
        delta_zero[sl]  = dz
        delta_reset[sl] = dr

    return delta_zero, delta_reset


def combine_scores_dist_adaptive(dz_list, dr_list, alpha_per_block,
                                  items_per_block, pi=None, beta=0.3):
    """Mix zero/reset scores, then add unweighted inter-dataset variance."""
    R = len(dz_list)
    if not R or len(dr_list) != R:
        raise ValueError("Provide matching, nonempty per-dataset score lists.")
    K = len(dz_list[0])

    alpha_vec = np.repeat(alpha_per_block, items_per_block).astype(np.float32)
    if len(alpha_vec) != K:
        raise ValueError("Block mixtures do not match the number of group scores.")

    Q_r = np.stack(
        [alpha_vec * dz + (1.0 - alpha_vec) * dr
         for dz, dr in zip(dz_list, dr_list)],
        axis=0,
    )   # (R, K)

    if pi is None:
        pi = np.full(R, 1.0 / R, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (R,) or not np.isfinite(pi).all() or (pi < 0).any() or pi.sum() <= 0:
        raise ValueError("Provide one finite, nonnegative weight per dataset with positive sum.")
    pi = pi / pi.sum()

    Q_mean = (pi[:, None] * Q_r).sum(axis=0)
    if beta == 0.0 or R == 1:
        return Q_mean.astype(np.float32)
    return (Q_mean + beta * Q_r.var(axis=0)).astype(np.float32)
