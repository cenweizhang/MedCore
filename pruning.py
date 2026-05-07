# -*- coding: utf-8 -*-
"""
Mask generation, mask application, and model statistics for cascade pruning.

Masks are applied via PyTorch forward pre-hooks (no weight deletion).
The model remains structurally intact; zeroed heads/neurons contribute
no signal but linear layers are unchanged, enabling easy ablation.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------

def apply_head_mask_to_model(model, head_mask, num_blocks=12, num_heads=12):
    """
    Zero pruned-head channels before attn.proj via a pre-hook.

    Returns hooks : list of hook handles (pass to remove_hooks when done).
    """
    hooks = []
    for l in range(num_blocks):
        attn_module  = model.image_encoder.blocks[l].attn
        head_dim     = attn_module.qkv.weight.shape[1] // num_heads
        channel_mask = np.repeat(head_mask[l * num_heads: (l + 1) * num_heads],
                                 head_dim).astype(np.float32)

        def _make_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hooks.append(attn_module.proj.register_forward_pre_hook(
            _make_hook(channel_mask)))
    return hooks


def apply_mlp_mask_to_model(model, neuron_mask, num_blocks=12):
    """
    Zero pruned-neuron channels before mlp.lin2 via a pre-hook.

    Returns hooks : list of hook handles.
    """
    hooks   = []
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]
    for l in range(num_blocks):
        block_mask = neuron_mask[l * mlp_dim: (l + 1) * mlp_dim].astype(np.float32)

        def _make_hook(ch_mask):
            def hook_fn(module, inp):
                x = inp[0]
                m = torch.tensor(ch_mask, dtype=x.dtype, device=x.device)
                return (x * m,)
            return hook_fn

        hooks.append(model.image_encoder.blocks[l].mlp.lin2
                     .register_forward_pre_hook(_make_hook(block_mask)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Block-level sensitivity
# ---------------------------------------------------------------------------

def compute_block_sensitivity(fisher, num_blocks=12):
    """
    Aggregate diagonal Fisher into one scalar per block (Σ|F_i| per block).
    Higher = more sensitive = prune less aggressively.
    """
    sensitivity = np.zeros(num_blocks, dtype=np.float32)
    for key, val in fisher.items():
        for l in range(num_blocks):
            if key.startswith(f"blocks.{l}."):
                sensitivity[l] += val.sum().item()
                break
    return sensitivity


# ---------------------------------------------------------------------------
# Nonuniform sparsity allocation (inverse-sensitivity budget split)
# ---------------------------------------------------------------------------

def _budget_split(inv_weights, n_total, max_per_block):
    w = inv_weights / (inv_weights.sum() + 1e-12)
    n = np.round(w * n_total).astype(int)
    n = np.clip(n, 0, max_per_block)
    diff = n_total - n.sum()
    if diff > 0:
        headroom = max_per_block - n
        for idx in np.argsort(-headroom):
            if diff <= 0: break
            add = min(diff, int(headroom[idx]))
            n[idx] += add; diff -= add
    elif diff < 0:
        for idx in np.argsort(-n):
            if diff >= 0: break
            rem = min(-diff, int(n[idx]))
            n[idx] -= rem; diff += rem
    return n


def allocate_nonuniform_head_sparsity(block_sensitivity, target_sp,
                                      num_heads=12, min_keep=1,
                                      protected_blocks=None):
    """
    Per-block head sparsity inversely proportional to Fisher sensitivity.

    Protected blocks receive sparsity = 0.
    Returns per_block_sp : (num_blocks,) float32.
    """
    num_blocks    = len(block_sensitivity)
    protected_set = set(protected_blocks or [])
    active        = [l for l in range(num_blocks) if l not in protected_set]
    per_block_sp  = np.zeros(num_blocks, dtype=np.float32)
    if not active: return per_block_sp

    n_to_prune = int(round(target_sp * num_blocks * num_heads))
    max_per    = np.full(len(active), num_heads - min_keep, dtype=int)
    inv_sens   = 1.0 / (block_sensitivity[active] + 1e-12)
    alloc      = _budget_split(inv_sens, n_to_prune, max_per)
    for i, l in enumerate(active):
        per_block_sp[l] = alloc[i] / num_heads
    return per_block_sp


def allocate_nonuniform_neuron_sparsity(block_sensitivity, target_sp,
                                        mlp_dim=3072, min_frac=0.05,
                                        protected_blocks=None):
    """
    Per-block MLP neuron sparsity inversely proportional to Fisher sensitivity.

    Returns per_block_sp : (num_blocks,) float32.
    """
    num_blocks    = len(block_sensitivity)
    protected_set = set(protected_blocks or [])
    active        = [l for l in range(num_blocks) if l not in protected_set]
    per_block_sp  = np.zeros(num_blocks, dtype=np.float32)
    if not active: return per_block_sp

    min_keep   = int(round(min_frac * mlp_dim))
    n_to_prune = int(round(target_sp * num_blocks * mlp_dim))
    max_per    = mlp_dim - min_keep
    inv_sens   = 1.0 / (block_sensitivity[active] + 1e-12)
    alloc      = _budget_split(inv_sens, n_to_prune,
                               np.full(len(active), max_per, dtype=int))
    for i, l in enumerate(active):
        per_block_sp[l] = alloc[i] / mlp_dim
    return per_block_sp


# ---------------------------------------------------------------------------
# Nonuniform mask generation
# ---------------------------------------------------------------------------

def generate_head_mask_nonuniform(head_scores, per_block_sp, num_heads=12):
    """
    Head mask with per-block sparsity targets.
    Within each block, prune the lowest-scored heads first.

    Args:
        head_scores  : (num_blocks * num_heads,) lower = prune first.
        per_block_sp : (num_blocks,) from allocate_nonuniform_head_sparsity.

    Returns:
        mask : (num_blocks * num_heads,) float32, 1=keep 0=prune.
    """
    num_blocks = len(per_block_sp)
    mask = np.ones(num_blocks * num_heads, dtype=np.float32)
    for l in range(num_blocks):
        n_prune = min(int(round(per_block_sp[l] * num_heads)), num_heads - 1)
        if n_prune <= 0: continue
        base = l * num_heads
        prune_idx = np.argsort(head_scores[base: base + num_heads])[:n_prune]
        mask[base + prune_idx] = 0.0
    return mask


def generate_neuron_mask_nonuniform(q_mlp, per_block_sp, mlp_dim=3072):
    """
    MLP neuron mask with per-block sparsity targets.

    Args:
        q_mlp        : (num_blocks * mlp_dim,) lower = prune first.
        per_block_sp : (num_blocks,) from allocate_nonuniform_neuron_sparsity.

    Returns:
        mask : (num_blocks * mlp_dim,) float32, 1=keep 0=prune.
    """
    num_blocks = len(per_block_sp)
    mask = np.ones(num_blocks * mlp_dim, dtype=np.float32)
    for l in range(num_blocks):
        n_prune = min(int(round(per_block_sp[l] * mlp_dim)), mlp_dim - 1)
        if n_prune <= 0: continue
        base = l * mlp_dim
        prune_idx = np.argsort(q_mlp[base: base + mlp_dim])[:n_prune]
        mask[base + prune_idx] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Cascade model statistics
# ---------------------------------------------------------------------------

def compute_cascade_stats(model, head_mask, neuron_mask,
                          num_blocks=12, num_heads=12,
                          img_size=1024, patch_size=16):
    """
    Combined parameter count and FLOPs for a cascade-pruned model.

    Returns dict with: n_params_*, param_reduction_pct,
                       flops_*_G, flop_reduction_pct,
                       n_heads_kept, n_neurons_kept, etc.
    """
    seq_len            = (img_size // patch_size) ** 2   # 4096
    global_attn_blocks = {2, 5, 8, 11}
    window_size        = 14
    seq_window         = window_size * window_size
    n_windows          = seq_len / seq_window

    embed    = model.image_encoder.blocks[0].attn.qkv.weight.shape[1]
    head_dim = embed // num_heads
    mlp_dim  = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    n_params_total = sum(p.numel() for p in model.image_encoder.parameters())

    # Attention params / FLOPs per head
    attn_params = np.zeros(num_blocks * num_heads, dtype=np.int64)
    attn_flops  = np.zeros(num_blocks * num_heads, dtype=np.float64)
    for l in range(num_blocks):
        attn     = model.image_encoder.blocks[l].attn
        per_head = 3 * head_dim * embed + embed * head_dim
        if attn.qkv.bias is not None: per_head += 3 * head_dim
        qkv_f  = 6 * seq_len * embed * head_dim
        proj_f = 2 * seq_len * head_dim * embed
        attn_f = (4 * seq_len * seq_len * head_dim if l in global_attn_blocks
                  else 4 * n_windows * seq_window * seq_window * head_dim)
        for h in range(num_heads):
            attn_params[l * num_heads + h] = per_head
            attn_flops[l * num_heads + h]  = qkv_f + attn_f + proj_f

    head_pruned          = (head_mask == 0) if head_mask is not None else np.zeros(num_blocks * num_heads, bool)
    n_attn_params_pruned = int(np.sum(attn_params[head_pruned]))
    n_attn_flops_pruned  = float(np.sum(attn_flops[head_pruned]))

    # MLP params / FLOPs per neuron
    has_bias     = model.image_encoder.blocks[0].mlp.lin1.bias is not None
    per_neuron_p = 2 * embed + (1 if has_bias else 0)
    per_neuron_f = 4 * seq_len * embed

    neuron_pruned       = (neuron_mask == 0) if neuron_mask is not None else np.zeros(num_blocks * mlp_dim, bool)
    n_mlp_params_pruned = int(np.sum(neuron_pruned)) * per_neuron_p
    n_mlp_flops_pruned  = float(np.sum(neuron_pruned)) * per_neuron_f

    mlp_flops_total  = float(num_blocks * 2 * 2 * seq_len * embed * mlp_dim)
    attn_flops_total = float(np.sum(attn_flops))
    patch_emb_flops  = float(2 * seq_len * embed * (patch_size ** 2 * 3))
    total_flops      = attn_flops_total + mlp_flops_total + patch_emb_flops
    pruned_flops     = n_attn_flops_pruned + n_mlp_flops_pruned

    n_params_pruned    = n_attn_params_pruned + n_mlp_params_pruned
    n_params_remaining = n_params_total - n_params_pruned

    return {
        "n_params_total":           int(n_params_total),
        "n_params_remaining":       int(n_params_remaining),
        "n_params_pruned":          int(n_params_pruned),
        "param_reduction_pct":      round(100.0 * n_params_pruned   / max(n_params_total, 1), 2),
        "flops_total_G":            round(total_flops    / 1e9, 2),
        "flops_remaining_G":        round((total_flops - pruned_flops) / 1e9, 2),
        "flop_reduction_pct":       round(100.0 * pruned_flops / max(total_flops, 1), 2),
        "n_heads_kept":             int(np.sum(~head_pruned)),
        "n_heads_total":            num_blocks * num_heads,
        "n_neurons_kept":           int(np.sum(~neuron_pruned)),
        "n_neurons_total":          num_blocks * mlp_dim,
        "attn_param_reduction_pct": round(100.0 * n_attn_params_pruned / max(n_params_total, 1), 2),
        "mlp_param_reduction_pct":  round(100.0 * n_mlp_params_pruned  / max(n_params_total, 1), 2),
        "attn_flop_reduction_pct":  round(100.0 * n_attn_flops_pruned  / max(total_flops, 1), 2),
        "mlp_flop_reduction_pct":   round(100.0 * n_mlp_flops_pruned   / max(total_flops, 1), 2),
    }
