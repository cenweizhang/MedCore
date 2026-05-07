"""
Bake Stage-1 (h70_m95, protected=[10,11]) pruning masks into MedSAM weights.

Regenerates the exact masks from the Stage-1 Phase-B cache, then zeros the
corresponding weight columns so Stage-2 Fisher is computed on the already-
pruned model (not the original).

Usage:
    conda run -n medsam python -m pilot_dual.bake_stage1_masks \
        --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
        --cache_dir    results/pilot_cascade_v8_sweep/_phase_b_cache \
        --out          results/stage2/stage1_pruned_h70m95.pth
"""

import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry
from pilot_dual.run_cascade_v8 import combine_scores_dist_adaptive
from pilot_dual.pruning import (
    allocate_nonuniform_head_sparsity,
    allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform,
    generate_neuron_mask_nonuniform,
)


def bake_head_mask(model, head_mask, num_blocks=12, num_heads=12):
    """Zero the columns of attn.proj.weight that correspond to pruned heads."""
    for l in range(num_blocks):
        attn = model.image_encoder.blocks[l].attn
        head_dim = attn.qkv.weight.shape[1] // num_heads
        block_mask = head_mask[l * num_heads: (l + 1) * num_heads]      # (12,)
        channel_mask = np.repeat(block_mask, head_dim).astype(np.float32) # (768,)
        ch = torch.tensor(channel_mask, dtype=attn.proj.weight.dtype,
                          device=attn.proj.weight.device)
        with torch.no_grad():
            attn.proj.weight.mul_(ch.unsqueeze(0))   # zero dead columns


def bake_mlp_mask(model, neuron_mask, num_blocks=12):
    """Zero the columns of mlp.lin2.weight that correspond to pruned neurons."""
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]
    for l in range(num_blocks):
        mlp = model.image_encoder.blocks[l].mlp
        block_mask = neuron_mask[l * mlp_dim: (l + 1) * mlp_dim].astype(np.float32)
        ch = torch.tensor(block_mask, dtype=mlp.lin2.weight.dtype,
                          device=mlp.lin2.weight.device)
        with torch.no_grad():
            mlp.lin2.weight.mul_(ch.unsqueeze(0))    # zero dead columns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--cache_dir",   default="results/pilot_cascade_v8_sweep/_phase_b_cache")
    p.add_argument("--out",         default="results/stage2/stage1_pruned_h70m95.pth")
    p.add_argument("--head_sp",     type=float, default=0.7)
    p.add_argument("--mlp_sp",      type=float, default=0.95)
    p.add_argument("--protected_blocks", nargs="+", type=int, default=[10, 11])
    p.add_argument("--min_keep_head",    type=int, default=1)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load scores from Stage-1 cache
    sc = np.load(os.path.join(args.cache_dir, "scores.npz"), allow_pickle=True)
    block_sensitivity = sc["block_sensitivity"]
    alpha_per_block   = sc["alpha_per_block"]
    pi_r              = sc["pi_r"]
    dist_beta         = float(sc["dist_beta"])

    dz_h_per = sc["dz_head_per_subset"]   # (R, 144)
    dr_h_per = sc["dr_head_per_subset"]   # (R, 144)
    dz_m_per = sc["dz_mlp_per_subset"]   # (R, 36864)
    dr_m_per = sc["dr_mlp_per_subset"]   # (R, 36864)

    R = len(pi_r)

    # Reproduce Stage-1 head scores
    head_scores = combine_scores_dist_adaptive(
        [dz_h_per[i] for i in range(R)],
        [dr_h_per[i] for i in range(R)],
        alpha_per_block, items_per_block=12,
        pi=pi_r, beta=dist_beta)

    # Stage-1 head allocation (h=0.7, protected=[10,11])
    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sensitivity, args.head_sp,
        num_heads=12, min_keep=args.min_keep_head,
        protected_blocks=args.protected_blocks)
    head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)

    # Reproduce Stage-1 MLP scores
    mlp_dim = 3072
    n_blocks = 12
    mlp_scores = combine_scores_dist_adaptive(
        [dz_m_per[i] for i in range(R)],
        [dr_m_per[i] for i in range(R)],
        alpha_per_block, items_per_block=mlp_dim,
        pi=pi_r, beta=dist_beta)

    # Stage-1 MLP allocation (m=0.95, protected=[10,11])
    per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
        block_sensitivity, args.mlp_sp,
        mlp_dim=mlp_dim,
        protected_blocks=args.protected_blocks)
    neuron_mask = generate_neuron_mask_nonuniform(mlp_scores, per_block_sp_mlp)

    kept_h = int(head_mask.sum())
    kept_m = int(neuron_mask.sum())
    print(f"Stage-1 masks: heads kept={kept_h}/144 ({kept_h/144*100:.1f}%)  "
          f"neurons kept={kept_m}/36864 ({kept_m/36864*100:.1f}%)")

    # Load model and bake masks into weights
    print("Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model.eval()

    print("Baking Stage-1 masks into weights ...")
    bake_head_mask(model, head_mask)
    bake_mlp_mask(model, neuron_mask)

    # Save as new checkpoint (only image_encoder state is modified)
    # Raw state dict format so sam_model_registry can load it directly
    torch.save(model.state_dict(), args.out)

    # Save masks alongside for reference / Stage-2 bookkeeping
    meta_path = args.out.replace(".pth", "_masks.npz")
    np.savez(meta_path,
             head_mask=head_mask,
             neuron_mask=neuron_mask,
             stage1_head_sp=args.head_sp,
             stage1_mlp_sp=args.mlp_sp)
    print(f"Saved → {args.out}")
    print(f"Masks → {meta_path}")


if __name__ == "__main__":
    main()
