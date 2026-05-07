# -*- coding: utf-8 -*-
"""
Visualization of cascade pruning v5 results.

Loads the unpruned MedSAM model and re-applies saved pruning masks (without
recovery fine-tuning) to show the effect of pruning on test samples.

Usage
-----
    python -m pilot_dual.visualize_pruning \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --data_root   asserts/CVC-ColonDB \
        --scores_dir  results/pilot_cascade_v6 \
        --device      cuda:0 \
        --head_sp     0.5 \
        --mlp_sp      0.7 \
        --mlp_alpha   1.0 \
        --n_samples   8 \
        --output      results/pilot_cascade_v6/visualization.png


    python -m pilot_dual.visualize_pruning \
      --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
      --data_root   asserts/CVC-ColonDB \
      --scores_dir  results/pilot_cascade_v6 \
      --device      cuda:0 \
      --head_sp     0.5 \
      --mlp_sp      0.85 \
      --mlp_alpha   1.0 \
      --n_samples   8 \
      --output      results/pilot_cascade_v6/viz_v6_h50_m85.png
"""

import os
import sys
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.metrics import compute_all_metrics

from pilot_dual.scoring import combine_scores
from pilot_dual.pruning import (
    compute_block_sensitivity,
    allocate_nonuniform_head_sparsity,
    allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform,
    generate_neuron_mask_nonuniform,
    apply_head_mask_to_model,
    apply_mlp_mask_to_model,
    remove_hooks,
)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, batch, device):
    model.eval()
    images   = batch["image"].to(device)
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
    pred_bin = (torch.sigmoid(pred_1024) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    return pred_bin


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def overlay_mask(img_rgb, mask, color, alpha=0.45):
    """Return an image with a semi-transparent mask overlay."""
    out = img_rgb.astype(np.float32).copy()
    for c, val in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + val * alpha,
                                out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def make_figure(samples, head_sp, mlp_sp, mlp_alpha, out_path, viz_tag=""):
    """
    Each sample dict has keys:
        img, gt, pred_base, pred_pruned, name,
        metrics_base, metrics_pruned
    """
    n = len(samples)
    # 5 columns: Image | GT overlay | Baseline pred | Pruned pred | Diff
    ncols = 5
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.2, n * 3.2),
                              squeeze=False)

    col_titles = [
        "Input image",
        "Ground truth",
        f"Baseline (unpruned)",
        f"Pruned (head={head_sp:.0%}, mlp={mlp_sp:.0%}\nα={mlp_alpha})",
        "Diff (Base − Pruned)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    green = (0, 220, 60)
    red   = (255, 50, 50)
    blue  = (50, 100, 255)

    for i, s in enumerate(samples):
        img   = s["img"]           # (256,256,3) uint8
        gt    = s["gt"]            # (256,256) uint8 binary
        base  = s["pred_base"]     # (256,256) uint8 binary
        pruned = s["pred_pruned"]  # (256,256) uint8 binary
        mb    = s["metrics_base"]
        mp    = s["metrics_pruned"]

        # Col 0: raw image
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(s["name"], fontsize=7, rotation=0,
                              labelpad=60, va="center")

        # Col 1: GT overlay (green)
        axes[i, 1].imshow(overlay_mask(img, gt, green))

        # Col 2: baseline pred overlay (blue) + GT contour
        axes[i, 2].imshow(overlay_mask(img, base, blue))
        axes[i, 2].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        axes[i, 2].set_title(
            f"Dice={mb['dice']:.3f}  BF1={mb['boundary_f1']:.3f}\nHD95={mb['hd95']:.1f}",
            fontsize=7, pad=2)

        # Col 3: pruned pred overlay (red) + GT contour
        axes[i, 3].imshow(overlay_mask(img, pruned, red))
        axes[i, 3].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        delta_dice = mp["dice"] - mb["dice"]
        delta_bf1  = mp["boundary_f1"] - mb["boundary_f1"]
        axes[i, 3].set_title(
            f"Dice={mp['dice']:.3f} ({delta_dice:+.3f})\n"
            f"BF1={mp['boundary_f1']:.3f} ({delta_bf1:+.3f})",
            fontsize=7, pad=2)

        # Col 4: difference map
        tp  = ((base == 1) & (pruned == 1)).astype(np.uint8)
        fn  = ((base == 1) & (pruned == 0)).astype(np.uint8)  # lost by pruning
        fp  = ((base == 0) & (pruned == 1)).astype(np.uint8)  # gained by pruning
        diff_rgb = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        diff_rgb[tp == 1] = [180, 180, 180]   # grey: both agree
        diff_rgb[fn == 1] = [255,  80,  80]   # red : lost pixels
        diff_rgb[fp == 1] = [80,  80, 255]    # blue: gained pixels
        axes[i, 4].imshow(diff_rgb)
        axes[i, 4].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)

    # Legend for diff column
    legend_patches = [
        mpatches.Patch(color=(180/255, 180/255, 180/255), label="Both agree"),
        mpatches.Patch(color=(255/255,  80/255,  80/255), label="Lost (base only)"),
        mpatches.Patch(color=( 80/255,  80/255, 255/255), label="Gained (pruned only)"),
        mpatches.Patch(color=(  0/255, 220/255,  60/255), label="GT boundary"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Cascade pruning v5 — head_sp={head_sp:.0%}  mlp_sp={mlp_sp:.0%}  "
        f"mlp_α={mlp_alpha}  (no recovery, masks only){viz_tag}",
        fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize cascade v5 pruning effect on test samples"
    )
    parser.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--data_root",   default="asserts/CVC-ColonDB")
    parser.add_argument("--scores_dir",  default="results/pilot_cascade_v5",
                        help="Directory containing scores.npz and cascade_results_v5.json")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--n_cal",       type=int,   default=128,
                        help="Must match the value used when running run_cascade_v5")

    # Which pruning config to visualize
    parser.add_argument("--head_sp",   type=float, default=0.5,
                        help="Head sparsity target (e.g. 0.5)")
    parser.add_argument("--mlp_sp",    type=float, default=0.7,
                        help="MLP sparsity target (e.g. 0.7); 0 = head-only")
    parser.add_argument("--mlp_alpha", type=float, default=0.0,
                        help="MLP scoring alpha (0=reset_only, 1=zero_only)")
    parser.add_argument("--phase1_alpha", type=float, default=1.0)
    parser.add_argument("--tau",          type=float, default=0.0)
    parser.add_argument("--protected_blocks", type=int, nargs="+", default=[10, 11])
    parser.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    parser.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    # Sampling
    parser.add_argument("--n_samples",  type=int, default=6,
                        help="Number of test samples to visualize")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None,
                        help="Specific test-set indices to visualize (overrides --n_samples)")

    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: scores_dir/visualization.png)")
    args = parser.parse_args()

    device = torch.device(args.device)
    scores_path = os.path.join(args.scores_dir, "scores.npz")
    json_path   = os.path.join(args.scores_dir, "cascade_results_v5.json")
    out_path    = args.output or os.path.join(
        args.scores_dir,
        f"viz_h{args.head_sp:.0%}_m{args.mlp_sp:.0%}_a{args.mlp_alpha}.png"
    )

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("[1] Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # 2. Load saved scores
    # ------------------------------------------------------------------
    print("[2] Loading scores ...")
    sc = np.load(scores_path, allow_pickle=True)
    dz_head         = sc["delta_zero_head"]
    dr_head         = sc["delta_reset_head"]
    dz_mlp          = sc["delta_zero_mlp"]
    dr_mlp          = sc["delta_reset_mlp"]
    block_sensitivity = sc["block_sensitivity"]
    print(f"  block_sensitivity shape: {block_sensitivity.shape}")

    # v6 metadata (optional; present only if scores.npz comes from run_cascade_v6)
    n_subsets_saved = int(sc["n_dist_subsets"]) if "n_dist_subsets" in sc.files else 1
    dist_beta_saved = float(sc["dist_beta"])    if "dist_beta"      in sc.files else 0.0

    # If v6 additionally saved per-subset score stacks (R, K), use the TRUE
    # distribution-aware formula. Otherwise fall back to subset-mean (beta=0).
    has_per_subset = all(k in sc.files for k in
        ("dz_head_per_subset", "dr_head_per_subset",
         "dz_mlp_per_subset",  "dr_mlp_per_subset"))
    if has_per_subset:
        print(f"  [v6 exact mode] per-subset scores found  "
              f"(R={n_subsets_saved}, beta={dist_beta_saved})")
    elif n_subsets_saved > 1:
        print(f"  [v6 mean-only]  per-subset scores NOT saved in npz — "
              f"variance term (beta={dist_beta_saved}) cannot be reconstructed; "
              f"using subset-mean approximation (effective beta=0)")
    else:
        print(f"  [v5 / single-subset] standard mean scoring")

    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    # ------------------------------------------------------------------
    # 3. Build head mask
    # ------------------------------------------------------------------
    print("[3] Building head mask ...")
    if has_per_subset:
        dz_r = sc["dz_head_per_subset"]                     # (R, 144)
        dr_r = sc["dr_head_per_subset"]                     # (R, 144)
        q_r  = args.phase1_alpha * dz_r + (1 - args.phase1_alpha) * dr_r
        head_scores = q_r.mean(axis=0) + dist_beta_saved * q_r.var(axis=0)
        head_scores = head_scores.astype(np.float32)
    else:
        head_scores = combine_scores(dz_head, dr_head, alpha=args.phase1_alpha, tau=args.tau)
    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sensitivity, args.head_sp,
        num_heads=12, min_keep=args.nonuniform_min_keep_head,
        protected_blocks=args.protected_blocks,
    )
    head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
    n_heads_kept = int(head_mask.sum())
    print(f"  Head mask: {n_heads_kept}/144 heads kept  "
          f"(actual sparsity {1-n_heads_kept/144:.1%})")

    # ------------------------------------------------------------------
    # 4. Build MLP mask (optional)
    # ------------------------------------------------------------------
    neuron_mask = None
    if args.mlp_sp > 0:
        print("[4] Building MLP mask ...")
        if has_per_subset:
            dzm_r = sc["dz_mlp_per_subset"]                 # (R, 12*mlp_dim)
            drm_r = sc["dr_mlp_per_subset"]
            qm_r  = args.mlp_alpha * dzm_r + (1 - args.mlp_alpha) * drm_r
            q_mlp = qm_r.mean(axis=0) + dist_beta_saved * qm_r.var(axis=0)
            q_mlp = q_mlp.astype(np.float32)
        else:
            q_mlp = combine_scores(dz_mlp, dr_mlp, alpha=args.mlp_alpha, tau=args.tau)
        per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
            block_sensitivity, args.mlp_sp,
            mlp_dim=mlp_dim,
            min_frac=args.nonuniform_min_frac_mlp,
            protected_blocks=args.protected_blocks,
        )
        neuron_mask = generate_neuron_mask_nonuniform(q_mlp, per_block_sp_mlp, mlp_dim=mlp_dim)
        n_neurons_kept = int(neuron_mask.sum())
        print(f"  Neuron mask: {n_neurons_kept}/{12*mlp_dim} neurons kept  "
              f"(actual sparsity {1-n_neurons_kept/(12*mlp_dim):.1%})")

    # ------------------------------------------------------------------
    # 5. Data loaders — use same split as run_cascade_v5
    # ------------------------------------------------------------------
    print("[5] Building data loaders ...")
    _, test_loader, _, test_dataset, _ = build_dataloaders(
        args.data_root,
        n_calibration=args.n_cal,
        batch_size=1,
        seed=args.seed,
        num_workers=2,
    )
    n_test = len(test_dataset)
    print(f"  Test set size: {n_test}")

    # Decide which indices to visualize
    if args.sample_indices is not None:
        vis_indices = args.sample_indices
    else:
        rng = random.Random(args.seed)
        vis_indices = sorted(rng.sample(range(n_test), min(args.n_samples, n_test)))
    print(f"  Visualizing test indices: {vis_indices}")

    # ------------------------------------------------------------------
    # 6. Collect samples with baseline inference
    # ------------------------------------------------------------------
    from skimage.transform import resize as sk_resize

    def resize256(arr, order=0):
        if arr.ndim == 2:
            return sk_resize(arr, (256, 256), order=order,
                             preserve_range=True, anti_aliasing=False)
        return sk_resize(arr, (256, 256, arr.shape[2]), order=order,
                         preserve_range=True, anti_aliasing=(order > 0))

    print("[6] Running baseline inference ...")
    vis_index_set = set(vis_indices)
    samples = {}

    for i, batch in enumerate(test_loader):
        if i not in vis_index_set:
            continue
        gt_bin  = batch["mask_1024"][0].numpy().astype(np.uint8)
        img_np  = batch["image"][0].permute(1, 2, 0).cpu().numpy()  # (1024,1024,3)

        pred_base = run_inference(model, batch, device)
        m_base    = compute_all_metrics(pred_base, gt_bin)

        samples[i] = {
            "name":       batch["name"][0],
            "img":        (resize256(img_np, order=1) * 255).astype(np.uint8),
            "gt":         resize256(gt_bin, order=0).astype(np.uint8),
            "pred_base":  resize256(pred_base, order=0).astype(np.uint8),
            "metrics_base": m_base,
        }

        if len(samples) == len(vis_indices):
            break

    # ------------------------------------------------------------------
    # 7. Apply pruning masks and re-run inference
    # ------------------------------------------------------------------
    print("[7] Applying masks and running pruned inference ...")
    head_hooks   = apply_head_mask_to_model(model, head_mask)
    neuron_hooks = apply_mlp_mask_to_model(model, neuron_mask) if neuron_mask is not None else []

    for i, batch in enumerate(test_loader):
        if i not in vis_index_set:
            continue
        gt_bin      = batch["mask_1024"][0].numpy().astype(np.uint8)
        pred_pruned = run_inference(model, batch, device)
        m_pruned    = compute_all_metrics(pred_pruned, gt_bin)

        samples[i]["pred_pruned"]    = resize256(pred_pruned, order=0).astype(np.uint8)
        samples[i]["metrics_pruned"] = m_pruned

        if all("pred_pruned" in s for s in samples.values()):
            break

    remove_hooks(head_hooks)
    remove_hooks(neuron_hooks)

    # ------------------------------------------------------------------
    # 8. Print summary table
    # ------------------------------------------------------------------
    print("\nSample-level comparison (baseline vs pruned):")
    print(f"{'idx':>4}  {'name':>20}  {'Dice_B':>7}  {'Dice_P':>7}  "
          f"{'ΔDice':>7}  {'BF1_B':>7}  {'BF1_P':>7}  {'ΔBF1':>7}")
    for idx in vis_indices:
        if idx not in samples:
            continue
        s  = samples[idx]
        mb = s["metrics_base"]
        mp = s["metrics_pruned"]
        print(f"{idx:>4}  {s['name']:>20}  "
              f"{mb['dice']:>7.4f}  {mp['dice']:>7.4f}  {mp['dice']-mb['dice']:>+7.4f}  "
              f"{mb['boundary_f1']:>7.4f}  {mp['boundary_f1']:>7.4f}  "
              f"{mp['boundary_f1']-mb['boundary_f1']:>+7.4f}")

    # ------------------------------------------------------------------
    # 9. Make figure
    # ------------------------------------------------------------------
    print("[9] Generating figure ...")
    ordered_samples = [samples[i] for i in vis_indices if i in samples]
    if has_per_subset:
        viz_tag = f"  | v6 exact (R={n_subsets_saved}, β={dist_beta_saved})"
    elif n_subsets_saved > 1:
        viz_tag = f"  | v6 mean-only approx (saved β={dist_beta_saved} not applied)"
    else:
        viz_tag = ""
    make_figure(ordered_samples, args.head_sp, args.mlp_sp, args.mlp_alpha, out_path, viz_tag)
    print("Done.")


if __name__ == "__main__":
    main()
