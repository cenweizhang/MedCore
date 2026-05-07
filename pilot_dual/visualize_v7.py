"""
Standalone visualization for v7 results.

Usage:
    python -m pilot_dual.visualize_v7 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --data_root   asserts/CVC-ColonDB \
        --scores_dir  results/pilot_cascade_v7 \
        --head_sp     0.5 \
        --mlp_sp      0.7 \
        --output      results/pilot_cascade_v7/viz_v7.png
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
from pilot_phase1.metrics import compute_all_metrics
from pilot_dual.pruning import (
    compute_block_sensitivity,
    allocate_nonuniform_head_sparsity, allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform, generate_neuron_mask_nonuniform,
    apply_head_mask_to_model, apply_mlp_mask_to_model, remove_hooks,
)


@torch.no_grad()
def infer(model, batch, device):
    model.eval()
    images = batch["image"].to(device)
    bboxes = batch["bbox"].to(device).float()
    if bboxes.dim() == 2:
        bboxes = bboxes[:, None, :]
    emb = model.image_encoder(images)
    sp, dp = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
    low, _ = model.mask_decoder(
        image_embeddings=emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp, dense_prompt_embeddings=dp,
        multimask_output=False,
    )
    pred = F.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)
    return (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--data_root",   default="asserts/CVC-ColonDB")
    p.add_argument("--scores_dir",  default="results/pilot_cascade_v7")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--head_sp",     type=float, default=0.5)
    p.add_argument("--mlp_sp",      type=float, default=0.0,
                   help="0 = head-only (no MLP pruning)")
    p.add_argument("--n_samples",   type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=21)
    p.add_argument("--n_cal",       type=int,   default=128)
    p.add_argument("--protected_blocks", type=int, nargs="*", default=[10, 11])
    p.add_argument("--output",      default=None)
    args = p.parse_args()

    device   = torch.device(args.device)
    out_path = args.output or os.path.join(
        args.scores_dir, f"viz_v7_h{args.head_sp:.0%}_m{args.mlp_sp:.0%}.png")

    # ── 1. Load scores ──────────────────────────────────────────────────
    sc             = np.load(os.path.join(args.scores_dir, "scores.npz"))
    block_sens     = sc["block_sensitivity"]
    alpha_per_block = sc["alpha_per_block"]               # (12,)
    dist_beta      = float(sc["dist_beta"]) if "dist_beta" in sc.files else 0.0
    mlp_dim        = 3072

    # Reconstruct head scores with adaptive α + distribution-aware
    dz_r = sc["dz_head_per_subset"]                       # (R, 144)
    dr_r = sc["dr_head_per_subset"]
    alpha_vec = np.repeat(alpha_per_block, 12).astype(np.float32)
    Q_r  = alpha_vec * dz_r + (1.0 - alpha_vec) * dr_r   # (R, 144)
    head_scores = (Q_r.mean(axis=0) + dist_beta * Q_r.var(axis=0)).astype(np.float32)

    # MLP scores (needed even if mlp_sp=0 — just won't be used)
    if args.mlp_sp > 0:
        dzm_r = sc["dz_mlp_per_subset"]
        drm_r = sc["dr_mlp_per_subset"]
        alpha_mlp = np.repeat(alpha_per_block, mlp_dim).astype(np.float32)
        Qm_r = alpha_mlp * dzm_r + (1.0 - alpha_mlp) * drm_r
        mlp_scores = (Qm_r.mean(axis=0) + dist_beta * Qm_r.var(axis=0)).astype(np.float32)

    # ── 2. Build masks ──────────────────────────────────────────────────
    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sens, args.head_sp, num_heads=12, min_keep=1,
        protected_blocks=args.protected_blocks)
    head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
    print(f"Head mask: {int(head_mask.sum())}/144 kept  "
          f"(actual sp {1-head_mask.mean():.1%})")

    neuron_mask = None
    if args.mlp_sp > 0:
        per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
            block_sens, args.mlp_sp, mlp_dim=mlp_dim, min_frac=0.05,
            protected_blocks=args.protected_blocks)
        neuron_mask = generate_neuron_mask_nonuniform(
            mlp_scores, per_block_sp_mlp, mlp_dim=mlp_dim)
        print(f"MLP mask: {int(neuron_mask.sum())}/{12*mlp_dim} kept  "
              f"(actual sp {1-neuron_mask.mean():.1%})")

    # ── 3. Load model ───────────────────────────────────────────────────
    print("Loading MedSAM ...")
    model_b = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device).eval()
    model_p = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device).eval()
    h_hooks = apply_head_mask_to_model(model_p, head_mask)
    m_hooks = apply_mlp_mask_to_model(model_p, neuron_mask) if neuron_mask is not None else []

    # ── 4. Data ─────────────────────────────────────────────────────────
    _, test_loader, _, test_dataset, _ = build_dataloaders(
        args.data_root, n_calibration=args.n_cal,
        batch_size=1, seed=args.seed, num_workers=2)
    n_test = len(test_dataset)
    rng    = random.Random(args.seed)
    vis_idx = sorted(rng.sample(range(n_test), min(args.n_samples, n_test)))
    print(f"Test size={n_test}  visualizing indices={vis_idx}")

    # ── 5. Inference ────────────────────────────────────────────────────
    from skimage.transform import resize as sk_resize
    def r256(a, order=0):
        return sk_resize(a, (256, 256), order=order,
                         preserve_range=True, anti_aliasing=False)

    vis_set = set(vis_idx)
    samples = {}
    print("Running inference ...")
    for i, batch in enumerate(test_loader):
        if i not in vis_set: continue
        gt = batch["mask_1024"][0].numpy().astype(np.uint8)
        img = batch["image"][0].permute(1, 2, 0).cpu().numpy()
        pb = infer(model_b, batch, device)
        pp = infer(model_p, batch, device)
        samples[i] = {
            "name": batch["name"][0],
            "img":  (r256(img, order=1) * 255).clip(0, 255).astype(np.uint8),
            "gt":   r256(gt).astype(np.uint8),
            "base": r256(pb).astype(np.uint8),
            "pruned": r256(pp).astype(np.uint8),
            "mb": compute_all_metrics(pb, gt),
            "mp": compute_all_metrics(pp, gt),
        }
        if len(samples) == len(vis_idx): break

    remove_hooks(h_hooks + m_hooks)

    # ── 6. Figure ───────────────────────────────────────────────────────
    rows = [samples[i] for i in vis_idx if i in samples]
    n    = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(16, n * 3.2), squeeze=False)

    col_titles = ["Image", "GT", "Baseline",
                  f"Pruned (h={args.head_sp:.0%}, m={args.mlp_sp:.0%})",
                  "Diff (red=lost, blue=gained)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9, fontweight="bold")

    for row, s in enumerate(rows):
        img, gt, base, pruned = s["img"], s["gt"], s["base"], s["pruned"]
        mb, mp = s["mb"], s["mp"]

        def overlay(im, mask, rgb, a=0.45):
            out = im.astype(np.float32).copy()
            for c, v in enumerate(rgb):
                out[:,:,c] = np.where(mask>0, out[:,:,c]*(1-a)+v*a, out[:,:,c])
            return out.clip(0,255).astype(np.uint8)

        axes[row,0].imshow(img)
        axes[row,0].set_ylabel(s["name"][:20], fontsize=7,
                                rotation=0, labelpad=60, va="center")
        axes[row,1].imshow(overlay(img, gt, (0,220,60)))
        axes[row,2].imshow(overlay(img, base, (50,100,255)))
        axes[row,2].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        axes[row,2].set_title(
            f"Dice={mb['dice']:.3f}  BF1={mb['boundary_f1']:.3f}", fontsize=7, pad=2)
        axes[row,3].imshow(overlay(img, pruned, (255,50,50)))
        axes[row,3].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        dd = mp['dice']-mb['dice']; db = mp['boundary_f1']-mb['boundary_f1']
        axes[row,3].set_title(
            f"Dice={mp['dice']:.3f}({dd:+.3f})  BF1={mp['boundary_f1']:.3f}({db:+.3f})",
            fontsize=7, pad=2)

        diff = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        diff[(base==1)&(pruned==1)] = [180,180,180]
        diff[(base==1)&(pruned==0)] = [255, 80, 80]
        diff[(base==0)&(pruned==1)] = [ 80, 80,255]
        axes[row,4].imshow(diff)
        axes[row,4].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)

        for c in range(5): axes[row,c].axis("off")

    legend = [
        mpatches.Patch(color=(180/255,180/255,180/255), label="Both agree"),
        mpatches.Patch(color=(255/255, 80/255, 80/255), label="Lost by pruning"),
        mpatches.Patch(color=( 80/255, 80/255,255/255), label="Gained by pruning"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"v7 pruning — head_sp={args.head_sp:.0%}  mlp_sp={args.mlp_sp:.0%}  "
        f"(no recovery, masks only)",
        fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
