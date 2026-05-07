"""
Visualize v8 pruned model: 5 samples × 3 datasets.
Columns: Image | GT | Baseline | Pruned | Diff

Usage:
    python -m pilot_dual.visualize_v8 \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --scores_dir  results/pilot_cascade_v8_sweep \
        --data_roots  asserts/kvasir-seg/Kvasir-SEG asserts/CVC-ColonDB asserts/CVC-ClinicDB \
        --head_sp 0.6 --mlp_sp 0.95 \
        --out results/pilot_cascade_v8_sweep/viz_v8_06095.png
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize as sk_resize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry
from pilot_phase1.dataset import PolypDataset
from pilot_phase1.metrics import compute_all_metrics
from pilot_dual.pruning import (
    allocate_nonuniform_head_sparsity, allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform, generate_neuron_mask_nonuniform,
    apply_head_mask_to_model, apply_mlp_mask_to_model, remove_hooks,
)

DATASETS = ["Kvasir", "ColonDB", "ClinicDB"]
DS_COLORS = [(0.2, 0.5, 0.9), (0.9, 0.2, 0.2), (0.1, 0.7, 0.3)]


@torch.no_grad()
def infer_single(model, batch, device):
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


def r256(arr, order=0):
    return sk_resize(arr, (256, 256), order=order, preserve_range=True, anti_aliasing=False)


def overlay(img, mask, rgb, alpha=0.45):
    out = img.astype(np.float32).copy()
    for c, v in enumerate(rgb):
        out[:, :, c] = np.where(mask > 0, out[:, :, c] * (1 - alpha) + v * 255 * alpha, out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--scores_dir",  default="results/pilot_cascade_v8_sweep")
    p.add_argument("--data_roots",  nargs=3,
                   default=["asserts/Kvasir-SEG", "asserts/CVC-ColonDB", "asserts/CVC-ClinicDB"])
    p.add_argument("--head_sp",     type=float, default=0.3)
    p.add_argument("--mlp_sp",      type=float, default=0.3)
    p.add_argument("--n_per_ds",    type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--protected_blocks", type=int, nargs="*", default=[10, 11])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = args.out or os.path.join(
        args.scores_dir,
        f"viz_v8_h{args.head_sp:.0%}_m{args.mlp_sp:.0%}.png")

    # ── 1. Build masks from scores ──────────────────────────────────────
    sc = np.load(os.path.join(args.scores_dir, "scores.npz"))
    block_sens      = sc["block_sensitivity"]
    alpha_per_block = sc["alpha_per_block"]
    dist_beta       = float(sc["dist_beta"])
    mlp_dim         = 3072

    alpha_vec  = np.repeat(alpha_per_block, 12).astype(np.float32)
    Q_head_r   = alpha_vec * sc["dz_head_per_subset"] + (1 - alpha_vec) * sc["dr_head_per_subset"]
    head_scores = Q_head_r.mean(0) + dist_beta * Q_head_r.var(0)

    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sens, args.head_sp, num_heads=12, min_keep=1,
        protected_blocks=args.protected_blocks)
    head_mask = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
    print(f"Head mask : {int(head_mask.sum())}/144 kept  (actual sp {1-head_mask.mean():.1%})")

    neuron_mask = None
    if args.mlp_sp > 0:
        alpha_mlp  = np.repeat(alpha_per_block, mlp_dim).astype(np.float32)
        Q_mlp_r    = alpha_mlp * sc["dz_mlp_per_subset"] + (1 - alpha_mlp) * sc["dr_mlp_per_subset"]
        mlp_scores = Q_mlp_r.mean(0) + dist_beta * Q_mlp_r.var(0)
        per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
            block_sens, args.mlp_sp, mlp_dim=mlp_dim, min_frac=0.05,
            protected_blocks=args.protected_blocks)
        neuron_mask = generate_neuron_mask_nonuniform(mlp_scores, per_block_sp_mlp, mlp_dim=mlp_dim)
        print(f"MLP mask  : {int(neuron_mask.sum())}/{12*mlp_dim} kept  (actual sp {1-neuron_mask.mean():.1%})")

    # ── 2. Load models ──────────────────────────────────────────────────
    print("Loading MedSAM ...")
    model_b = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device).eval()
    model_p = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device).eval()
    h_hooks = apply_head_mask_to_model(model_p, head_mask)
    m_hooks = apply_mlp_mask_to_model(model_p, neuron_mask) if neuron_mask is not None else []

    # ── 3. Sample images per dataset ────────────────────────────────────
    rng = random.Random(args.seed)
    all_samples = []   # list of (ds_name, ds_color, sample_dict)

    for ds_name, ds_root, ds_color in zip(DATASETS, args.data_roots, DS_COLORS):
        dataset = PolypDataset(ds_root, bbox_shift=0)
        n = len(dataset)
        indices = sorted(rng.sample(range(n), min(args.n_per_ds, n)))
        print(f"{ds_name} ({n} total): sampling {indices}")
        for idx in indices:
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                     for k, v in dataset[idx].items()}
            gt   = batch["mask_1024"][0].numpy().astype(np.uint8) if isinstance(batch["mask_1024"], torch.Tensor) else batch["mask_1024"].numpy().astype(np.uint8)
            img  = batch["image"][0].permute(1, 2, 0).cpu().numpy()
            pred_b = infer_single(model_b, batch, device)
            pred_p = infer_single(model_p, batch, device)
            all_samples.append((ds_name, ds_color, {
                "name":   batch["name"] if isinstance(batch["name"], str) else batch["name"][0],
                "img":    (r256(img, order=1) * 255).clip(0, 255).astype(np.uint8),
                "gt":     r256(gt).astype(np.uint8),
                "base":   r256(pred_b).astype(np.uint8),
                "pruned": r256(pred_p).astype(np.uint8),
                "mb":     compute_all_metrics(pred_b, gt),
                "mp":     compute_all_metrics(pred_p, gt),
            }))

    remove_hooks(h_hooks + m_hooks)

    # ── 4. Plot ─────────────────────────────────────────────────────────
    n_rows = len(all_samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, n_rows * 3.0), squeeze=False)
    col_titles = [
        "Image + bbox region",
        "Ground Truth",
        "Baseline",
        f"Pruned  h={args.head_sp:.0%} m={args.mlp_sp:.0%}",
        "Diff  (red=lost  blue=gained)",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8, fontweight="bold")

    prev_ds = None
    for row, (ds_name, ds_color, s) in enumerate(all_samples):
        img, gt, base, pruned = s["img"], s["gt"], s["base"], s["pruned"]
        mb, mp = s["mb"], s["mp"]

        # Dataset label on first row of each group
        if ds_name != prev_ds:
            axes[row, 0].set_ylabel(f"── {ds_name} ──", fontsize=9,
                                    fontweight="bold", color=ds_color,
                                    rotation=0, labelpad=72, va="center")
            prev_ds = ds_name
        else:
            axes[row, 0].set_ylabel(s["name"][:18], fontsize=6.5,
                                    rotation=0, labelpad=70, va="center")

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(overlay(img, gt, ds_color))
        axes[row, 2].imshow(overlay(img, base, (0.2, 0.4, 1.0)))
        axes[row, 2].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        axes[row, 2].set_title(f"Dice={mb['dice']:.3f}  BF1={mb['boundary_f1']:.3f}",
                               fontsize=7, pad=2)
        axes[row, 3].imshow(overlay(img, pruned, (1.0, 0.2, 0.2)))
        axes[row, 3].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)
        dd = mp["dice"] - mb["dice"]
        db = mp["boundary_f1"] - mb["boundary_f1"]
        axes[row, 3].set_title(
            f"Dice={mp['dice']:.3f}({dd:+.3f})  BF1={mp['boundary_f1']:.3f}({db:+.3f})",
            fontsize=7, pad=2)

        diff = np.zeros((256, 256, 3), dtype=np.uint8)
        diff[(base == 1) & (pruned == 1)] = [180, 180, 180]
        diff[(base == 1) & (pruned == 0)] = [255,  60,  60]
        diff[(base == 0) & (pruned == 1)] = [ 60,  60, 255]
        axes[row, 4].imshow(diff)
        axes[row, 4].contour(gt, levels=[0.5], colors=["lime"], linewidths=0.8)

        # Divider line between datasets
        if ds_name != (all_samples[row + 1][0] if row + 1 < n_rows else ds_name):
            for c in range(5):
                axes[row, c].spines["bottom"].set_linewidth(2.5)
                axes[row, c].spines["bottom"].set_color("black")

        for c in range(5):
            axes[row, c].axis("off")

    legend_handles = [
        mpatches.Patch(color=(180/255, 180/255, 180/255), label="Both agree"),
        mpatches.Patch(color=(255/255,  60/255,  60/255), label="Lost by pruning"),
        mpatches.Patch(color=( 60/255,  60/255, 255/255), label="Gained by pruning"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"v8 pruning — head_sp={args.head_sp:.0%}  mlp_sp={args.mlp_sp:.0%}  "
        f"| 5 samples × 3 datasets  (Kvasir / ColonDB / ClinicDB)",
        fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.025, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
