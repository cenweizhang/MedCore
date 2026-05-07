"""
Visualize fine-tuned pruned model.  Columns: Image | GT | Baseline | FT-Pruned | Diff

Usage:
    python -m pilot_dual.visualize_finetune \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --ft_dirs results/finetune_pruned_sweep/h60_m50 \
                  results/finetune_pruned_sweep/h70_m70 \
        --data_roots asserts/kvasir-seg/Kvasir-SEG asserts/CVC-ColonDB asserts/CVC-ClinicDB \
        --n_per_ds 3 \
        --out results/finetune_pruned_sweep/viz_ft.png
"""

import os, sys, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry
from pilot_phase1.dataset import PolypDataset
from pilot_phase1.metrics import compute_all_metrics
from pilot_dual.pruning import (
    apply_head_mask_to_model, apply_mlp_mask_to_model, remove_hooks,
)
from skimage.transform import resize as sk_resize

DATASETS = ["Kvasir", "ColonDB", "ClinicDB"]


@torch.no_grad()
def infer(model, batch, device):
    model.eval()
    imgs   = batch["image"].to(device)
    bboxes = batch["bbox"].to(device).float()
    if bboxes.dim() == 2:
        bboxes = bboxes[:, None, :]
    emb = model.image_encoder(imgs)
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


def overlay(img, mask, rgb=(1.0, 0.2, 0.2), alpha=0.45):
    out = img.astype(np.float32).copy()
    for c, v in enumerate(rgb):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + v * 255 * alpha,
                                out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)


def load_ft_model(ft_dir, base_ckpt, device):
    """Load pruned + fine-tuned model from ft_dir/checkpoint_best.pth."""
    masks_path = os.path.join(ft_dir, "pruning_masks.npz")
    ckpt_path  = os.path.join(ft_dir, "checkpoint_best.pth")

    masks = np.load(masks_path, allow_pickle=True)
    head_mask   = masks["head_mask"]
    neuron_mask = masks["neuron_mask"] if masks["neuron_mask"].size > 0 else None

    model = sam_model_registry["vit_b"](checkpoint=base_ckpt).to(device).eval()
    h_hooks = apply_head_mask_to_model(model, head_mask)
    m_hooks = apply_mlp_mask_to_model(model, neuron_mask) if neuron_mask is not None else []

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    tag = os.path.basename(ft_dir)
    print(f"  Loaded {tag}  (best epoch={ck.get('epoch', '?')+1}  "
          f"val_loss={ck.get('best_val_loss', float('nan')):.4f})")
    return model, h_hooks + m_hooks, tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--ft_dirs",  nargs="+", required=True,
                   help="One or more fine-tuned config dirs (e.g. results/.../h50_m70)")
    p.add_argument("--data_roots", nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+", default=DATASETS)
    p.add_argument("--n_per_ds", type=int, default=3)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--out",      default=None)
    args = p.parse_args()

    device   = torch.device(args.device)
    rng      = random.Random(args.seed)
    n_models = len(args.ft_dirs)

    out_path = args.out or os.path.join(
        os.path.dirname(args.ft_dirs[0]),
        "viz_ft_" + "_".join(os.path.basename(d) for d in args.ft_dirs) + ".png")

    # ── 1. Load baseline ──────────────────────────────────────────────────
    print("Loading baseline MedSAM ...")
    baseline = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device).eval()

    # ── 2. Load each fine-tuned model ─────────────────────────────────────
    print("Loading fine-tuned models ...")
    ft_models, all_hooks, tags = [], [], []
    for d in args.ft_dirs:
        m, hooks, tag = load_ft_model(d, args.medsam_ckpt, device)
        ft_models.append(m)
        all_hooks.extend(hooks)
        tags.append(tag)

    # ── 3. Sample images ──────────────────────────────────────────────────
    samples = []   # list of (ds_name, img_np, gt_np, pred_base, [pred_ft, ...])
    for ds_name, ds_root in zip(args.dataset_names, args.data_roots):
        ds  = PolypDataset(ds_root, bbox_shift=0)
        idx = sorted(rng.sample(range(len(ds)), min(args.n_per_ds, len(ds))))
        print(f"  {ds_name}: {idx}")
        for i in idx:
            item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in ds[i].items()}
            gt   = item["mask_1024"][0].numpy().astype(np.uint8)
            img  = item["image"][0].permute(1, 2, 0).cpu().numpy()
            pb   = infer(baseline, item, device)
            pft  = [infer(m, item, device) for m in ft_models]
            samples.append((ds_name, img, gt, pb, pft))

    # ── 4. Plot ───────────────────────────────────────────────────────────
    # Columns: Image | GT | Baseline | FT_0 | FT_1 | ... | Diff (last FT vs base)
    n_rows = len(samples)
    n_cols = 3 + n_models + 1   # img, gt, base, ft×N, diff
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)

    col_titles = (["Image", "GT", "Baseline"] +
                  [f"FT {t}" for t in tags] +
                  [f"Diff  (FT[0] vs base)"])
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8, fontweight="bold")

    prev_ds = None
    for row, (ds_name, img, gt, pb, pft) in enumerate(samples):
        img8  = (r256(img, order=1) * 255).clip(0, 255).astype(np.uint8)
        gt8   = r256(gt).astype(np.uint8)
        pb8   = r256(pb).astype(np.uint8)
        pft8  = [r256(p).astype(np.uint8) for p in pft]

        mb   = compute_all_metrics(pb,    gt)
        mft0 = compute_all_metrics(pft[0], gt)

        if ds_name != prev_ds:
            axes[row, 0].set_ylabel(f"── {ds_name} ──", fontsize=9,
                                    fontweight="bold", rotation=0,
                                    labelpad=72, va="center")
            prev_ds = ds_name

        # Image
        axes[row, 0].imshow(img8)
        # GT
        axes[row, 1].imshow(overlay(img8, gt8, (0.1, 0.7, 0.3)))
        # Baseline
        axes[row, 2].imshow(overlay(img8, pb8, (0.2, 0.4, 1.0)))
        axes[row, 2].contour(gt8, levels=[0.5], colors=["lime"], linewidths=0.8)
        axes[row, 2].set_title(f"Dice={mb['dice']:.3f} BF1={mb['boundary_f1']:.3f}",
                               fontsize=7, pad=2)
        # FT models
        for k, (p8, m) in enumerate(zip(pft8, [compute_all_metrics(p, gt) for p in pft])):
            col = 3 + k
            axes[row, col].imshow(overlay(img8, p8, (1.0, 0.2, 0.2)))
            axes[row, col].contour(gt8, levels=[0.5], colors=["lime"], linewidths=0.8)
            dd = m["dice"]         - mb["dice"]
            db = m["boundary_f1"] - mb["boundary_f1"]
            axes[row, col].set_title(
                f"Dice={m['dice']:.3f}({dd:+.3f}) BF1={m['boundary_f1']:.3f}({db:+.3f})",
                fontsize=7, pad=2)
        # Diff (first FT vs baseline)
        diff = np.zeros((256, 256, 3), dtype=np.uint8)
        diff[(pb8 == 1) & (pft8[0] == 1)] = [180, 180, 180]
        diff[(pb8 == 1) & (pft8[0] == 0)] = [255,  60,  60]
        diff[(pb8 == 0) & (pft8[0] == 1)] = [ 60,  60, 255]
        axes[row, -1].imshow(diff)
        axes[row, -1].contour(gt8, levels=[0.5], colors=["lime"], linewidths=0.8)

        for c in range(n_cols):
            axes[row, c].axis("off")

    fig.suptitle("Post-pruning fine-tuning visualization  "
                 f"| red=FT  blue=baseline  diff: red=lost blue=gained",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    remove_hooks(all_hooks)


if __name__ == "__main__":
    main()
