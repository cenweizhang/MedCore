# -*- coding: utf-8 -*-
"""
Post-pruning full-dataset fine-tuning for MedSAM  (Phase D — single worker).

Reads pruning configuration DIRECTLY from Phase C results
(cascade_results_v8.json), regenerates the same head + MLP masks,
then fine-tunes the pruned model on the full polyp dataset pool.

Mask derivation
---------------
cascade_results_v8.json carries block_sensitivity, alpha_per_block, pi_r,
and the sparsity targets for each config.  scores.npz (colocated in the
SAME directory as the JSON) provides the per-group Q-scores needed to
reproduce the binary masks.  No Fisher re-computation is required.

Training
--------
image_encoder + mask_decoder are trainable; prompt_encoder is frozen
(identical to original MedSAM training in train_one_gpu.py).  Pruning
hooks remain active throughout training: zeroed activations produce
near-zero gradients for pruned parameters, so only the kept weights adapt.
Loss    : Dice + BCE at 256×256 decoder resolution.
Optimizer: AdamW + cosine LR decay.

Usage
-----
  # Select by sparsity values (closest match from Phase C):
  python -m pilot_dual.run_finetune_pruned \
      --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
      --phase_c_json results/pilot_cascade_v8_sweep/cascade_results_v8.json \
      --head_sparsity 0.7 --mlp_sparsity 0.7 \
      --output_dir results/finetune_pruned \
      --num_epochs 20 --batch_size 4 --lr 1e-4 --use_amp --device cuda:0

  # Auto-select the config with best Phase-C BF1:
  python -m pilot_dual.run_finetune_pruned \\
      --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \\
      --phase_c_json results/pilot_cascade_v8_sweep/cascade_results_v8.json \\
      --select_best \\
      --output_dir results/finetune_pruned_best --device cuda:0

  # Resume interrupted training:
  python -m pilot_dual.run_finetune_pruned \\
      --phase_c_json results/pilot_cascade_v8_sweep/cascade_results_v8.json \\
      --head_sparsity 0.5 --mlp_sparsity 0.5 \\
      --output_dir results/finetune_pruned \\
      --resume     results/finetune_pruned/h50_m50/checkpoint_latest.pth


# 补做最终
for entry in "0.30 0.80 0 h30_m80" "0.30 0.85 1 h30_m85" "0.30 0.90 2 h30_m90" "0.30 0.95 3 h30_m95" \
               "0.40 0.30 0 h40_m30" "0.50 0.70 1 h50_m70" "0.60 0.50 2 h60_m50" "0.70 0.70 3 h70_m70" \
               "0.70 0.85 0 h70_m85" "0.70 0.90 1 h70_m90" "0.70 0.95 2 h70_m95" "0.80 0.30 3 h80_m30"; do 
    read h m gpu tag <<< "$entry"
    CUDA_VISIBLE_DEVICES=$gpu python -m pilot_dual.run_finetune_pruned \
        --phase_c_json results/pilot_cascade_v8_sweep/cascade_results_v8.json \
        --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
        --output_dir   results/finetune_pruned_sweep \
        --head_sparsity $h --mlp_sparsity $m \
        --num_epochs 1 \
        --resume results/finetune_pruned_sweep/${tag}/checkpoint_best.pth \
        --device cuda:0 --use_amp \
        >> results/finetune_pruned_sweep/finetune_${tag}.log 2>&1 &
done
wait && echo "All evaluations done"
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from pilot_phase1.dataset import PolypDataset
from pilot_phase1.metrics import compute_all_metrics
from pilot_dual.pruning import (
    apply_head_mask_to_model,
    apply_mlp_mask_to_model,
    remove_hooks,
    compute_cascade_stats,
    allocate_nonuniform_head_sparsity,
    allocate_nonuniform_neuron_sparsity,
    generate_head_mask_nonuniform,
    generate_neuron_mask_nonuniform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


class _DiceLoss(nn.Module):
    """Inline replacement for monai.losses.DiceLoss(sigmoid=True, squared_pred=True)."""
    def forward(self, pred, target):
        p     = torch.sigmoid(pred)
        num   = (p * target).sum(dim=(-2, -1))
        denom = p.pow(2).sum(dim=(-2, -1)) + target.pow(2).sum(dim=(-2, -1)) + 1e-6
        return (1.0 - 2.0 * num / denom).mean()


def _combine_scores(dz_list, dr_list, alpha_per_block, items_per_block,
                    pi, beta=0.3):
    """Mirrors combine_scores_dist_adaptive used in run_cascade_v8."""
    R = len(dz_list)
    K = len(dz_list[0])
    alpha_vec = np.repeat(alpha_per_block, items_per_block).astype(np.float32)
    if len(alpha_vec) != K:
        alpha_vec = np.full(K, float(alpha_per_block.mean()), dtype=np.float32)
    Q_r = np.stack(
        [alpha_vec * dz + (1.0 - alpha_vec) * dr
         for dz, dr in zip(dz_list, dr_list)],
        axis=0,
    )
    pi = np.asarray(pi, dtype=np.float64) / np.sum(pi)
    Q_mean = (pi[:, None] * Q_r).sum(axis=0)
    if beta == 0.0 or R == 1:
        return Q_mean.astype(np.float32)
    return (Q_mean + beta * Q_r.var(axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Mask derivation: read Phase C JSON + colocated scores.npz
# ---------------------------------------------------------------------------

def select_config(cascade_results, head_sp, mlp_sp, select_best=False):
    """Pick one cascade config from Phase C results.

    select_best=True  → config with highest macro BF1 among all cascade entries.
    select_best=False → closest (head_sp_target, mlp_sp_target) to the given values.
    """
    rows = [r for r in cascade_results if r.get("phase") == "cascade"]
    if not rows:
        rows = cascade_results

    if select_best:
        chosen = max(rows, key=lambda r: r["macro"]["mean_boundary_f1"])
        print(f"  [select_best] h={chosen['head_sp_target']:.2f}  "
              f"m={chosen.get('mlp_sp_target', 0):.2f}  "
              f"Phase-C BF1={chosen['macro']['mean_boundary_f1']:.4f}")
    else:
        def _dist(r):
            return (abs(r.get("head_sp_target", 0) - head_sp) +
                    abs(r.get("mlp_sp_target",  0) - mlp_sp))
        chosen = min(rows, key=_dist)
        if _dist(chosen) > 0.01:
            print(f"  WARNING: no exact match for h={head_sp} m={mlp_sp}; "
                  f"using closest → h={chosen['head_sp_target']:.2f} "
                  f"m={chosen.get('mlp_sp_target', 0):.2f}")
    return chosen


def get_pruning_masks(args, model):
    """
    Load Phase C JSON + colocated scores.npz → regenerate pruning masks.

    Returns (head_mask, neuron_mask, head_sp, mlp_sp, chosen_cfg).
    neuron_mask is None when mlp_sp == 0.
    """
    mlp_dim = model.image_encoder.blocks[0].mlp.lin1.weight.shape[0]

    # ------------------------------------------------------------------
    # 1. Phase C JSON  →  meta parameters + config selection
    # ------------------------------------------------------------------
    print(f"  Phase C JSON: {args.phase_c_json}")
    with open(args.phase_c_json) as f:
        phase_c = json.load(f)

    block_sensitivity = np.array(phase_c["block_sensitivity"],
                                  dtype=np.float32)
    alpha_per_block   = np.array(phase_c["alpha_per_block"],
                                  dtype=np.float32)
    pi_r              = np.array(phase_c["pi_r"], dtype=np.float64)
    pi_r             /= pi_r.sum()
    dist_beta         = float(phase_c.get("config", {}).get("dist_beta", 0.3))

    chosen      = select_config(
        phase_c["cascade_results"],
        head_sp=args.head_sparsity,
        mlp_sp=args.mlp_sparsity,
        select_best=args.select_best,
    )
    head_sp_target = float(chosen["head_sp_target"])
    mlp_sp_target  = float(chosen.get("mlp_sp_target", 0.0))

    print(f"  Config: h={head_sp_target:.2f}  m={mlp_sp_target:.2f}  "
          f"Phase-C macro BF1={chosen['macro']['mean_boundary_f1']:.4f}")

    # ------------------------------------------------------------------
    # 2. scores.npz  →  per-group Q-scores (colocated with JSON)
    # ------------------------------------------------------------------
    scores_path = os.path.join(
        os.path.dirname(os.path.abspath(args.phase_c_json)), "scores.npz")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(
            f"scores.npz not found at expected path: {scores_path}\n"
            "It should be in the same directory as cascade_results_v8.json.")
    print(f"  scores.npz : {scores_path}")
    sc = np.load(scores_path)

    n_sub = sc["dz_head_per_subset"].shape[0]
    dz_h  = [sc["dz_head_per_subset"][i] for i in range(n_sub)]
    dr_h  = [sc["dr_head_per_subset"][i] for i in range(n_sub)]
    dz_m  = [sc["dz_mlp_per_subset"][i]  for i in range(n_sub)]
    dr_m  = [sc["dr_mlp_per_subset"][i]  for i in range(n_sub)]

    head_scores = _combine_scores(dz_h, dr_h, alpha_per_block,
                                   12,      pi_r, dist_beta)
    mlp_scores  = _combine_scores(dz_m, dr_m, alpha_per_block,
                                   mlp_dim, pi_r, dist_beta)

    # ------------------------------------------------------------------
    # 3. Generate masks  (identical nonuniform allocation as Phase C)
    # ------------------------------------------------------------------
    per_block_sp_head = allocate_nonuniform_head_sparsity(
        block_sensitivity, head_sp_target,
        num_heads=12,
        min_keep=args.nonuniform_min_keep_head,
        protected_blocks=args.protected_blocks,
    )
    head_mask     = generate_head_mask_nonuniform(head_scores, per_block_sp_head)
    actual_head_sp = 1.0 - float(head_mask.sum()) / 144.0
    print(f"  head_mask  : kept={int(head_mask.sum())}/144  "
          f"actual_sp={actual_head_sp*100:.1f}%")

    neuron_mask = None
    if mlp_sp_target > 0.0:
        per_block_sp_mlp = allocate_nonuniform_neuron_sparsity(
            block_sensitivity, mlp_sp_target,
            mlp_dim=mlp_dim,
            min_frac=args.nonuniform_min_frac_mlp,
            protected_blocks=args.protected_blocks,
        )
        neuron_mask    = generate_neuron_mask_nonuniform(
            mlp_scores, per_block_sp_mlp, mlp_dim=mlp_dim)
        actual_mlp_sp  = 1.0 - float(neuron_mask.sum()) / (12 * mlp_dim)
        print(f"  neuron_mask: kept={int(neuron_mask.sum())}/{12*mlp_dim}  "
              f"actual_sp={actual_mlp_sp*100:.1f}%")

    return head_mask, neuron_mask, head_sp_target, mlp_sp_target, chosen


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_train_val_loaders(data_roots, dataset_names, val_frac,
                             batch_size, seed, num_workers):
    """
    Per-dataset train/val split by val_frac.  Returns:
      train_loader : ConcatDataset, shuffled, bbox_shift=5 (augmentation)
      val_loaders  : list[DataLoader] per dataset, bbox_shift=0 (deterministic)
    """
    train_subsets, val_loaders = [], []
    print(f"\n  Building loaders (val_frac={val_frac}) ...")

    for root, name in zip(data_roots, dataset_names):
        ds_tr = PolypDataset(root, bbox_shift=5)
        ds_va = PolypDataset(root, bbox_shift=0)
        n     = len(ds_tr)
        n_val = max(1, int(round(n * val_frac)))
        gen   = torch.Generator().manual_seed(seed)
        idx   = torch.randperm(n, generator=gen).tolist()

        train_subsets.append(Subset(ds_tr, idx[:n - n_val]))
        val_loaders.append(DataLoader(
            Subset(ds_va, idx[n - n_val:]),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ))
        print(f"    [{name}] total={n}  train={n - n_val}  val={n_val}")

    combined_train = ConcatDataset(train_subsets)
    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    print(f"    [combined train] n={len(combined_train)}")
    return train_loader, val_loaders


# ---------------------------------------------------------------------------
# Evaluation (per-dataset + macro-avg, same metric set as v8)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loaders, dataset_names, device):
    model.eval()
    per_dataset = {}
    all_samples = []

    for loader, name in zip(val_loaders, dataset_names):
        samples = []
        for batch in tqdm(loader, desc=f"Eval[{name}]", leave=False):
            images   = batch["image"].to(device)            # (B, 3, 1024, 1024)
            masks_gt = batch["mask_1024"].numpy()            # (B, 1024, 1024)
            bboxes   = batch["bbox"].to(device).float()     # (B, 4)
            if bboxes.dim() == 2:
                bboxes = bboxes[:, None, :]

            emb = model.image_encoder(images)
            sp, dp = model.prompt_encoder(
                points=None, boxes=bboxes, masks=None)
            low, _ = model.mask_decoder(
                image_embeddings=emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
            )
            pred_1024 = F.interpolate(
                low, (1024, 1024), mode="bilinear", align_corners=False)
            pred_bin = (torch.sigmoid(pred_1024) > 0.5).cpu().numpy().astype(np.uint8)
            if pred_bin.ndim == 4:
                pred_bin = pred_bin[:, 0]

            for i in range(images.shape[0]):
                m = compute_all_metrics(
                    pred_bin[i], masks_gt[i].astype(np.uint8))
                samples.append(m)

        keys = ["dice", "iou", "boundary_f1", "hd95"]
        avg  = {f"mean_{k}": float(np.mean([s[k] for s in samples])) for k in keys}
        avg.update({f"std_{k}": float(np.std([s[k] for s in samples])) for k in keys})
        per_dataset[name] = avg
        all_samples.extend(samples)

    keys  = ["dice", "iou", "boundary_f1", "hd95"]
    macro = {f"mean_{k}": float(np.mean(
        [per_dataset[n][f"mean_{k}"] for n in dataset_names])) for k in keys}
    worst = {f"mean_{k}": float(min(
        per_dataset[n][f"mean_{k}"] for n in dataset_names)) for k in keys}
    macro["n_sets"] = len(dataset_names)
    return {"per_dataset": per_dataset, "macro": macro, "worst": worst}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, seg_loss_fn, ce_loss_fn,
                    device, scaler=None, grad_clip=1.0):
    model.train()
    for p in model.prompt_encoder.parameters():
        p.requires_grad_(False)

    total, steps = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)            # (B, 3, 1024, 1024)
        gt_256 = batch["mask_256"].float().to(device)  # (B, 1, 256, 256)
        bboxes = batch["bbox"].to(device).float()      # (B, 4)
        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]               # (B, 1, 4)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = model.image_encoder(images)
                with torch.no_grad():
                    sp, dp = model.prompt_encoder(
                        points=None, boxes=bboxes, masks=None)
                low, _ = model.mask_decoder(
                    image_embeddings=emb,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp,
                    dense_prompt_embeddings=dp,
                    multimask_output=False,
                )                                     # (B, 1, 256, 256) logits
                loss = seg_loss_fn(low, gt_256) + ce_loss_fn(low, gt_256)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            emb = model.image_encoder(images)
            with torch.no_grad():
                sp, dp = model.prompt_encoder(
                    points=None, boxes=bboxes, masks=None)
            low, _ = model.mask_decoder(
                image_embeddings=emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
            )
            loss = seg_loss_fn(low, gt_256) + ce_loss_fn(low, gt_256)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


@torch.no_grad()
def compute_val_loss(model, val_loaders, seg_loss_fn, ce_loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for loader in val_loaders:
        for batch in loader:
            images = batch["image"].to(device)
            gt_256 = batch["mask_256"].float().to(device)
            bboxes = batch["bbox"].to(device).float()
            if bboxes.dim() == 2:
                bboxes = bboxes[:, None, :]
            emb = model.image_encoder(images)
            sp, dp = model.prompt_encoder(
                points=None, boxes=bboxes, masks=None)
            low, _ = model.mask_decoder(
                image_embeddings=emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
            )
            total += (seg_loss_fn(low, gt_256) + ce_loss_fn(low, gt_256)).item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase D — post-pruning fine-tuning (single-config worker)")

    # --- Primary input (Phase C results) ---
    p.add_argument("--phase_c_json", required=True,
                   help="Path to cascade_results_v8.json produced by Phase C. "
                        "scores.npz must be in the same directory.")
    p.add_argument("--head_sparsity", type=float, default=0.5,
                   help="Select the Phase-C config closest to this head sparsity.")
    p.add_argument("--mlp_sparsity",  type=float, default=0.5,
                   help="Select the Phase-C config closest to this MLP sparsity.")
    p.add_argument("--select_best", action="store_true",
                   help="Ignore sparsity args; auto-select the best BF1 config.")

    # --- Model + output ---
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--output_dir",  default="results/finetune_pruned",
                   help="Parent output dir. Per-config results go to "
                        "<output_dir>/h{H}_m{M}/.")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint_latest.pth to resume training.")
    p.add_argument("--worker_idx", type=int, default=-1,
                   help="Worker index for parallel sweep (affects log naming only).")

    # --- Datasets ---
    p.add_argument("--data_roots", nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of each dataset held out for validation.")

    # --- Pruning mask parameters ---
    p.add_argument("--protected_blocks",         type=int, nargs="*", default=[10, 11])
    p.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    p.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    # --- Training hyper-parameters ---
    p.add_argument("--num_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--use_amp",      action="store_true")
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early stopping patience in epochs (0 = disabled).")
    p.add_argument("--eval_every",   type=int,   default=1)

    # --- Misc ---
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")

    # ------------------------------------------------------------------
    # 1. Load MedSAM
    # ------------------------------------------------------------------
    print("\n[1] Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model = model.to(device)
    for p in model.prompt_encoder.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2. Derive pruning masks from Phase C results
    # ------------------------------------------------------------------
    print("\n[2] Deriving pruning masks from Phase C ...")
    (head_mask, neuron_mask,
     head_sp, mlp_sp, chosen_cfg) = get_pruning_masks(args, model)

    # Per-config output subdirectory
    config_tag = f"h{int(head_sp * 100)}_m{int(mlp_sp * 100)}"
    out_dir    = os.path.join(args.output_dir, config_tag)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"POST-PRUNING FINE-TUNING — {config_tag}")
    print(f"  run_id       : {run_id}")
    print(f"  phase_c_json : {args.phase_c_json}")
    print(f"  config       : h={head_sp:.2f}  m={mlp_sp:.2f}")
    print(f"  Phase-C BF1  : {chosen_cfg['macro']['mean_boundary_f1']:.4f}")
    print(f"  datasets     : {args.dataset_names}  val_frac={args.val_frac}")
    print(f"  epochs       : {args.num_epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  AMP          : {args.use_amp}  patience={args.patience}")
    print(f"  output_dir   : {out_dir}")
    print("=" * 80)

    np.savez(
        os.path.join(out_dir, "pruning_masks.npz"),
        head_mask=head_mask,
        neuron_mask=neuron_mask if neuron_mask is not None else np.array([]),
    )

    # ------------------------------------------------------------------
    # 3. Apply masks via forward hooks
    # ------------------------------------------------------------------
    print("\n[3] Applying pruning masks ...")
    head_hooks = apply_head_mask_to_model(model, head_mask)
    mlp_hooks  = (apply_mlp_mask_to_model(model, neuron_mask)
                  if neuron_mask is not None else [])
    all_hooks  = head_hooks + mlp_hooks

    stats = compute_cascade_stats(model, head_mask, neuron_mask)
    print(f"  param_reduction={stats.get('param_reduction_pct', 0):.1f}%  "
          f"flop_reduction={stats.get('flop_reduction_pct', 0):.1f}%")

    # ------------------------------------------------------------------
    # 4. Build data loaders
    # ------------------------------------------------------------------
    print("\n[4] Building train / val loaders ...")
    train_loader, val_loaders = build_train_val_loaders(
        args.data_roots, args.dataset_names,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # 5. Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    train_params = (list(model.image_encoder.parameters()) +
                    list(model.mask_decoder.parameters()))
    optimizer = torch.optim.AdamW(
        train_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    # Same loss as original MedSAM training (train_one_gpu.py)
    seg_loss_fn = _DiceLoss()
    ce_loss_fn  = nn.BCEWithLogitsLoss(reduction="mean")
    scaler      = torch.cuda.amp.GradScaler() if args.use_amp else None

    start_epoch   = 0
    best_val_loss = float("inf")
    patience_ctr  = 0
    ckpt_best     = os.path.join(out_dir, "checkpoint_best.pth")
    ckpt_latest   = os.path.join(out_dir, "checkpoint_latest.pth")

    if args.resume and os.path.isfile(args.resume):
        print(f"\n  Resuming from {args.resume} ...")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch   = ck["epoch"] + 1
        best_val_loss = ck.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    history = {"train_loss": [], "val_loss": [], "eval_per_epoch": []}

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\n[6] Training {args.num_epochs} epochs ...")
    t_start = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        ep_train = train_one_epoch(
            model, train_loader, optimizer,
            seg_loss_fn, ce_loss_fn, device,
            scaler=scaler, grad_clip=args.grad_clip,
        )
        ep_val = compute_val_loss(
            model, val_loaders, seg_loss_fn, ce_loss_fn, device)
        scheduler.step()

        history["train_loss"].append(float(ep_train))
        history["val_loss"].append(float(ep_val))

        elapsed = (time.time() - t_start) / 60.0
        print(f"  Epoch {epoch+1:03d}/{args.num_epochs}  "
              f"train={ep_train:.4f}  val={ep_val:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}min]")

        # ---- latest checkpoint ----
        torch.save({
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "head_mask":     head_mask,
            "neuron_mask":   neuron_mask,
            "config_tag":    config_tag,
            "args":          vars(args),
        }, ckpt_latest)

        # ---- best checkpoint ----
        if ep_val < best_val_loss:
            best_val_loss = ep_val
            patience_ctr  = 0
            torch.save({
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "epoch":         epoch,
                "best_val_loss": best_val_loss,
                "head_mask":     head_mask,
                "neuron_mask":   neuron_mask,
                "config_tag":    config_tag,
                "args":          vars(args),
            }, ckpt_best)
            print(f"    ↑ best val_loss={best_val_loss:.4f} → saved")
        else:
            patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(f"  Early stopping (patience={args.patience}).")
                break

        # ---- full metric evaluation ----
        if (epoch + 1) % args.eval_every == 0:
            ev = evaluate(model, val_loaders, args.dataset_names, device)
            m  = ev["macro"]
            print(f"    macro Dice={m['mean_dice']:.4f}  "
                  f"BF1={m['mean_boundary_f1']:.4f}  "
                  f"HD95={m['mean_hd95']:.2f}")
            history["eval_per_epoch"].append({"epoch": epoch + 1, **ev})

    # ------------------------------------------------------------------
    # 7. Final evaluation on best checkpoint
    # ------------------------------------------------------------------
    print(f"\n[7] Final evaluation on best checkpoint ...")
    best_ck = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model"])
    final_eval  = evaluate(model, val_loaders, args.dataset_names, device)
    final_stats = compute_cascade_stats(model, head_mask, neuron_mask)
    fm          = final_eval["macro"]

    delta_bf1 = fm["mean_boundary_f1"] - chosen_cfg["macro"]["mean_boundary_f1"]
    print("\n" + "=" * 80)
    print(f"RESULTS [{config_tag}]  (best epoch={int(best_ck['epoch'])+1})")
    print(f"  Phase-C BF1 (no finetune) : "
          f"{chosen_cfg['macro']['mean_boundary_f1']:.4f}")
    print(f"  Fine-tuned macro BF1      : {fm['mean_boundary_f1']:.4f}  "
          f"(Δ={delta_bf1:+.4f})")
    print(f"  Fine-tuned macro Dice     : {fm['mean_dice']:.4f}")
    print(f"  Fine-tuned macro HD95     : {fm['mean_hd95']:.2f}")
    for n in args.dataset_names:
        pd = final_eval["per_dataset"][n]
        print(f"  [{n}] Dice={pd['mean_dice']:.4f}  "
              f"BF1={pd['mean_boundary_f1']:.4f}  "
              f"HD95={pd['mean_hd95']:.2f}")
    print(f"  param_reduction : {final_stats.get('param_reduction_pct', 0):.1f}%")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 8. Save results JSON
    # ------------------------------------------------------------------
    results = {
        "run_id":          run_id,
        "config_tag":      config_tag,
        "config":          vars(args),
        "head_sp_target":  float(head_sp),
        "mlp_sp_target":   float(mlp_sp),
        "pruning_stats":   final_stats,
        "best_epoch":      int(best_ck["epoch"]) + 1,
        "best_val_loss":   float(best_val_loss),
        "phase_c_metrics": chosen_cfg["macro"],
        "final_eval":      final_eval,
        "history":         history,
    }
    results_path = os.path.join(out_dir, "finetune_results.json")
    with open(results_path, "w") as fp:
        json.dump(
            json.loads(json.dumps(results, default=_json_safe)),
            fp, indent=2,
        )

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[Done]  {total_min:.1f} min")
    print(f"  results   → {results_path}")
    print(f"  best ckpt → {ckpt_best}")

    remove_hooks(all_hooks)


if __name__ == "__main__":
    main()
