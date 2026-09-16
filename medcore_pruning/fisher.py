"""Boundary-aware and cross-checkpoint Fisher scoring (paper Sections 3.3-3.5)."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry
from .scoring import compute_head_scores, compute_mlp_neuron_scores


def _dice_loss_sum(pred_logits, target):
    pred   = torch.sigmoid(pred_logits)
    pred   = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    inter  = (pred * target).sum(dim=1)
    per    = 1.0 - (2.0 * inter + 1e-5) / (
        pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5)
    return per.sum()



def compute_diagonal_fisher_boundary_aware(model, dataloader, device,
                                            boundary_weight=3.0):
    """Estimate per-sample diagonal Fisher using boundary-weighted BCE and Dice."""
    if dataloader.batch_size != 1:
        raise ValueError("Fisher estimation requires batch_size=1 for per-sample gradients.")
    model.eval()
    for p in model.parameters():        p.requires_grad_(False)
    for p in model.image_encoder.parameters(): p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p, device="cpu")
              for n, p in model.image_encoder.named_parameters()}
    k3 = torch.ones(1, 1, 3, 3, device=device)
    n_processed = nan_batches = 0

    for batch in tqdm(dataloader, desc="Fisher (boundary-aware)", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask_256"].to(device).float()
        bboxes = batch["bbox"].to(device).float()
        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]
        B = images.shape[0]

        with torch.no_grad():
            dilated      = F.conv2d(masks,           k3, padding=1).clamp(0, 1)
            eroded       = 1.0 - F.conv2d(1.0 - masks, k3, padding=1).clamp(0, 1)
            weight_map   = 1.0 + boundary_weight * (dilated - eroded)

        model.zero_grad()
        image_emb = model.image_encoder(images)
        with torch.no_grad():
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=bboxes, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        bce_pp = F.binary_cross_entropy_with_logits(low_res, masks, reduction="none")
        loss   = (weight_map * bce_pp).sum() + _dice_loss_sum(low_res, masks)
        loss.backward()

        has_nan = any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                      for _, p in model.image_encoder.named_parameters())
        if has_nan:
            nan_batches += 1
            model.zero_grad()
            continue

        with torch.no_grad():
            for n, p in model.image_encoder.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float().cpu() ** 2
        model.zero_grad()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        n_processed += B

    if nan_batches: print(f"  WARNING: {nan_batches} batches skipped (NaN grads).")
    if n_processed == 0:
        raise RuntimeError("No finite calibration gradients were available.")
    for n in fisher: fisher[n] /= n_processed
    for p in model.image_encoder.parameters(): p.requires_grad_(False)
    return fisher



def compute_sam_fisher_on_medical(sam_ckpt, dataloader, device,
                                   boundary_weight=3.0):
    print("  Loading SAM for cross-Fisher estimation ...")
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam_model = sam_model.to(device).eval()
    for p in sam_model.prompt_encoder.parameters(): p.requires_grad_(False)
    for p in sam_model.mask_decoder.parameters():   p.requires_grad_(False)

    fisher_s = compute_diagonal_fisher_boundary_aware(
        sam_model, dataloader, device, boundary_weight=boundary_weight)

    del sam_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("  SAM Fisher done; SAM model freed from GPU.")
    return fisher_s



def compute_cross_fisher(fisher_m, fisher_s):
    cross = {}
    for n in fisher_m:
        fm = fisher_m[n].float()
        fs = fisher_s.get(n, torch.zeros_like(fm)).float()
        cross[n] = torch.sqrt(fm * fs + 1e-30)
    return cross



def build_cross_fisher_list(fisher_m_list, fisher_s):
    return [compute_cross_fisher(fm, fisher_s) for fm in fisher_m_list]



def sum_fisher_dicts(fisher_list, pi=None):
    R  = len(fisher_list)
    if pi is None:
        pi = np.full(R, 1.0 / R)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    combined = {}
    for n in fisher_list[0]:
        acc = torch.zeros_like(fisher_list[0][n])
        for r, f in enumerate(fisher_list):
            acc = acc + float(pi[r]) * f[n]
        combined[n] = acc
    return combined



def compute_scores_per_subset(model, sam_params, fisher_m_list, cross_fisher_list):
    """Δ_zero uses F^M; Δ_reset uses √(F^M·F^S)."""
    dz_head_list, dr_head_list = [], []
    dz_mlp_list,  dr_mlp_list  = [], []

    for fm, fc in zip(fisher_m_list, cross_fisher_list):
        dz_h, _  = compute_head_scores(model, sam_params, fm)
        dz_m, _  = compute_mlp_neuron_scores(model, sam_params, fm)
        _,  dr_h = compute_head_scores(model, sam_params, fc)
        _,  dr_m = compute_mlp_neuron_scores(model, sam_params, fc)
        dz_head_list.append(dz_h);  dr_head_list.append(dr_h)
        dz_mlp_list.append(dz_m);   dr_mlp_list.append(dr_m)

    return dz_head_list, dr_head_list, dz_mlp_list, dr_mlp_list



def compute_adaptive_alpha_per_block(dz_head, dr_head,
                                      num_blocks=12, num_heads=12):
    alpha_b = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        dz_b = dz_head[b * num_heads: (b + 1) * num_heads]
        dr_b = dr_head[b * num_heads: (b + 1) * num_heads]
        if dz_b.std() > 1e-12 and dr_b.std() > 1e-12:
            dz_centered = dz_b.astype(np.float64) - dz_b.astype(np.float64).mean()
            dr_centered = dr_b.astype(np.float64) - dr_b.astype(np.float64).mean()
            denominator = np.sqrt(np.sum(dz_centered ** 2) * np.sum(dr_centered ** 2))
            r = float(np.sum(dz_centered * dr_centered) / denominator)
            r = float(np.clip(r, -1.0, 1.0))
        else:
            r = 1.0
        alpha_b[b] = (1.0 + r) / 2.0
    return alpha_b
