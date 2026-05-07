# -*- coding: utf-8 -*-
"""
Post-pruning recovery fine-tuning for MedSAM cascade pruning (v5).

Provides recovery_finetune() which runs a short gradient-based adaptation
of the pruned image encoder using five loss components:
  1. Dice + BCE segmentation loss
  2. Boundary-weighted BCE (boundary_loss_weight)
  3. Feature distillation from unpruned teacher (feat_distill_weight)
  4. Logit distillation in boundary regions (logit_distill_weight)
  5. High-frequency prediction loss — L_freq (freq_pred_loss_weight)

v8.5 additions:
  boundary_freq_gamma: per-sample adaptive boundary loss weight scale.
    effective_lambda_b^(i) = boundary_loss_weight * (1 + gamma * omega_freq^(i))
    High-frequency samples (complex boundaries) receive stronger boundary supervision.
  contour_loss_weight: L_contour = MSE(|∇σ(pred)|, boundary_map).
    Forces prediction gradient magnitude to align with GT boundary locations.
    Directly optimises contour sharpness, targeting BF1 degradation.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _dice_bce_loss(pred, gt):
    """pred: (B,1,H,W) logits; gt: (B,1,H,W) float in {0,1}"""
    bce = F.binary_cross_entropy_with_logits(pred, gt)
    p   = torch.sigmoid(pred)
    num   = (p * gt).sum(dim=(-2, -1))
    denom = p.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) + 1e-6
    dice  = 1.0 - (2.0 * num / denom).mean()
    return 0.5 * bce + 0.5 * dice


def _boundary_mask(gt):
    """
    Morphological boundary via 3×3 dilation minus erosion.
    gt: (B,1,H,W) float in {0,1} → returns (B,1,H,W) float boundary.
    """
    k   = torch.ones(1, 1, 3, 3, device=gt.device, dtype=gt.dtype)
    dil = F.conv2d(gt, k, padding=1).clamp(0, 1)
    ero = 1.0 - F.conv2d(1.0 - gt, k, padding=1).clamp(0, 1)
    return (dil - ero).clamp(0, 1)


def _freq_loss(pred_sig, gt, low_cutoff=0.25):
    """
    L_freq = Σ_{ω>cutoff} |FFT(pred_sig) − FFT(gt)|.
    pred_sig: (B,1,H,W) sigmoid values; gt: (B,1,H,W) float masks.
    """
    B, C, H, W = pred_sig.shape
    pf = torch.fft.rfft2(pred_sig.float())
    gf = torch.fft.rfft2(gt.float())
    fy = torch.fft.fftfreq(H, device=pred_sig.device)
    fx = torch.fft.rfftfreq(W, device=pred_sig.device)
    dist = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()          # (H, W//2+1)
    high_mask = (dist > low_cutoff).float()[None, None]           # (1,1,H,W//2+1)
    return (torch.abs(pf - gf) * high_mask).mean()


def _contour_consistency_loss(pred_logits, boundary_map):
    """
    L_contour = MSE(|∇σ(pred)|, boundary_map * 0.25)

    sigmoid gradient max = σ'(0) = 0.25, so boundary_map is scaled to [0, 0.25]
    to match the reachable range of |∇σ|.  Without this scaling the MSE target
    (0/1) is unreachable and produces only constant gradients.

    Forces prediction gradient magnitude to be large (~0.25) at GT boundary
    pixels and near-zero at interior pixels.
    boundary_map: (B,1,H,W) float [0,1] from _boundary_mask().
    """
    p  = torch.sigmoid(pred_logits)
    gy = F.pad(p[:, :, 1:, :] - p[:, :, :-1, :], (0, 0, 0, 1))   # (B,1,H,W)
    gx = F.pad(p[:, :, :, 1:] - p[:, :, :, :-1], (0, 1, 0, 0))   # (B,1,H,W)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    target = boundary_map * 0.25   # scale to reachable sigmoid-gradient range
    return F.mse_loss(grad_mag, target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recovery_finetune(
    model,
    cal_loader,
    device,
    head_mask=None,
    neuron_mask=None,
    n_steps=100,
    lr=1e-5,
    feature_teacher=None,
    feat_distill_weight=0.5,
    boundary_loss_weight=1.0,
    logit_distill_weight=2.0,
    freq_pred_loss_weight=0.5,
    freq_weights=None,
    boundary_freq_gamma=0.0,
    contour_loss_weight=0.0,
):
    """
    Fine-tune the image encoder of a pruned MedSAM model.

    The pruning hooks are assumed to be already registered externally;
    head_mask / neuron_mask are accepted only for API compatibility and
    are not used here.

    Args:
        model               : pruned student model on `device`.
        cal_loader          : calibration DataLoader.
        device              : torch.device for student forward pass.
        head_mask           : unused (hooks registered externally).
        neuron_mask         : unused (hooks registered externally).
        n_steps             : gradient steps to run.
        lr                  : AdamW learning rate.
        feature_teacher     : deepcopy of unpruned MedSAM on CPU (optional).
        feat_distill_weight : MSE weight on image-encoder embedding.
        boundary_loss_weight: boundary-pixel-weighted BCE weight.
        logit_distill_weight: boundary-region logit MSE distillation weight.
        freq_pred_loss_weight: high-frequency prediction loss weight.
        freq_weights        : (n_cal,) float32 per-sample sampling weights.
        boundary_freq_gamma : v8.5 — scale boundary_loss_weight per sample by
                              (1 + gamma * omega_freq). gamma=0 = original behaviour.
        contour_loss_weight : v8.5 — MSE(|∇σ(pred)|, boundary_map) weight.
                              Directly targets contour sharpness / BF1.
    """
    if n_steps <= 0:
        return

    # Image encoder is trainable; prompt encoder + mask decoder stay frozen.
    model.train()
    for p in model.image_encoder.parameters():
        p.requires_grad_(True)
    for p in model.prompt_encoder.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.image_encoder.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    cal_dataset = cal_loader.dataset
    n_cal       = len(cal_dataset)
    batch_size  = cal_loader.batch_size or 1

    if freq_weights is not None:
        sample_probs = (freq_weights / freq_weights.sum()).astype(np.float64)
        sample_probs /= sample_probs.sum()  # re-normalize to fix float64 rounding
    else:
        sample_probs = None

    need_teacher = feature_teacher is not None and (
        feat_distill_weight > 0 or logit_distill_weight > 0
    )

    for step in range(n_steps):
        # ---- sample a mini-batch (weighted or uniform) ----
        if sample_probs is not None:
            idxs = np.random.choice(n_cal, size=batch_size, replace=True,
                                    p=sample_probs)
        else:
            idxs = np.random.choice(n_cal, size=batch_size, replace=True)

        items   = [cal_dataset[int(i)] for i in idxs]
        images  = torch.stack([b["image"]     for b in items]).to(device)      # (B,3,1024,1024)
        gt_1024 = (torch.stack([b["mask_1024"] for b in items])
                   .float().unsqueeze(1).to(device))                            # (B,1,1024,1024)
        gt_256  = (torch.stack([b["mask_256"]  for b in items])
                   .float().to(device))                                         # (B,1,256,256)
        bboxes  = torch.stack([b["bbox"] for b in items]).to(device)            # (B,4)

        optimizer.zero_grad()

        # ---- student forward ----
        image_emb = model.image_encoder(images)                                # (B,256,64,64)
        box_t = bboxes[:, None, :]                                             # (B,1,4)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_t, masks=None
        )
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )                                                                       # (B,1,256,256)

        pred_1024 = F.interpolate(
            low_res, size=(1024, 1024), mode="bilinear", align_corners=False
        )                                                                       # (B,1,1024,1024)

        # ---- 1. Dice + BCE ----
        total_loss = _dice_bce_loss(pred_1024, gt_1024)

        # ---- 2. Boundary-weighted BCE (v8.5: per-sample adaptive lambda_b) ----
        if boundary_loss_weight > 0:
            bnd  = _boundary_mask(gt_1024)                                     # (B,1,1024,1024)
            w    = 1.0 + bnd * 4.0
            if boundary_freq_gamma > 0 and "omega_freq" in items[0]:
                ofs = torch.stack([b["omega_freq"] for b in items]).float().to(device)  # (B,)
                lam = (boundary_loss_weight * (1.0 + boundary_freq_gamma * ofs)
                       ).view(-1, 1, 1, 1)                                     # (B,1,1,1)
                loss_bnd = (lam * F.binary_cross_entropy_with_logits(
                    pred_1024, gt_1024, weight=w, reduction="none"
                )).mean()
            else:
                loss_bnd = F.binary_cross_entropy_with_logits(
                    pred_1024, gt_1024, weight=w, reduction="mean"
                )
                loss_bnd = loss_bnd * boundary_loss_weight
            total_loss = total_loss + loss_bnd

        # ---- 3+4. Teacher distillation (CPU-offloaded) ----
        if need_teacher:
            with torch.no_grad():
                imgs_cpu = images.cpu()
                t_emb = feature_teacher.image_encoder(imgs_cpu)                # (B,256,64,64) CPU

                if logit_distill_weight > 0:
                    tb = bboxes.cpu()[:, None, :]
                    ts, td = feature_teacher.prompt_encoder(
                        points=None, boxes=tb, masks=None
                    )
                    t_low_res, _ = feature_teacher.mask_decoder(
                        image_embeddings=t_emb,
                        image_pe=feature_teacher.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=ts,
                        dense_prompt_embeddings=td,
                        multimask_output=False,
                    )                                                           # (B,1,256,256) CPU
                    t_low_res = t_low_res.to(device)

            if feat_distill_weight > 0:
                loss_feat = F.mse_loss(image_emb, t_emb.to(device))
                total_loss = total_loss + feat_distill_weight * loss_feat

            if logit_distill_weight > 0:
                bnd_256   = _boundary_mask(gt_256)                             # (B,1,256,256)
                loss_logit = (bnd_256 * (low_res - t_low_res).pow(2)).mean()
                total_loss = total_loss + logit_distill_weight * loss_logit

        # ---- 5. Frequency-domain prediction loss ----
        if freq_pred_loss_weight > 0:
            loss_freq  = _freq_loss(torch.sigmoid(low_res), gt_256)
            total_loss = total_loss + freq_pred_loss_weight * loss_freq

        # ---- 6. Contour consistency loss (v8.5) ----
        if contour_loss_weight > 0:
            bnd_1024 = _boundary_mask(gt_1024) if boundary_loss_weight <= 0 else bnd
            loss_contour = _contour_consistency_loss(pred_1024, bnd_1024)
            total_loss = total_loss + contour_loss_weight * loss_contour

        total_loss.backward()
        optimizer.step()

    model.eval()
