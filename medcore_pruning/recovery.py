"""Five-loss recovery of a pruned MedSAM image encoder."""

import numpy as np
import torch
import torch.nn.functional as F


def _dice_bce_loss(pred, gt):
    probabilities = pred.sigmoid()
    intersection = (probabilities * gt).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) + 1e-6
    dice = 1.0 - (2.0 * intersection / denominator).mean()
    return 0.5 * F.binary_cross_entropy_with_logits(pred, gt) + 0.5 * dice


def _boundary_mask(gt):
    """Morphological boundary from 3-by-3 dilation minus erosion."""
    kernel = torch.ones(1, 1, 3, 3, device=gt.device, dtype=gt.dtype)
    dilated = F.conv2d(gt, kernel, padding=1).clamp(0, 1)
    eroded = 1.0 - F.conv2d(1.0 - gt, kernel, padding=1).clamp(0, 1)
    return (dilated - eroded).clamp(0, 1)


def _freq_loss(pred_sig, gt, low_cutoff=0.25):
    """Mean absolute FFT error above the radial frequency cutoff."""
    height, width = pred_sig.shape[-2:]
    prediction_fft, target_fft = torch.fft.rfft2(pred_sig.float()), torch.fft.rfft2(gt.float())
    fy = torch.fft.fftfreq(height, device=pred_sig.device)
    fx = torch.fft.rfftfreq(width, device=pred_sig.device)
    high = (fy[:, None].square() + fx[None, :].square()).sqrt() > low_cutoff
    return ((prediction_fft - target_fft).abs() * high).mean()


def _decode(model, embeddings, boxes):
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes[:, None, :], masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embeddings, image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    return logits


def recovery_finetune(model, cal_loader, device, n_steps=100, lr=1e-5,
                      feature_teacher=None, feat_distill_weight=0.5,
                      boundary_loss_weight=1.0, logit_distill_weight=2.0,
                      freq_pred_loss_weight=0.5, sampling_weights=None):
    """Adapt the encoder with segmentation, boundary, distillation, and FFT losses.

    Attach structured pruning masks before calling. The prompt encoder and mask
    decoder stay frozen. A teacher may remain on CPU to reduce GPU memory use.
    Sampling weights follow calibration dataset order and favor complex masks.
    """
    if n_steps <= 0:
        return
    dataset = cal_loader.dataset
    n_cal, batch_size = len(dataset), cal_loader.batch_size or 1
    if not n_cal:
        raise ValueError("Recovery requires nonempty calibration data.")
    if lr <= 0 or not np.isfinite(lr):
        raise ValueError("Recovery learning rate must be positive and finite.")
    sample_probs = None
    if sampling_weights is not None:
        sample_probs = np.asarray(sampling_weights, dtype=np.float64)
        if (sample_probs.shape != (n_cal,) or not np.isfinite(sample_probs).all()
                or (sample_probs < 0).any() or sample_probs.sum() <= 0):
            raise ValueError("Sampling weights must be finite, nonnegative, and match calibration data.")
        sample_probs = sample_probs / sample_probs.sum()
    model.train()
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad_(True)
    for module in (model.prompt_encoder, model.mask_decoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.image_encoder.parameters(), lr=lr, weight_decay=0.0)
    need_teacher = feature_teacher is not None and (feat_distill_weight > 0 or logit_distill_weight > 0)
    if need_teacher:
        feature_teacher.eval()
        teacher_device = next(feature_teacher.parameters()).device

    for _ in range(n_steps):
        indices = np.random.choice(n_cal, size=batch_size, replace=True, p=sample_probs)
        items = [dataset[int(index)] for index in indices]
        images = torch.stack([item["image"] for item in items]).to(device)
        targets = torch.stack([item["mask_1024"] for item in items]).unsqueeze(1).float().to(device)
        targets_low = torch.stack([item["mask_256"] for item in items]).float().to(device)
        boxes = torch.stack([item["bbox"] for item in items]).float().to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.image_encoder(images)
        logits = _decode(model, embeddings, boxes)
        predictions = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
        loss = _dice_bce_loss(predictions, targets)

        if boundary_loss_weight > 0:
            weights = 1.0 + 4.0 * _boundary_mask(targets)
            loss = loss + boundary_loss_weight * F.binary_cross_entropy_with_logits(
                predictions, targets, weight=weights)
        if need_teacher:
            with torch.no_grad():
                teacher_embeddings = feature_teacher.image_encoder(images.to(teacher_device))
                if logit_distill_weight > 0:
                    teacher_logits = _decode(feature_teacher, teacher_embeddings, boxes.to(teacher_device))
            if feat_distill_weight > 0:
                loss = loss + feat_distill_weight * F.mse_loss(embeddings, teacher_embeddings.to(device))
            if logit_distill_weight > 0:
                logit_error = (logits - teacher_logits.to(device)).square()
                loss = loss + logit_distill_weight * (_boundary_mask(targets_low) * logit_error).mean()
        if freq_pred_loss_weight > 0:
            loss = loss + freq_pred_loss_weight * _freq_loss(logits.sigmoid(), targets_low)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite recovery loss; check data and recovery settings.")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.image_encoder.parameters()):
            raise RuntimeError("Non-finite recovery gradients; check data and recovery settings.")
        optimizer.step()
    model.eval()
