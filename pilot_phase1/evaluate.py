# -*- coding: utf-8 -*-
"""
Evaluation pipeline for pruned MedSAM model.

Given a head mask, run inference on the test set and compute metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .metrics import compute_all_metrics


@torch.no_grad()
def evaluate_pruned_model(model, test_loader, head_mask, device,
                          num_blocks=12, num_heads=12):
    """
    Evaluate a pruned MedSAM model on the test set.
    
    Args:
        model: MedSAM (Sam) model
        test_loader: test data loader (batch_size=1 recommended)
        head_mask: (total_heads,) binary mask or None for unpruned
        device: torch device
        num_blocks, num_heads: architecture params
    
    Returns:
        dict with averaged metrics and per-sample results
    """
    from .head_pruning import apply_head_mask_to_model, remove_hooks
    
    model.eval()
    
    # Apply head mask if provided
    hooks = []
    if head_mask is not None:
        hooks = apply_head_mask_to_model(model, head_mask, num_blocks, num_heads)
    
    all_metrics = []
    
    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)      # (1, 3, 1024, 1024)
        masks_gt = batch["mask_1024"].numpy()    # (1, 1024, 1024)
        bboxes = batch["bbox"].numpy()           # (1, 4)
        
        # Forward pass through image encoder
        image_embedding = model.image_encoder(images)  # (1, 256, 64, 64)
        
        # Prompt encoding (box prompt)
        box_torch = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (1, 1, 4)
        
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        
        # Mask decoding
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )  # (1, 1, 256, 256)
        
        # Upsample to original resolution
        pred_mask = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, 1024, 1024)
        
        pred_mask = torch.sigmoid(pred_mask)
        pred_binary = (pred_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        gt_binary = masks_gt[0].astype(np.uint8)
        
        metrics = compute_all_metrics(pred_binary, gt_binary)
        metrics["name"] = batch["name"][0]
        all_metrics.append(metrics)
    
    # Remove hooks
    if hooks:
        remove_hooks(hooks)
    
    # Aggregate metrics
    metric_keys = ["dice", "iou", "boundary_f1", "hd95"]
    avg_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[f"mean_{key}"] = np.mean(values)
        avg_metrics[f"std_{key}"] = np.std(values)
    
    # Count parameters (approximate: count kept heads)
    if head_mask is not None:
        n_kept = int(head_mask.sum())
        n_total = len(head_mask)
        avg_metrics["kept_heads"] = n_kept
        avg_metrics["total_heads"] = n_total
        avg_metrics["head_sparsity"] = 1.0 - n_kept / n_total
    else:
        avg_metrics["kept_heads"] = num_blocks * num_heads
        avg_metrics["total_heads"] = num_blocks * num_heads
        avg_metrics["head_sparsity"] = 0.0
    
    return {
        "avg_metrics": avg_metrics,
        "per_sample": all_metrics,
    }
