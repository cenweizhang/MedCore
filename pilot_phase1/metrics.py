# -*- coding: utf-8 -*-
"""
Segmentation evaluation metrics for pilot study.

Provides: Dice, IoU, Boundary F1, HD95.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def compute_dice(pred, gt):
    """
    Dice coefficient.
    
    Args:
        pred: binary prediction (H, W), values in {0, 1}
        gt: binary ground truth (H, W), values in {0, 1}
    
    Returns:
        float: Dice score in [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    
    intersection = (pred & gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


def compute_iou(pred, gt):
    """
    Intersection over Union (Jaccard index).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-8)


def _extract_boundary(mask, radius=2):
    """
    Extract boundary region of a binary mask using morphological operations.
    
    Args:
        mask: binary mask (H, W)
        radius: boundary width in pixels
    
    Returns:
        boundary: binary boundary mask
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    
    struct = np.ones((2 * radius + 1, 2 * radius + 1))
    dilated = binary_dilation(mask, structure=struct)
    eroded = binary_erosion(mask, structure=struct)
    boundary = dilated & (~eroded)
    return boundary


def compute_boundary_f1(pred, gt, radius=2):
    """
    Boundary F1-score.
    
    Extracts boundaries from both prediction and ground truth,
    then computes precision, recall, F1 on boundary pixels.
    
    Args:
        pred: binary prediction (H, W)
        gt: binary ground truth (H, W)
        radius: boundary extraction width (pixels)
    
    Returns:
        float: Boundary F1 score in [0, 1]
    """
    pred_boundary = _extract_boundary(pred, radius)
    gt_boundary = _extract_boundary(gt, radius)
    
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    
    # Dilate gt boundary for tolerance matching
    struct = np.ones((2 * radius + 1, 2 * radius + 1))
    gt_dilated = binary_dilation(gt_boundary, structure=struct)
    pred_dilated = binary_dilation(pred_boundary, structure=struct)
    
    # Precision: what fraction of pred boundary is near gt boundary
    precision = (pred_boundary & gt_dilated).sum() / (pred_boundary.sum() + 1e-8)
    # Recall: what fraction of gt boundary is near pred boundary
    recall = (gt_boundary & pred_dilated).sum() / (gt_boundary.sum() + 1e-8)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def compute_hd95(pred, gt):
    """
    95th percentile Hausdorff Distance.
    
    Args:
        pred: binary prediction (H, W)
        gt: binary ground truth (H, W)
    
    Returns:
        float: HD95 in pixels. Returns 0 if both empty, max_dim if one is empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return max(pred.shape)  # worst case
    
    # Distance transform: for each False pixel, distance to nearest True pixel
    # We need the opposite: for each True pixel in pred, distance to nearest True pixel in gt
    
    # Distance from each pixel to nearest gt foreground
    dist_to_gt = distance_transform_edt(~gt)
    # Distance from each pixel to nearest pred foreground
    dist_to_pred = distance_transform_edt(~pred)
    
    # Directed distances
    d_pred_to_gt = dist_to_gt[pred]  # distance of each pred pixel to gt
    d_gt_to_pred = dist_to_pred[gt]  # distance of each gt pixel to pred
    
    hd95_pred_to_gt = np.percentile(d_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(d_gt_to_pred, 95)
    
    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def compute_all_metrics(pred, gt):
    """
    Compute all segmentation metrics.
    
    Args:
        pred: binary prediction (H, W), numpy uint8/bool
        gt: binary ground truth (H, W), numpy uint8/bool
    
    Returns:
        dict with keys: dice, iou, boundary_f1, hd95
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    return {
        "dice": compute_dice(pred, gt),
        "iou": compute_iou(pred, gt),
        "boundary_f1": compute_boundary_f1(pred, gt),
        "hd95": compute_hd95(pred, gt),
    }
