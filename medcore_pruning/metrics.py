"""Binary region and boundary metrics; distances are measured in pixels."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def compute_dice(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    denominator = pred.sum() + gt.sum()
    return 1.0 if denominator == 0 else float(2 * (pred & gt).sum() / denominator)


def compute_iou(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def _extract_boundary(mask, radius=2):
    mask = np.asarray(mask, dtype=bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask, structure=structure) & ~binary_erosion(mask, structure=structure)


def compute_boundary_f1(pred, gt, radius=2):
    """Match morphological boundary bands with a radius-pixel tolerance."""
    pred_boundary, gt_boundary = _extract_boundary(pred, radius), _extract_boundary(gt, radius)
    n_pred, n_gt = pred_boundary.sum(), gt_boundary.sum()
    if n_pred == 0 or n_gt == 0:
        return float(n_pred == n_gt)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    precision = (pred_boundary & binary_dilation(gt_boundary, structure=structure)).sum() / n_pred
    recall = (gt_boundary & binary_dilation(pred_boundary, structure=structure)).sum() / n_gt
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compute_hd95(pred, gt):
    """Maximum directed 95th-percentile surface distance.

    Both empty masks give 0; exactly one empty mask gives the largest image
    dimension. Surfaces are one-pixel inner boundaries using 8-connectivity.
    """
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if not pred.any() or not gt.any():
        return 0.0 if pred.any() == gt.any() else float(max(pred.shape))
    structure = np.ones((3, 3), dtype=bool)
    pred_surface = pred & ~binary_erosion(pred, structure=structure)
    gt_surface = gt & ~binary_erosion(gt, structure=structure)
    to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    return float(max(np.percentile(to_gt, 95), np.percentile(to_pred, 95)))


def compute_all_metrics(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Metrics require binary masks with matching two-dimensional shapes.")
    return {"dice": compute_dice(pred, gt), "iou": compute_iou(pred, gt),
            "boundary_f1": compute_boundary_f1(pred, gt), "hd95": compute_hd95(pred, gt)}
