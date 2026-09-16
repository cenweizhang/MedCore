"""Segmentation evaluation and calibration sampling weights."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_metrics
from .recovery import _boundary_mask
from .training import segmentation_logits


@torch.no_grad()
def _eval_and_collect(model, loader, device, desc="Eval"):
    model.eval()
    metrics = []
    for batch in tqdm(loader, desc=desc, leave=False):
        targets = batch["mask_1024"].cpu().numpy()
        logits = segmentation_logits(model, batch, device)
        logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
        predictions = (logits.sigmoid() > 0.5).cpu().numpy()[:, 0]
        metrics.extend(compute_all_metrics(pred, target) for pred, target in zip(predictions, targets))
    if not metrics:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {f"mean_{key}": float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def eval_all(model, per_dataset_test_loaders, dataset_names, device):
    """Return image means per dataset, equal-weight macro means, and worst values."""
    if not dataset_names or len(per_dataset_test_loaders) != len(dataset_names):
        raise ValueError("Provide one nonempty loader and unique name per dataset.")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Dataset names must be unique.")
    per = {name: _eval_and_collect(model, loader, device, desc=f"Eval[{name}]")
           for name, loader in zip(dataset_names, per_dataset_test_loaders)}
    keys = next(iter(per.values())).keys()
    macro = {key: float(np.mean([values[key] for values in per.values()])) for key in keys}
    worst = {key: float((max if key == "mean_hd95" else min)(
        values[key] for values in per.values())) for key in keys}
    return {"per_dataset": per, "macro": macro, "worst": worst}


def compute_boundary_sampling_weights(cal_loader):
    """Compute perimeter / (sqrt(area) + 1) in dataset order at mask resolution."""
    ordered = DataLoader(cal_loader.dataset, batch_size=1, shuffle=False, num_workers=0)
    weights = []
    for batch in ordered:
        mask = batch["mask_256"].float()
        perimeter, area = _boundary_mask(mask).sum().item(), mask.sum().item()
        weights.append(max(perimeter / (area ** 0.5 + 1.0), 1e-4))
    return np.asarray(weights, dtype=np.float32)
