# -*- coding: utf-8 -*-
"""
Polyp Segmentation Dataset Loader for MedSAM Pruning Pilot Study.

Supports: CVC-ColonDB, CVC-ClinicDB, Kvasir-SEG (and any images/masks layout).

Changes vs original:
  P1: compute_freq_weight()  — FFT-based high-frequency energy ratio (ω_freq)
  P1: __getitem__ returns "omega_freq" for each sample
  P2: _sort_key()            — robust sort for numeric and alphanumeric filenames
  P2: bbox_shift=0 for calibration set (deterministic bboxes for gradient stability)
  P2: build_dataloaders returns cal_freq_weights as 5th element
"""

import os
import random
import numpy as np
from skimage import io, transform as sk_transform
import torch
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# P1: Frequency Prior Utility
# ==============================================================================

def compute_freq_weight(mask, r_c=None):
    """
    Compute high-frequency energy ratio of a binary mask via 2D FFT.

    ω_i^freq = Σ_{(u,v)∈H} |F(mask)(u,v)|²  /  Σ_{(u,v)} |F(mask)(u,v)|²

    where H = {(u,v) : dist_from_center > r_c},
          r_c = min(H, W) / 4  by default (spec §3).

    Args:
        mask: 2D binary numpy array (H, W), values in {0, 1}
        r_c:  cutoff radius in pixels. Default: min(H,W)/4.

    Returns:
        float in [0, 1].  Higher → more high-frequency content → harder boundary.
    """
    H, W = mask.shape
    if r_c is None:
        r_c = min(H, W) / 4.0

    F = np.fft.fft2(mask.astype(np.float32))
    F_shifted = np.fft.fftshift(F)
    power = np.abs(F_shifted) ** 2                   # (H, W) power spectrum

    cy, cx = H // 2, W // 2
    y_grid, x_grid = np.ogrid[:H, :W]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    total_power = power.sum()
    if total_power < 1e-8:
        return 0.0
    return float(power[dist > r_c].sum() / total_power)


# ==============================================================================
# P2: Robust filename sorting
# ==============================================================================

def _sort_key(filename):
    """
    Sort key that handles both numeric filenames (CVC-ColonDB: "1.png")
    and alphanumeric filenames (Kvasir-SEG: "cju0roawvklxq...jpg").
    Numeric files sort numerically; others fall back to lexicographic.
    """
    stem = os.path.splitext(filename)[0]
    try:
        return (0, int(stem), "")
    except ValueError:
        return (1, 0, stem)


# ==============================================================================
# Dataset
# ==============================================================================

class PolypDataset(Dataset):
    """
    Dataset for polyp segmentation.

    Directory structure:
        data_root/
            images/   -> *.png / *.jpg  (RGB colonoscopy images)
            masks/    -> *.png / *.jpg  (binary segmentation masks)

    Supports CVC-ColonDB, CVC-ClinicDB (numeric filenames) and
    Kvasir-SEG (alphanumeric filenames).
    """

    def __init__(self, data_root, image_size=1024, bbox_shift=5):
        """
        Args:
            data_root:   path to dataset root
            image_size:  target size for SAM input (default 1024)
            bbox_shift:  random bbox perturbation range in pixels.
                         Pass 0 for calibration sets so bboxes are deterministic
                         across multiple accesses of the same sample.
        """
        self.data_root = data_root
        self.image_size = image_size
        self.bbox_shift = bbox_shift

        img_dir = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")

        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)],
            key=_sort_key,
        )
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.lower().endswith(img_exts)],
            key=_sort_key,
        )

        assert len(self.img_files) == len(self.mask_files), (
            f"Image/mask count mismatch: {len(self.img_files)} vs {len(self.mask_files)}"
        )

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # ------------------------------------------------------------------ #
        # Image
        # ------------------------------------------------------------------ #
        img_path = os.path.join(self.img_dir, self.img_files[index])
        img = io.imread(img_path)          # (H, W, 3) uint8

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]            # drop alpha channel

        H_orig, W_orig = img.shape[:2]

        img_1024 = sk_transform.resize(
            img, (self.image_size, self.image_size),
            order=3, preserve_range=True, anti_aliasing=True,
        ).astype(np.float64)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)   # (3, 1024, 1024)

        # ------------------------------------------------------------------ #
        # Mask
        # ------------------------------------------------------------------ #
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])
        mask = io.imread(mask_path)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask = (mask > 127).astype(np.uint8)         # binary, original resolution

        # ------------------------------------------------------------------ #
        # P1: Frequency weight — computed on the ORIGINAL resolution mask
        # ------------------------------------------------------------------ #
        omega_freq = compute_freq_weight(mask)        # scalar float in [0, 1]

        # ------------------------------------------------------------------ #
        # Resize masks
        # ------------------------------------------------------------------ #
        mask_256 = sk_transform.resize(
            mask, (256, 256),
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

        mask_1024 = sk_transform.resize(
            mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

        # ------------------------------------------------------------------ #
        # Bounding box from 1024x1024 mask
        # ------------------------------------------------------------------ #
        y_indices, x_indices = np.where(mask_1024 > 0)
        if len(y_indices) == 0:
            bbox = np.array([0, 0, self.image_size, self.image_size], dtype=np.float32)
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # P2: only perturb when bbox_shift > 0 (i.e. test set)
            if self.bbox_shift > 0:
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(self.image_size, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(self.image_size, y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return {
            "image":         img_tensor,                                     # (3, 1024, 1024)
            "mask_256":      torch.tensor(mask_256).long().unsqueeze(0),     # (1, 256, 256)
            "mask_1024":     torch.tensor(mask_1024).long(),                 # (1024, 1024)
            "bbox":          torch.tensor(bbox).float(),                     # (4,)
            "omega_freq":    torch.tensor(omega_freq, dtype=torch.float32),  # P1: scalar
            "name":          self.img_files[index],
            "original_size": (H_orig, W_orig),
        }


# ==============================================================================
# Data loader builder
# ==============================================================================

def build_dataloaders(data_root, n_calibration=128, batch_size=4, seed=42, num_workers=4):
    """
    Split dataset into calibration and test sets.

    Key differences vs original:
      P2: Calibration set uses bbox_shift=0 (deterministic bbox → stable gradients).
          Test set keeps bbox_shift=5 (standard evaluation).
      P1: Returns cal_freq_weights (np.ndarray, shape [n_calibration]) as 5th element.

    Returns:
        cal_loader, test_loader, cal_dataset, test_dataset, cal_freq_weights
    """
    # Two separate dataset objects so bbox behaviour differs between splits
    cal_full  = PolypDataset(data_root, bbox_shift=0)   # P2: no perturbation for calibration
    test_full = PolypDataset(data_root, bbox_shift=5)

    n_total = len(cal_full)
    assert n_calibration < n_total, (
        f"Calibration size {n_calibration} >= total dataset size {n_total}"
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    cal_indices  = indices[:n_calibration]
    test_indices = indices[n_calibration:]

    cal_dataset  = torch.utils.data.Subset(cal_full,  cal_indices)
    test_dataset = torch.utils.data.Subset(test_full, test_indices)

    cal_loader = DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # P1: Pre-extract frequency weights for all calibration samples
    # (access the underlying full dataset with bbox_shift=0)
    cal_freq_weights = np.array(
        [cal_full[idx]["omega_freq"].item() for idx in cal_indices],
        dtype=np.float32,
    )

    cv = cal_freq_weights.std() / max(cal_freq_weights.mean(), 1e-8)
    print(f"Dataset split : {n_calibration} calibration, {len(test_indices)} test")
    print(f"ω_freq stats  : mean={cal_freq_weights.mean():.3f}  "
          f"std={cal_freq_weights.std():.3f}  cv={cv:.3f}  "
          f"(cv >= 0.3 recommended for discriminative prior)")

    return cal_loader, test_loader, cal_dataset, test_dataset, cal_freq_weights
