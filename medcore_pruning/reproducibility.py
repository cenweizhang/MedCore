"""Random seeds and a shared, portable split manifest for all public commands."""

import copy
import json
import os
import random
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from medcore_pruning.dataset import PolypDataset


def seed_everything(seed=42, deterministic=True):
    """Seed Python, NumPy, and PyTorch; request deterministic kernels when available.

    PyTorch warns if an operation has no deterministic implementation. Exact
    floating-point equality across hardware or dependency versions is not promised.
    """
    if not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id):
    """Seed Python and NumPy from the seed assigned to a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_splits(data_roots, dataset_names, cal_sizes, seed=42,
                      val_fraction=0.2, test_fraction=0.2):
    """Create a JSON-safe train/validation/test manifest for each dataset.

    Calibration is a fixed subset of training. Validation and test sizes are
    floor(N * fraction), with at least one sample each. The remaining samples
    form training. The default is approximately 60/20/20. Reuse the saved
    manifest for pruning, fine-tuning, and evaluation.
    """
    data_roots, dataset_names, cal_sizes = list(data_roots), list(dataset_names), list(cal_sizes)
    if not data_roots or not len(data_roots) == len(dataset_names) == len(cal_sizes):
        raise ValueError("data_roots, dataset_names, and cal_sizes need equal, nonzero lengths.")
    if any(not isinstance(n, str) or not n.strip() for n in dataset_names):
        raise ValueError("Dataset names must be nonempty strings.")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Dataset names must be unique.")
    if not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if (not np.isfinite([val_fraction, test_fraction]).all()
            or min(val_fraction, test_fraction) <= 0
            or val_fraction + test_fraction >= 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    manifest = {
        "version": 1, "seed": int(seed), "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction), "datasets": [],
    }
    resolved_roots = set()
    for root, name, n_cal in zip(data_roots, dataset_names, cal_sizes):
        dataset = PolypDataset(root, bbox_shift=0)
        if dataset.data_root in resolved_roots:
            raise ValueError(f"Dataset root is listed more than once: {dataset.data_root}")
        resolved_roots.add(dataset.data_root)
        n_total = len(dataset)
        n_val, n_test = max(1, int(n_total * val_fraction)), max(1, int(n_total * test_fraction))
        n_train = n_total - n_val - n_test
        if not isinstance(n_cal, Integral) or not 0 < n_cal <= n_train:
            raise ValueError(
                f"{name}: calibration size {n_cal} must be between 1 and the training "
                f"size {n_train} (total={n_total}, validation={n_val}, test={n_test})."
            )
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
        train = indices[:n_train]
        manifest["datasets"].append({
            "name": name, "root": dataset.data_root,
            "images": dataset.img_files, "masks": dataset.mask_files,
            "train": train, "validation": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:], "calibration": train[:int(n_cal)],
        })
    return manifest


def _validate_manifest(manifest, check_files=True):
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported split manifest; expected version 1.")
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("The manifest seed must be an integer in [0, 2**32).")
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The manifest must contain a nonempty datasets list.")
    names, roots = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each dataset manifest entry must be an object.")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("The manifest dataset names must be nonempty and unique.")
        names.add(name)
        root = entry.get("root")
        if not isinstance(root, str) or not root:
            raise ValueError(f"{name}: missing dataset root.")
        root = str(Path(root).expanduser().resolve())
        if root in roots:
            raise ValueError(f"Duplicate dataset root in manifest: {root}")
        roots.add(root)
        entry["root"] = root
        images, masks = entry.get("images"), entry.get("masks")
        if (not isinstance(images, list) or not isinstance(masks, list)
                or not images or len(images) != len(masks)
                or any(not isinstance(f, str) or Path(f).name != f for f in images + masks)
                or len(set(images)) != len(images) or len(set(masks)) != len(masks)):
            raise ValueError(f"{name}: invalid image/mask filename inventory.")
        if any(Path(i).stem != Path(m).stem for i, m in zip(images, masks)):
            raise ValueError(f"{name}: image/mask filename stems do not match.")
        n_total = len(images)
        split_sets = {}
        for key in ("train", "validation", "test", "calibration"):
            indices = entry.get(key)
            if (not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < n_total for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError(f"{name}: invalid or empty {key} indices.")
            split_sets[key] = set(indices)
        train, val, test = (split_sets[k] for k in ("train", "validation", "test"))
        if train & val or train & test or val & test:
            raise ValueError(f"{name}: train, validation, and test must be disjoint.")
        if train | val | test != set(range(n_total)):
            raise ValueError(f"{name}: train, validation, and test must cover every sample.")
        if not split_sets["calibration"] <= train:
            raise ValueError(f"{name}: calibration must be a subset of training.")
        if check_files:
            dataset = PolypDataset(root, bbox_shift=0)
            if dataset.img_files != images or dataset.mask_files != masks:
                raise ValueError(
                    f"{name}: dataset file inventory changed since the manifest was saved. "
                    "Restore the original files or explicitly create a new experiment split."
                )
    return manifest


def save_split_manifest(manifest, path):
    """Write a validated manifest as readable JSON, creating its parent directory."""
    validated = _validate_manifest(copy.deepcopy(manifest))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")


def load_split_manifest(path, data_roots=None, dataset_names=None):
    """Validate a JSON file or embedded manifest, optionally relocating its roots.

    Relocated roots must follow the recorded dataset order. Exact filename lists
    are checked; pixel content is not hashed. Dataset names, if supplied, must
    match the recorded order. Invalid or overlapping split indices are rejected.
    """
    if isinstance(path, dict):
        manifest = copy.deepcopy(path)
    else:
        manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Validate structure before consulting optional relocation paths.
    _validate_manifest(manifest, check_files=False)
    entries = manifest["datasets"]
    if dataset_names is not None and list(dataset_names) != [d["name"] for d in entries]:
        raise ValueError("Dataset names/order do not match the split manifest.")
    if data_roots is not None:
        data_roots = list(data_roots)
        if len(data_roots) != len(entries):
            raise ValueError("The number of relocated roots must match the manifest datasets.")
        for entry, root in zip(entries, data_roots):
            entry["root"] = str(Path(root).expanduser().resolve())
    return _validate_manifest(manifest)


def build_split_loaders(manifest, batch_size=4, eval_batch_size=1, num_workers=0,
                        image_size=1024, bbox_shift=5):
    """Build shared training, validation, test, and calibration loaders.

    Training prompts use stochastic expansion; validation/test prompts use fixed
    per-filename expansion; calibration prompts are tight boxes. Recovery
    sampling weights are proportional to mask boundary complexity.
    """
    from medcore_pruning.evaluation import compute_boundary_sampling_weights

    manifest = load_split_manifest(manifest)
    seed = manifest["seed"]
    train_subsets, val_subsets, test_subsets, cal_subsets = [], [], [], []
    for entry in manifest["datasets"]:
        train = PolypDataset(entry["root"], image_size=image_size, bbox_shift=bbox_shift)
        evaluation = PolypDataset(entry["root"], image_size=image_size,
                                  bbox_shift=bbox_shift, bbox_seed=seed)
        calibration = PolypDataset(entry["root"], image_size=image_size, bbox_shift=0)
        train_subsets.append(Subset(train, entry["train"]))
        val_subsets.append(Subset(evaluation, entry["validation"]))
        test_subsets.append(Subset(evaluation, entry["test"]))
        cal_subsets.append(Subset(calibration, entry["calibration"]))

    def make_loader(dataset, size, shuffle=False):
        return DataLoader(
            dataset, batch_size=size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    combined_cal = ConcatDataset(cal_subsets)
    combined_cal_loader = make_loader(combined_cal, 1)
    return {
        "train_loader": make_loader(ConcatDataset(train_subsets), batch_size, shuffle=True),
        "val_loaders": [make_loader(s, eval_batch_size) for s in val_subsets],
        "test_loaders": [make_loader(s, eval_batch_size) for s in test_subsets],
        "cal_loaders": [make_loader(s, 1) for s in cal_subsets],
        "combined_cal_loader": combined_cal_loader,
        "combined_cal": combined_cal,
        "sampling_weights": compute_boundary_sampling_weights(combined_cal_loader),
    }
