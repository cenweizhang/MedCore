"""Paired binary segmentation images, masks, and box prompts for MedSAM."""

import hashlib
import random
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from skimage.transform import resize
import torch
from torch.utils.data import Dataset


def _sort_key(filename):
    stem = Path(filename).stem
    return (0, int(stem), stem) if stem.isdecimal() else (1, 0, stem)


def _index_files(directory):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    files = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in files:
                raise ValueError(f"Duplicate filename stem in {directory}: {path.stem}")
            files[path.stem] = path.name
    return files


class PolypDataset(Dataset):
    """Read matching filename stems from images/ and masks/.

    Images become three-channel, min-max normalized square inputs. Binary
    masks may use 0/1 or 0/255. bbox_shift=0 gives tight calibration boxes;
    bbox_seed makes expanded evaluation boxes deterministic per filename.
    Without a box seed, expansion follows the seeded Python random generator.
    """

    def __init__(self, data_root, image_size=1024, bbox_shift=5, bbox_seed=None):
        if image_size <= 0 or bbox_shift < 0:
            raise ValueError("image_size must be positive and bbox_shift nonnegative.")
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.bbox_seed = bbox_seed
        self.img_dir = Path(self.data_root) / "images"
        self.mask_dir = Path(self.data_root) / "masks"
        images, masks = _index_files(self.img_dir), _index_files(self.mask_dir)
        if not images:
            raise ValueError(f"No images found in {self.img_dir}")
        if images.keys() != masks.keys():
            raise ValueError(
                f"Image/mask filename stems differ in {self.data_root}: "
                f"missing masks={sorted(images.keys() - masks.keys())[:5]}, "
                f"missing images={sorted(masks.keys() - images.keys())[:5]}"
            )
        self.img_files = sorted(images.values(), key=_sort_key)
        self.mask_files = [masks[Path(name).stem] for name in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        name = self.img_files[index]
        image = iio.imread(self.img_dir / name)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError(f"Expected a finite grayscale or RGB image: {name}")
        original_size = tuple(image.shape[:2])
        image = resize(image, (self.image_size, self.image_size),
                       order=3, preserve_range=True, anti_aliasing=True)
        image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)

        mask = iio.imread(self.mask_dir / self.mask_files[index])
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.ndim != 2 or mask.shape != original_size or not np.isfinite(mask).all():
            raise ValueError(f"Mask must be finite and match the image dimensions: {name}")
        threshold = 0.5 if mask.max() <= 1 else 127
        mask = (mask > threshold).astype(np.uint8)

        def resize_mask(size):
            return resize(mask, (size, size), order=0,
                          preserve_range=True, anti_aliasing=False).astype(np.uint8)

        mask_256, mask_input = resize_mask(256), resize_mask(self.image_size)
        ys, xs = np.where(mask_input)
        if len(ys):
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            rng = random
            if self.bbox_seed is not None:
                digest = hashlib.sha256(f"{self.bbox_seed}:{name}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "big"))
            if self.bbox_shift:
                x_min = max(0, x_min - rng.randint(0, self.bbox_shift))
                x_max = min(self.image_size, x_max + rng.randint(0, self.bbox_shift))
                y_min = max(0, y_min - rng.randint(0, self.bbox_shift))
                y_max = min(self.image_size, y_max + rng.randint(0, self.bbox_shift))
            box = [x_min, y_min, x_max, y_max]
        else:
            box = [0, 0, self.image_size, self.image_size]
        return {
            "image": torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1),
            "mask_256": torch.from_numpy(mask_256.astype(np.int64)).unsqueeze(0),
            "mask_1024": torch.from_numpy(mask_input.astype(np.int64)),
            "bbox": torch.tensor(box, dtype=torch.float32),
            "name": name,
            "original_size": original_size,
        }
