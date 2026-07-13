"""
src/transforms.py
Image transforms for Telugu HTR.

All images are:
  • Converted to grayscale
  • Resized to a fixed height (H=64), width kept proportional
  • Width-padded (white background) to max_width (512)
  • Normalised to [0, 1] then standardised with ImageNet-compatible stats

Augmentation (training only):
  • Small random rotation  (±5°)
  • Random perspective distortion
  • Random brightness / contrast jitter
  • Optional elastic distortion (via grid_distort)
"""

from __future__ import annotations
import random
import math
from typing import Tuple, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_HEIGHT   = 64          # fixed height for all images
MAX_WIDTH       = 512         # maximum width; wider images are clipped/padded
MEAN            = [0.5]       # grayscale single-channel
STD             = [0.5]       # maps [0,1] → [-1, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_keep_aspect(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    """Resize PIL image to target_h, keep aspect ratio, crop if wider than max_w."""
    w, h = img.size
    scale = target_h / h
    new_w = int(w * scale)
    resized = img.resize((new_w, target_h), Image.BICUBIC)
    if new_w > max_w:
        return resized.crop((0, 0, max_w, target_h))
    return resized


def _pad_to_width(img: Image.Image, target_w: int, fill: int = 255) -> Image.Image:
    """Pad image on the right to target_w with fill colour (white by default)."""
    w, h = img.size
    if w >= target_w:
        return img.crop((0, 0, target_w, h))    # crop if accidentally wider
    padded = Image.new(img.mode, (target_w, h), fill)
    padded.paste(img, (0, 0))
    return padded


def _elastic_distort(img: Image.Image, alpha: float = 10.0, sigma: float = 3.0) -> Image.Image:
    """
    Lightweight elastic distortion using a random displacement grid.
    alpha  – magnitude of displacement (pixels)
    sigma  – smoothing radius
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random displacement fields
    dx = (np.random.rand(h, w) * 2 - 1) * alpha
    dy = (np.random.rand(h, w) * 2 - 1) * alpha

    # Smooth with a simple box filter
    from scipy.ndimage import gaussian_filter   # lazy import; only used if called
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    # Build mapping
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    import cv2   # lazy import
    distorted = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
    return Image.fromarray(distorted.astype(np.uint8))


def _morphological_ops(img: Image.Image) -> Image.Image:
    """Randomly apply erosion or dilation to simulate varying pen thickness.
    Assuming dark text on light background:
    MinFilter (dilation of dark strokes) -> thicker pen
    MaxFilter (erosion of dark strokes) -> thinner pen
    """
    op = random.choice(["thicken", "thin", "none"])
    if op == "thicken":
        # MinFilter makes dark pixels expand (thickens strokes)
        return img.filter(ImageFilter.MinFilter(3))
    elif op == "thin":
        # MaxFilter makes light pixels expand (thins strokes)
        return img.filter(ImageFilter.MaxFilter(3))
    return img


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class TrainTransform:
    """
    Augmentation + normalisation pipeline for training images.

    Returns a tuple of (float tensor [1, H, W] in [-1, 1], scaled_width).
    """

    def __init__(
        self,
        target_h: int  = TARGET_HEIGHT,
        max_w: int     = MAX_WIDTH,
        rotation: float = 5.0,      # ± degrees
        use_elastic: bool = False,  # requires scipy + cv2; off by default
    ):
        self.target_h    = target_h
        self.max_w       = max_w
        self.rotation    = rotation
        self.use_elastic = use_elastic

        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Grayscale
        img = ImageOps.grayscale(img)

        # 2. Random elastic distort (optional)
        if self.use_elastic and random.random() < 0.3:
            try:
                img = _elastic_distort(img)
            except ImportError:
                pass  # scipy / cv2 not installed — skip silently

        # 2b. Random stroke thickness variation
        if random.random() < 0.3:
            img = _morphological_ops(img)

        # 3. Random rotation
        if self.rotation > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation, self.rotation)
            img = TF.rotate(img, angle, fill=255)

        # 4. Random perspective
        if random.random() < 0.3:
            w, h = img.size
            startpoints = [
                [0,     0    ],
                [w - 1, 0    ],
                [w - 1, h - 1],
                [0,     h - 1],
            ]
            endpoints = self._rand_perspective_pts(img)
            img = TF.perspective(
                img,
                startpoints=startpoints,
                endpoints=endpoints,
                fill=255,
            )

        # 5. Resize + pad
        img = _resize_keep_aspect(img, self.target_h, self.max_w)
        scaled_w = img.size[0]  # width before padding
        img = _pad_to_width(img, self.max_w, fill=255)

        # 6. Brightness / contrast jitter (applied on PIL then converted)
        img = img.convert("RGB")   # ColorJitter needs 3-ch
        img = self.jitter(img)
        img = ImageOps.grayscale(img)

        # 7. To tensor [1, H, W], then normalize
        tensor = TF.to_tensor(img)          # [1, H, W] in [0, 1]
        tensor = self.normalize(tensor)     # → [-1, 1]
        return tensor, scaled_w

    @staticmethod
    def _rand_perspective_pts(img: Image.Image, jitter: float = 0.05):
        """Return four jittered corners for use as perspective endpoints."""
        w, h = img.size
        d = int(min(w, h) * jitter)

        def rnd(lo, hi):
            return random.randint(lo, hi) if lo < hi else lo

        return [
            [rnd(0, d),       rnd(0, d)],
            [rnd(w-d-1, w-1), rnd(0, d)],
            [rnd(w-d-1, w-1), rnd(h-d-1, h-1)],
            [rnd(0, d),       rnd(h-d-1, h-1)],
        ]


class ValTransform:
    """
    Deterministic resize + pad + normalise for validation / test / inference.

    Returns a tuple of (float tensor [1, H, W] in [-1, 1], scaled_width).
    """

    def __init__(self, target_h: int = TARGET_HEIGHT, max_w: int = MAX_WIDTH):
        self.target_h  = target_h
        self.max_w     = max_w
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, int]:
        img = ImageOps.grayscale(img)
        img = _resize_keep_aspect(img, self.target_h, self.max_w)
        scaled_w = img.size[0]  # width before padding
        img = _pad_to_width(img, self.max_w, fill=255)
        tensor = TF.to_tensor(img)       # [1, H, W] in [0, 1]
        tensor = self.normalize(tensor)  # → [-1, 1]
        return tensor, scaled_w


# ---------------------------------------------------------------------------
# Collate helper (used in DataLoader)
# ---------------------------------------------------------------------------

def collate_fn_pad(batch):
    """
    Pads a batch of (image_tensor, original_width, label_ids, label_len) tuples.
    Images are already fixed-width from the transform, so only labels need padding.
    Returns:
        images   : [B, 1, H, W]
        labels   : [B, T_max]  (0-padded)
        lengths  : [B]         (original label lengths, LongTensor)
        widths   : [B]         (scaled image widths before padding, LongTensor)
    """
    images, widths, labels, lengths = zip(*batch)
    images  = torch.stack(images, dim=0)             # [B, 1, H, W]
    max_len = max(len(l) for l in labels)
    padded  = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, lab in enumerate(labels):
        t = torch.tensor(lab, dtype=torch.long)
        padded[i, :len(lab)] = t
    lengths = torch.tensor(lengths, dtype=torch.long)
    widths  = torch.tensor(widths, dtype=torch.long)
    return images, padded, lengths, widths
