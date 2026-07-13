"""
src/dataset.py
PyTorch Dataset for IIIT-HW-Telugu.

Annotation file format (one per split):
    <image_filename>  <ground_truth_label>

e.g.
    train/word_00001.png  కాలం
    train/word_00002.png  పూజ

Images live at:  <image_root>/<image_filename>
"""

from __future__ import annotations
import os
import unicodedata
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.vocab import TeluguVocab, DEFAULT_VOCAB
from src.transforms import TrainTransform, ValTransform, collate_fn_pad


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TeluguHTRDataset(Dataset):
    """
    Word-level handwritten Telugu dataset.

    Parameters
    ----------
    annotation_file : str
        Path to the split's annotation .txt file.
    image_root : str
        Root directory that contains the images (images paths in the
        annotation file are relative to this root).
    vocab : TeluguVocab
        Character vocabulary.
    transform : callable
        Image transform (TrainTransform or ValTransform).
    max_label_len : int
        Labels longer than this are skipped during loading.
    add_sos_eos : bool
        If True, prepend SOS and append EOS to every label sequence.
        Required for autoregressive training; not needed for CTC.
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        vocab: TeluguVocab,
        transform: Callable,
        max_label_len: int = 32,
        add_sos_eos: bool = False,
    ):
        self.image_root    = image_root
        self.vocab         = vocab
        self.transform     = transform
        self.max_label_len = max_label_len
        self.add_sos_eos   = add_sos_eos

        self.samples: List[Tuple[str, str]] = []   # (image_path, label)
        self._load_annotations(annotation_file)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_annotations(self, annotation_file: str):
        skipped = 0
        with open(annotation_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                img_rel, label = parts
                label = unicodedata.normalize("NFC", label.strip())
                if not label or len(label) > self.max_label_len:
                    skipped += 1
                    continue
                img_path = os.path.join(self.image_root, img_rel)
                self.samples.append((img_path, label))

        print(
            f"[Dataset] Loaded {len(self.samples)} samples "
            f"(skipped {skipped}) from {annotation_file}"
        )

        # Validate that all image files exist (check first 50)
        missing = []
        for img_path, label in self.samples[:min(50, len(self.samples))]:
            if not os.path.exists(img_path):
                missing.append(img_path)
        if missing:
            raise FileNotFoundError(
                f"[Dataset] {len(missing)} image files not found (showing first 5):\n"
                + "\n".join(missing[:5])
                + f"\nCheck that image_root='{self.image_root}' matches your labels.txt format."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image (no silent fallback — crash on missing files)
        img = Image.open(img_path).convert("RGB")

        # Apply transform → (tensor [1, H, W], scaled_width)
        img_tensor, img_width = self.transform(img)

        # Encode label
        label_ids = self.vocab.encode(label)
        if self.add_sos_eos:
            label_ids = [self.vocab.sos_id] + label_ids + [self.vocab.eos_id]

        label_len = len(label_ids)
        return img_tensor, img_width, label_ids, label_len

    # ------------------------------------------------------------------
    # Extra utilities
    # ------------------------------------------------------------------

    def get_label(self, idx: int) -> str:
        return self.samples[idx][1]

    def sample_batch(self, n: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility: fetch a small batch for debugging."""
        indices = list(range(min(n, len(self))))
        items   = [self[i] for i in indices]
        return collate_fn_pad(items)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    annotation_file: str,
    image_root: str,
    vocab: TeluguVocab,
    split: str = "train",                  # "train" | "val" | "test"
    batch_size: int = 64,
    num_workers: int = 4,
    max_label_len: int = 32,
    add_sos_eos: bool = False,
    use_elastic: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Parameters
    ----------
    split : 'train' uses TrainTransform (with augmentation),
            'val' or 'test' uses ValTransform (deterministic).
    add_sos_eos : set True for autoregressive models, False for CTC.
    """
    if split == "train":
        transform = TrainTransform(use_elastic=use_elastic)
        shuffle   = True
    else:
        transform = ValTransform()
        shuffle   = False

    dataset = TeluguHTRDataset(
        annotation_file = annotation_file,
        image_root      = image_root,
        vocab           = vocab,
        transform       = transform,
        max_label_len   = max_label_len,
        add_sos_eos     = add_sos_eos,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn_pad,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )
    return loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.dataset <annotation_file> <image_root>")
        sys.exit(1)

    ann_file   = sys.argv[1]
    image_root = sys.argv[2]

    vocab  = DEFAULT_VOCAB
    loader = build_dataloader(
        ann_file, image_root, vocab,
        split       = "train",
        batch_size  = 4,
        num_workers = 0,
        add_sos_eos = True,
    )

    images, labels, lengths, widths = next(iter(loader))
    print(f"Batch images shape : {images.shape}")    # [4, 1, 64, 512]
    print(f"Batch labels shape : {labels.shape}")    # [4, T_max]
    print(f"Label lengths      : {lengths}")
    print(f"Image widths       : {widths}")
    for i in range(len(lengths)):
        decoded = vocab.decode(labels[i].tolist())
        print(f"  Sample {i}: '{decoded}'  (len={lengths[i]}, w={widths[i]})")
