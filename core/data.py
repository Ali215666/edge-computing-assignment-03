"""CIFAR-10 data pipeline.

The dataset is small (50k + 10k 32x32 images), so it is loaded into memory once as uint8
tensors and batched / augmented with vectorised tensor ops. This avoids DataLoader worker
processes (slow to spawn on Windows) and keeps CPU cores free for the model.

Splits (fixed by ``seed``):
  * train      : 45,000 images of the official training set (augmented)
  * val        :  5,000 held-out images of the official training set (no augmentation)
  * test       : the official 10,000-image test set (only used for final evaluation)
  * calibration: a random subset of the *train* split (no augmentation) for PTQ
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


def normalize_uint8(x: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW in [0,255] -> normalised float32 NCHW (the model's preprocessing)."""
    return (x.float().div_(255.0) - _MEAN) / _STD


def _random_crop_flip(x: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Standard CIFAR augmentation: 4-px zero-padded random 32x32 crop + horizontal flip."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    oh = torch.randint(0, 2 * pad + 1, (b,), generator=gen)
    ow = torch.randint(0, 2 * pad + 1, (b,), generator=gen)
    ih = (oh[:, None] + torch.arange(h)[None, :])  # (b, h)
    iw = (ow[:, None] + torch.arange(w)[None, :])  # (b, w)
    bi = torch.arange(b)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[bi, ih[:, :, None], iw[:, None, :]]  # (b, h, w, c)
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(b, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()


class TensorLoader:
    """Minimal in-memory batch iterator over uint8 images + int64 labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                 shuffle: bool = False, augment: bool = False, seed: int = 0,
                 channels_last: bool = False):
        self.images, self.labels = images, labels
        self.batch_size, self.shuffle, self.augment = batch_size, shuffle, augment
        self.channels_last = channels_last
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.labels) / self.batch_size)

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.labels)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            x = self.images[idx]
            if self.augment:
                x = _random_crop_flip(x, self.gen)
            x = normalize_uint8(x)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            yield x, self.labels[idx]


@dataclass
class CIFAR10Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def train_loader(self, batch_size: int, seed: int, channels_last: bool = False) -> TensorLoader:
        return TensorLoader(self.train_x, self.train_y, batch_size, shuffle=True, augment=True,
                            seed=seed, channels_last=channels_last)

    def val_loader(self, batch_size: int = 500) -> TensorLoader:
        return TensorLoader(self.val_x, self.val_y, batch_size)

    def test_loader(self, batch_size: int = 500, limit: Optional[int] = None) -> TensorLoader:
        x, y = (self.test_x, self.test_y) if limit is None else (self.test_x[:limit], self.test_y[:limit])
        return TensorLoader(x, y, batch_size)

    def calibration_loader(self, num_samples: int, batch_size: int, seed: int) -> TensorLoader:
        g = torch.Generator().manual_seed(seed + 1)
        idx = torch.randperm(len(self.train_y), generator=g)[:num_samples]
        return TensorLoader(self.train_x[idx], self.train_y[idx], batch_size)


def _to_tensors(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def load_cifar10(root: str = "data", val_size: int = 5000, seed: int = 42,
                 train_subset: Optional[int] = None, download: bool = True) -> CIFAR10Data:
    """Download (if needed) and load CIFAR-10 with a seeded train/val split.

    ``train_subset`` restricts the training split to N images (smoke tests only).
    """
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=download)
    te = CIFAR10(root, train=False, download=download)
    x, y = _to_tensors(tr)
    tx, ty = _to_tensors(te)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    if train_subset is not None:
        tr_idx = tr_idx[:train_subset]
    return CIFAR10Data(x[tr_idx], y[tr_idx], x[val_idx], y[val_idx], tx, ty)
