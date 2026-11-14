"""Inference-time preprocessing shared by the PyTorch and ONNX Runtime pipelines.

Any RGB image -> centre-crop to a square -> resize to 32x32 (bilinear) -> [0,1] ->
normalise with the CIFAR-10 mean/std -> NCHW float32. CIFAR-10 test samples are already
32x32 and only go through the last two steps, exactly like during training/evaluation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from core.data import CIFAR10_MEAN, CIFAR10_STD

_MEAN = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def load_image(path: str) -> np.ndarray:
    """Read an image file -> uint8 HWC RGB 32x32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((32, 32), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalize_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """uint8 HWC (or NHWC) -> normalised float32 NCHW."""
    if img.ndim == 3:
        img = img[None]
    x = img.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return ((x - _MEAN) / _STD).astype(np.float32)


def cifar_test_sample(data_dir: str, index: int) -> Tuple[np.ndarray, int]:
    """uint8 HWC image + label of CIFAR-10 test sample ``index``."""
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(data_dir, train=False, download=True)
    return ds.data[index], int(ds.targets[index])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
