"""Saving / loading of the five model checkpoints.

Every checkpoint is a dict::

    {"kind": "fp32" | "ptq" | "qat" | "unstructured" | "structured",
     "config": <ResNet18 channel config>,        # structured -> the *reduced* widths
     "state_dict": ...,                           # INT8 kinds -> quantized state_dict
     "quant_engine": "onednn" | ... (INT8 only),
     "reduce_range": bool (INT8 only),
     "meta": {...}}                               # metrics, hyper-parameters, provenance
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from models.resnet import ResNet18
from core.quant import build_int8_skeleton, select_engine
from core.utils import CKPT

INT8_KINDS = ("ptq", "qat")


def save_checkpoint(path: Path, kind: str, model: nn.Module, config: dict, meta: dict, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "config": config, "state_dict": model.state_dict(), "meta": meta}
    payload.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leave a half-written checkpoint


def load_checkpoint(kind_or_path, device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, dict]:
    """Return (model in eval mode, checkpoint dict). INT8 models are always on CPU."""
    path = Path(CKPT[kind_or_path]) if kind_or_path in CKPT else Path(kind_or_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kind = ckpt["kind"]
    if kind in INT8_KINDS:
        engine = select_engine(ckpt.get("quant_engine", "auto")
                               if ckpt.get("quant_engine") in torch.backends.quantized.supported_engines
                               else "auto")
        model = build_int8_skeleton(ckpt["config"], ckpt.get("reduce_range", True))
        model.load_state_dict(ckpt["state_dict"])
        ckpt["runtime_engine"] = engine
        return model.eval(), ckpt
    model = ResNet18(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def model_device(kind: str, device: torch.device) -> torch.device:
    """Quantized eager models only have CPU kernels."""
    return torch.device("cpu") if kind in INT8_KINDS else device
