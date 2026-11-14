"""Eager-mode INT8 quantization helpers (PTQ + QAT share these).

Backend selection
-----------------
PyTorch ships several quantized CPU kernels ("engines"): ``x86`` / ``fbgemm`` (Intel/AMD
server kernels), ``onednn`` and ``qnnpack`` (ARM/mobile). Which ones exist depends on how
the wheel was built: ``torch.backends.quantized.supported_engines`` tells. We pick the
first available in the order x86 > fbgemm > onednn > qnnpack.

QConfig
-------
Weights: symmetric per-channel INT8 (qint8). Activations: affine UINT8 with
``reduce_range=True`` (7-bit range). The reduced range avoids int16 saturation inside the
u8*s8 multiply-add instruction (``vpmaddubsw``) on x86 CPUs *without* AVX512-VNNI. This is
exactly fbgemm's default qconfig. It is backend independent (observers only), so we use it
for whichever engine is selected. ``--no-reduce-range`` switches to the full 8-bit range for
VNNI-capable CPUs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.ao.quantization import (QConfig, FakeQuantize, HistogramObserver,
                                   MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
                                   PerChannelMinMaxObserver, convert, prepare, prepare_qat)

from models.resnet import ResNet18

ENGINE_PREFERENCE = ("x86", "fbgemm", "onednn", "qnnpack")


def select_engine(requested: str = "auto") -> str:
    supported = torch.backends.quantized.supported_engines
    if requested != "auto":
        if requested not in supported:
            raise RuntimeError(f"quantized engine '{requested}' not available; supported: {supported}")
        engine = requested
    else:
        engine = next((e for e in ENGINE_PREFERENCE if e in supported), None)
        if engine is None:
            raise RuntimeError(f"no quantized CPU engine available (supported: {supported})")
    torch.backends.quantized.engine = engine
    return engine


def ptq_qconfig(reduce_range: bool = True) -> QConfig:
    return QConfig(
        activation=HistogramObserver.with_args(reduce_range=reduce_range),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )


def qat_qconfig(reduce_range: bool = True) -> QConfig:
    return QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                          quant_max=255, dtype=torch.quint8,
                                          qscheme=torch.per_tensor_affine, reduce_range=reduce_range),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128,
                                      quant_max=127, dtype=torch.qint8,
                                      qscheme=torch.per_channel_symmetric, ch_axis=0),
    )


def build_int8_skeleton(config: dict, reduce_range: bool = True) -> nn.Module:
    """Create a converted INT8 ResNet18 with the right module structure so that a saved
    INT8 ``state_dict`` (scales, zero points, packed weights) can be loaded into it.
    Valid for both PTQ and QAT checkpoints (both convert to the same quantized modules)."""
    m = ResNet18(config).eval()
    m.fuse_model(qat=False)
    m.qconfig = ptq_qconfig(reduce_range)
    prepare(m, inplace=True)
    with torch.no_grad():
        m(torch.randn(2, 3, 32, 32))  # give observers finite ranges; overwritten by load
    convert(m, inplace=True)
    return m


def prepare_qat_model(model: ResNet18, reduce_range: bool = True) -> ResNet18:
    model.train()
    model.fuse_model(qat=True)
    model.qconfig = qat_qconfig(reduce_range)
    prepare_qat(model, inplace=True)
    return model
