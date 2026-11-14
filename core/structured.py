"""Structured (channel) pruning for the CIFAR ResNet-18: physically removes channels.

Two kinds of channels exist in a ResNet:

1. **Block-internal channels** (output of ``conv1`` = input of ``conv2`` inside a
   BasicBlock). Only those two convs (and ``bn1``) touch them, so they can be pruned
   independently per block.

2. **Residual-stream channels.** Inside a stage every block output is *added* to its
   input, so all tensors on the stage's residual path must keep the same channel
   indexing. One "channel group" per stage:

       stage 1 : stem conv1/bn1, layer1.0.conv2/bn2, layer1.1.conv2/bn2   (identity shortcuts)
       stage k : layerk.0.conv2/bn2, layerk.0.shortcut conv/bn, layerk.1.conv2/bn2

   A residual channel can only be removed from *all* of its producers at once, and then
   also from every consumer (input channels of the next convs / shortcut convs / the fc).

Importance criterion (documented, L1-norm based)
------------------------------------------------
* internal channel c of a block:   ||W_conv1[c]||_1  (L1 norm of the producing filter)
* residual channel c of a group:   sum over producers p of ||W_p[c]||_1 / mean_c'(||W_p[c']||_1)
  (per-layer normalisation so each producer contributes on the same scale)

The lowest-importance channels are dropped. Kept counts are rounded up to a multiple of
``round_to`` (8 by default) for SIMD-friendly shapes. The pruned network is rebuilt with
``ResNet18(new_config)`` and the surviving weight slices are copied in. No masks are used:
the resulting tensors are genuinely smaller.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.resnet import ResNet18


def _l1_filters(conv: nn.Conv2d) -> torch.Tensor:
    return conv.weight.detach().abs().flatten(1).sum(1)


def _keep_count(n: int, ratio: float, round_to: int) -> int:
    k = max(1, int(math.ceil(n * (1.0 - ratio))))
    if round_to > 1:
        k = min(n, int(math.ceil(k / round_to) * round_to))
    return k


def _topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(torch.topk(scores, k).indices).values  # keep original channel order


def stage_groups(model: ResNet18) -> List[Dict]:
    """Producers (conv, bn) of each residual-stream channel group."""
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    groups = []
    for si, stage in enumerate(stages):
        producers: List[Tuple[nn.Conv2d, nn.BatchNorm2d]] = []
        if si == 0:
            producers.append((model.conv1, model.bn1))
        for blk in stage:
            producers.append((blk.conv2, blk.bn2))
            if blk.has_shortcut:
                producers.append((blk.shortcut[0], blk.shortcut[1]))
        groups.append({"stage": si, "producers": producers})
    return groups


def compute_keep_indices(model: ResNet18, internal_ratio: float, stream_ratio: float,
                         round_to: int = 8) -> Dict:
    """Decide which channels survive. Returns {'stream': [idx per stage], 'internal': [idx per block]}."""
    stream = []
    for g in stage_groups(model):
        score = None
        for conv, _ in g["producers"]:
            s = _l1_filters(conv)
            s = s / s.mean().clamp_min(1e-12)
            score = s if score is None else score + s
        k = _keep_count(score.numel(), stream_ratio, round_to)
        stream.append(_topk(score, k))
    internal = []
    for blk in model.blocks():
        s = _l1_filters(blk.conv1)
        internal.append(_topk(s, _keep_count(s.numel(), internal_ratio, round_to)))
    return {"stream": stream, "internal": internal}


def _copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d, idx: torch.Tensor):
    dst.weight.data.copy_(src.weight.data[idx])
    dst.bias.data.copy_(src.bias.data[idx])
    dst.running_mean.copy_(src.running_mean[idx])
    dst.running_var.copy_(src.running_var[idx])
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


def build_pruned_model(model: ResNet18, keep: Dict) -> ResNet18:
    """Create the physically smaller ResNet18 and copy the kept weight slices."""
    model = model.cpu().eval()
    old_blocks = model.blocks()
    stage_of_block = [0, 0, 1, 1, 2, 2, 3, 3]
    cfg = copy.deepcopy(model.config)
    cfg["stem"] = len(keep["stream"][0])
    for i, (b, ob) in enumerate(zip(cfg["blocks"], old_blocks)):
        s = stage_of_block[i]
        in_idx = keep["stream"][s] if (i % 2 == 1 or s == 0) else keep["stream"][s - 1]
        b["in"] = len(in_idx)
        b["mid"] = len(keep["internal"][i])
        b["out"] = len(keep["stream"][s])
        b["shortcut"] = ob.has_shortcut
    new = ResNet18(cfg).eval()

    with torch.no_grad():
        k0 = keep["stream"][0]
        new.conv1.weight.copy_(model.conv1.weight[k0])
        _copy_bn(new.bn1, model.bn1, k0)
        for i, (nb, ob) in enumerate(zip(new.blocks(), old_blocks)):
            s = stage_of_block[i]
            in_idx = keep["stream"][s] if (i % 2 == 1 or s == 0) else keep["stream"][s - 1]
            mid_idx = keep["internal"][i]
            out_idx = keep["stream"][s]
            nb.conv1.weight.copy_(ob.conv1.weight[mid_idx][:, in_idx])
            _copy_bn(nb.bn1, ob.bn1, mid_idx)
            nb.conv2.weight.copy_(ob.conv2.weight[out_idx][:, mid_idx])
            _copy_bn(nb.bn2, ob.bn2, out_idx)
            if ob.has_shortcut:
                nb.shortcut[0].weight.copy_(ob.shortcut[0].weight[out_idx][:, in_idx])
                _copy_bn(nb.shortcut[1], ob.shortcut[1], out_idx)
        new.fc.weight.copy_(model.fc.weight[:, keep["stream"][3]])
        new.fc.bias.copy_(model.fc.bias)
    return new


def verify_dimensions(model: ResNet18) -> List[str]:
    """Check that every connection is shape-consistent and run a forward pass."""
    issues = []
    prev = model.conv1.out_channels
    if model.bn1.num_features != prev:
        issues.append("stem bn mismatch")
    for i, b in enumerate(model.blocks()):
        if b.conv1.in_channels != prev:
            issues.append(f"block{i}.conv1 in={b.conv1.in_channels} but stream={prev}")
        if b.conv2.in_channels != b.conv1.out_channels or b.bn1.num_features != b.conv1.out_channels:
            issues.append(f"block{i} internal mismatch")
        if b.bn2.num_features != b.conv2.out_channels:
            issues.append(f"block{i}.bn2 mismatch")
        if b.has_shortcut:
            sc = b.shortcut[0]
            if sc.in_channels != prev or sc.out_channels != b.conv2.out_channels:
                issues.append(f"block{i} shortcut mismatch")
        elif prev != b.conv2.out_channels:
            issues.append(f"block{i} identity shortcut but {prev} != {b.conv2.out_channels}")
        prev = b.conv2.out_channels
    if model.fc.in_features != prev:
        issues.append(f"fc in={model.fc.in_features} but stream={prev}")
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 3, 32, 32))
    if tuple(out.shape) != (2, model.fc.out_features):
        issues.append(f"bad output shape {tuple(out.shape)}")
    return issues


def layer_shapes(model: nn.Module) -> List[Dict]:
    return [{"layer": n, "type": type(m).__name__, "shape": list(m.weight.shape)}
            for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
