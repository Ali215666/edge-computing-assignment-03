"""Multiply-accumulate (MAC) counter for Conv2d / Linear layers via forward hooks."""
from __future__ import annotations

import torch
import torch.nn as nn


def count_macs(model: nn.Module, input_size=(1, 3, 32, 32)) -> int:
    total = 0

    def conv_hook(m, inp, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total += out.numel() // out.shape[0] * k

    def fc_hook(m, inp, out):
        nonlocal total
        total += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total
