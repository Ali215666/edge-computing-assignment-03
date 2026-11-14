"""ResNet-18 adapted for CIFAR-10 (32x32 inputs, 10 classes).

Differences from the ImageNet ResNet-18:
  * 3x3 stride-1 stem convolution, no max-pool (keeps spatial resolution for 32x32 inputs)
  * 10-way classifier

The network is written to be *quantization friendly* (eager-mode PyTorch quantization):
  * QuantStub / DeQuantStub at the input / output (identity in float mode)
  * every ReLU is a distinct module so Conv-BN-ReLU can be fused
  * the residual addition uses ``FloatFunctional.add_relu`` so it becomes a
    quantized ``add_relu`` after conversion

It is also *width configurable*: every block takes its own (in, mid, out) channel
counts. Structured channel pruning uses this to rebuild a physically smaller network
(see ``scripts/structured_pruning.py``). With the default config the model is the
standard CIFAR ResNet-18 (11,173,962 parameters).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

NUM_CLASSES = 10


def default_config(num_classes: int = NUM_CLASSES) -> Dict:
    """Channel configuration of the standard CIFAR ResNet-18."""
    blocks = []
    in_planes = 64
    for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        for i in range(2):
            s = stride if i == 0 else 1
            blocks.append({"in": in_planes, "mid": planes, "out": planes, "stride": s})
            in_planes = planes
    return {"stem": 64, "blocks": blocks, "num_classes": num_classes}


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, mid_planes: int, out_planes: int, stride: int = 1,
                 force_shortcut: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.has_shortcut = force_shortcut or stride != 1 or in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.shortcut = nn.Identity()
        self.skip_add = FloatFunctional()  # quantizable residual add (+ fused ReLU)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = copy.deepcopy(config or default_config())
        cfg = self.config
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, cfg["stem"], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg["stem"])
        self.relu = nn.ReLU(inplace=False)
        blocks: List[nn.Module] = []
        for b in cfg["blocks"]:
            # A block that had a projection shortcut in the original net keeps it, even if
            # pruning makes in == out, so the residual stream semantics are unchanged.
            blocks.append(BasicBlock(b["in"], b["mid"], b["out"], b["stride"],
                                     force_shortcut=b.get("shortcut", False)))
        # group into 4 stages of 2 blocks (names layer1..layer4 like torchvision)
        self.layer1 = nn.Sequential(*blocks[0:2])
        self.layer2 = nn.Sequential(*blocks[2:4])
        self.layer3 = nn.Sequential(*blocks[4:6])
        self.layer4 = nn.Sequential(*blocks[6:8])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(cfg["blocks"][-1]["out"], cfg["num_classes"])
        self.dequant = DeQuantStub()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def blocks(self) -> List[BasicBlock]:
        return [b for layer in (self.layer1, self.layer2, self.layer3, self.layer4) for b in layer]

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.flatten(self.avgpool(x))
        x = self.fc(x)
        return self.dequant(x)

    # ------------------------------------------------------------------ quantization
    def _fusion_groups(self) -> List[List[str]]:
        groups = [["conv1", "bn1", "relu"]]
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            for i, blk in enumerate(getattr(self, lname)):
                p = f"{lname}.{i}"
                groups.append([f"{p}.conv1", f"{p}.bn1", f"{p}.relu1"])
                groups.append([f"{p}.conv2", f"{p}.bn2"])
                if blk.has_shortcut:
                    groups.append([f"{p}.shortcut.0", f"{p}.shortcut.1"])
        return groups

    def fuse_model(self, qat: bool = False) -> "ResNet18":
        """Fuse Conv+BN(+ReLU) in place. ``qat=True`` keeps BN for training (ConvBnReLU2d)."""
        fn = fuse_modules_qat if qat else fuse_modules
        fn(self, self._fusion_groups(), inplace=True)
        return self


def count_parameters(model: nn.Module) -> int:
    """Number of parameters. Works for float models and converted INT8 models
    (whose packed weights are not nn.Parameters)."""
    total = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        # quantized Conv/Linear expose weight()/bias() as methods over packed params
        if callable(getattr(m, "weight", None)) and not isinstance(getattr(m, "weight"), torch.Tensor):
            total += m.weight().numel()
            b = m.bias() if callable(getattr(m, "bias", None)) else None
            if b is not None:
                total += b.numel()
    return total


def is_quantized(model: nn.Module) -> bool:
    return any(callable(getattr(m, "weight", None)) and not isinstance(getattr(m, "weight"), torch.Tensor)
               for m in model.modules())
