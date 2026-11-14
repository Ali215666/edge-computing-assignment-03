"""Training / evaluation loops shared by all scripts."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device = torch.device("cpu"),
             channels_last: bool = False) -> Dict[str, float]:
    """Top-1 accuracy (%) and mean cross-entropy loss over ``loader``."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return {"acc": 100.0 * correct / total, "loss": loss_sum / total}


def train_one_epoch(model: nn.Module, loader, optimizer, device, scheduler=None,
                    step_scheduler_per_batch: bool = False, after_step: Optional[Callable] = None,
                    log_every: int = 50, label: str = "") -> Dict[str, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    n = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if after_step is not None:
            after_step()
        if scheduler is not None and step_scheduler_per_batch:
            scheduler.step()
        loss_sum += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  {label} [{i + 1}/{n}] loss {loss_sum / total:.4f} acc {100 * correct / total:.2f}% "
                  f"({el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s)", flush=True)
    if scheduler is not None and not step_scheduler_per_batch:
        scheduler.step()
    return {"acc": 100.0 * correct / total, "loss": loss_sum / total, "time_s": time.time() - t0}


def append_history(path: Path, rows: List[Dict]):
    """Append rows to a CSV (header written when the file is new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    keys = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        w.writerows(rows)


def finetune(model: nn.Module, data, args, device, epochs: int, lr: float, stage: str,
             history_path: Path, after_step: Optional[Callable] = None,
             epoch_callback: Optional[Callable[[int], None]] = None,
             weight_decay: float = 5e-4, start_epoch_index: int = 0,
             eval_fn: Optional[Callable] = None) -> List[Dict]:
    """SGD + cosine fine-tuning used by pruning / QAT. Logs one row per epoch."""
    if epochs <= 0:
        return []
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    loader = data.train_loader(args.batch_size, seed=args.seed + 1000 + start_epoch_index)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    rows = []
    for ep in range(epochs):
        if epoch_callback is not None:
            epoch_callback(ep)
        tr = train_one_epoch(model, loader, opt, device, sched, step_scheduler_per_batch=True,
                             after_step=after_step, label=f"{stage} ep{ep + 1}/{epochs}")
        va = (eval_fn or evaluate)(model, data.val_loader(), device)
        row = {"stage": stage, "epoch": start_epoch_index + ep + 1, "lr_start": lr,
               "train_loss": round(tr["loss"], 5), "train_acc": round(tr["acc"], 3),
               "val_loss": round(va["loss"], 5), "val_acc": round(va["acc"], 3),
               "epoch_time_s": round(tr["time_s"], 1)}
        print(f"[{stage}] epoch {ep + 1}/{epochs}: train {tr['acc']:.2f}% val {va['acc']:.2f}% "
              f"({tr['time_s']:.0f}s)", flush=True)
        append_history(history_path, [row])
        rows.append(row)
    return rows
