"""Q2-A: Iterative unstructured L1-magnitude pruning.

Criterion: the |w| of every individual weight in *all* Conv2d and Linear layers, ranked
globally (``prune.global_unstructured`` with ``L1Unstructured``), so layers with many
redundant weights lose more than sensitive ones (e.g. the 3-channel stem conv).

Schedule: ``--iterations`` rounds reach ``--target-sparsity`` geometrically. Each round
removes the same fraction ``a = 1 - (1 - S)^(1/k)`` of the *still-remaining* weights
(PyTorch's PruningContainer applies the amount to unpruned entries only), followed by
``--finetune-epochs`` of fine-tuning. Masks stay active during fine-tuning
(``weight = weight_orig * weight_mask`` via a forward pre-hook), so pruned weights stay at
zero. After the final fine-tuning, ``prune.remove`` makes the pruning permanent: the
mask/orig re-parametrisation is folded into a plain ``weight`` tensor with exact zeros.

Note: unstructured sparsity keeps the dense tensor shapes. The saved file is therefore
the same size as FP32 unless it is compressed (see benchmark.py, gzip column) or run on
a sparse-aware runtime. This is a property of the method, not a bug.

Usage:
    python scripts/unstructured_pruning.py --target-sparsity 0.8 --iterations 3
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.utils.prune as prune  # noqa: E402

from core.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.engine import evaluate, finetune  # noqa: E402
from core.history import reset_stage  # noqa: E402
from core.utils import CKPT, RESULTS_DIR, add_common_args, setup  # noqa: E402
from models.resnet import count_parameters  # noqa: E402


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--target-sparsity", type=float, default=0.8,
                   help="final fraction of zero weights over all conv + fc weights")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=1, help="fine-tuning epochs after each round")
    p.add_argument("--final-epochs", type=int, default=1, help="extra fine-tuning after the last round")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output", default=str(CKPT["unstructured"]))
    return p.parse_args(argv)


def prunable(model: nn.Module):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_report(model: nn.Module):
    """Per-layer and global sparsity of conv/fc weights (works before and after prune.remove)."""
    layers, zeros, total = [], 0, 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight  # masked weight while pruning is active
            z, n = int((w == 0).sum()), w.numel()
            layers.append({"layer": name, "zeros": z, "numel": n, "sparsity": z / n})
            zeros += z
            total += n
    all_params = sum(p.numel() for p in model.parameters())
    return {"global_sparsity": zeros / total, "zero_weights": zeros, "prunable_weights": total,
            "layers": layers, "all_params": all_params}


def main(argv=None):
    args = parse_args(argv)
    device = setup(args)
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[unstructured] {out} exists - reusing it (pass --force to redo).")
        return
    data = load_cifar10(args.data_dir, seed=args.seed, train_subset=args.train_subset)
    test_loader = data.test_loader(limit=args.test_limit)
    fp32, base_ckpt = load_checkpoint("fp32", device)
    base = evaluate(fp32, test_loader, device)
    print(f"[unstructured] baseline test acc {base['acc']:.2f}%")

    model = copy.deepcopy(fp32)
    params = prunable(model)
    k, S = args.iterations, args.target_sparsity
    per_iter = 1.0 - (1.0 - S) ** (1.0 / k)
    hist = RESULTS_DIR / "training_history.csv"
    reset_stage(hist, "unstructured")
    log = []
    t0 = time.time()
    for it in range(1, k + 1):
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=per_iter)
        sp = sparsity_report(model)["global_sparsity"]
        acc_pruned = evaluate(model, data.val_loader(), device)["acc"]
        print(f"[unstructured] round {it}/{k}: pruned {per_iter:.3f} of remaining -> sparsity {sp:.4f}, "
              f"val acc before fine-tune {acc_pruned:.2f}%")
        epochs = args.finetune_epochs + (args.final_epochs if it == k else 0)
        rows = finetune(model, data, args, device, epochs, args.lr, f"unstructured_r{it}", hist)
        acc_ft = rows[-1]["val_acc"] if rows else acc_pruned
        log.append({"round": it, "sparsity": sp, "val_acc_after_prune": acc_pruned, "val_acc_after_ft": acc_ft})

    # make pruning permanent: weight_orig * weight_mask -> weight, hooks + buffers removed
    for m, name in params:
        prune.remove(m, name)
    assert not any(n.endswith("_mask") or n.endswith("_orig") for n, _ in model.named_parameters())
    assert not any(n.endswith("_mask") for n, _ in model.named_buffers())

    rep = sparsity_report(model)
    model = model.cpu().eval()
    val = evaluate(model, data.val_loader())
    test = evaluate(model, test_loader)
    print(f"[unstructured] final global sparsity {rep['global_sparsity']:.4f} "
          f"({rep['zero_weights']:,}/{rep['prunable_weights']:,} conv+fc weights are zero)")
    print(f"[unstructured] val {val['acc']:.2f}% | test {test['acc']:.2f}% (loss {test['loss']:.4f}) | "
          f"drop vs FP32 {base['acc'] - test['acc']:+.2f} pts")
    meta = {"test_acc": test["acc"], "test_loss": test["loss"], "val_acc": val["acc"],
            "fp32_test_acc": base["acc"], "target_sparsity": S, "iterations": k,
            "amount_per_iteration": per_iter, "global_sparsity": rep["global_sparsity"],
            "zero_weights": rep["zero_weights"], "prunable_weights": rep["prunable_weights"],
            "layer_sparsity": {l["layer"]: round(l["sparsity"], 4) for l in rep["layers"]},
            "rounds": log, "finetune_epochs": args.finetune_epochs, "final_epochs": args.final_epochs,
            "lr": args.lr, "time_s": time.time() - t0, "params": count_parameters(model),
            "nonzero_params": int(sum((p != 0).sum() for p in model.parameters())), "seed": args.seed,
            "train_samples": len(data.train_y),
            "smoke_test": args.test_limit is not None
            or base_ckpt["meta"].get("smoke_test", False)}
    save_checkpoint(out, "unstructured", model, base_ckpt["config"], meta)
    print(f"[unstructured] saved {out}")


if __name__ == "__main__":
    main()
