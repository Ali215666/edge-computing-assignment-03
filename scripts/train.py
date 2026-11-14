"""Train the FP32 ResNet-18 baseline on CIFAR-10.

* SGD (momentum 0.9, nesterov, weight decay 5e-4) + OneCycle LR schedule (per batch)
* standard augmentation (random crop with 4px padding + horizontal flip)
* the epoch with the best *validation* accuracy is kept; test accuracy is computed once,
  at the end, on that checkpoint
* resumable: full training state is written every epoch to models/checkpoints/train_state.pth

Usage:
    python scripts/train.py --epochs 15
    python scripts/train.py --resume            # continue an interrupted run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from core.checkpoints import save_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.engine import append_history, evaluate, train_one_epoch  # noqa: E402
from core.history import reset_stage  # noqa: E402
from core.utils import CKPT, CKPT_DIR, RESULTS_DIR, add_common_args, setup  # noqa: E402
from models.resnet import ResNet18, count_parameters, default_config  # noqa: E402

STATE = CKPT_DIR / "train_state.pth"
HISTORY = RESULTS_DIR / "training_history.csv"


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate of the OneCycle schedule")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--resume", action="store_true", help="resume from train_state.pth")
    p.add_argument("--force", action="store_true", help="retrain even if baseline_fp32.pth exists")
    p.add_argument("--output", default=str(CKPT["fp32"]))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = setup(args)
    out = Path(args.output)
    if out.exists() and not (args.force or args.resume):
        print(f"[train] {out} exists - reusing it (pass --force to retrain).")
        return
    channels_last = device.type == "cpu"  # ~15% faster convolutions on CPU
    data = load_cifar10(args.data_dir, seed=args.seed, train_subset=args.train_subset)
    print(f"[train] device={device} train={len(data.train_y)} val={len(data.val_y)} test={len(data.test_y)}")

    model = ResNet18(default_config()).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(data.train_y) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.25)
    start_epoch, best = 0, {"val_acc": -1.0, "epoch": 0, "state_dict": None}

    if args.resume and STATE.exists():
        st = torch.load(STATE, map_location="cpu", weights_only=False)
        if st["args"]["epochs"] != args.epochs:
            raise SystemExit(f"resume: saved run used --epochs {st['args']['epochs']}; pass the same value")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_epoch, best = st["epoch"], st["best"]
        print(f"[train] resumed from epoch {start_epoch} (best val {best['val_acc']:.2f}%)")
    else:
        reset_stage(HISTORY, "baseline")

    for ep in range(start_epoch, args.epochs):
        loader = data.train_loader(args.batch_size, seed=args.seed + ep, channels_last=channels_last)
        tr = train_one_epoch(model, loader, opt, device, sched, step_scheduler_per_batch=True,
                             label=f"baseline ep{ep + 1}/{args.epochs}")
        va = evaluate(model, data.val_loader(), device, channels_last)
        row = {"stage": "baseline", "epoch": ep + 1, "lr_start": round(opt.param_groups[0]["lr"], 6),
               "train_loss": round(tr["loss"], 5), "train_acc": round(tr["acc"], 3),
               "val_loss": round(va["loss"], 5), "val_acc": round(va["acc"], 3),
               "epoch_time_s": round(tr["time_s"], 1)}
        append_history(HISTORY, [row])
        print(f"[train] epoch {ep + 1}/{args.epochs} train {tr['acc']:.2f}% val {va['acc']:.2f}% "
              f"loss {va['loss']:.4f} ({tr['time_s']:.0f}s)", flush=True)
        if va["acc"] > best["val_acc"]:
            best = {"val_acc": va["acc"], "epoch": ep + 1,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": ep + 1, "best": best,
                    "args": vars(args)}, STATE)

    model.load_state_dict(best["state_dict"])
    model = model.to(memory_format=torch.contiguous_format)
    te = evaluate(model, data.test_loader(limit=args.test_limit), device)
    meta = {"val_acc": best["val_acc"], "best_epoch": best["epoch"], "test_acc": te["acc"],
            "test_loss": te["loss"], "params": count_parameters(model), "epochs": args.epochs,
            "lr": args.lr, "batch_size": args.batch_size, "weight_decay": args.weight_decay,
            "train_samples": len(data.train_y), "seed": args.seed,
            "smoke_test": args.train_subset is not None or args.test_limit is not None,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(out, "fp32", model.cpu(), model.config, meta)
    print(f"[train] saved {out}: best val {best['val_acc']:.2f}% (epoch {best['epoch']}), "
          f"test {te['acc']:.2f}% / loss {te['loss']:.4f}")


if __name__ == "__main__":
    main()
