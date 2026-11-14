"""Q1-B: Quantization-Aware Training (eager mode, INT8).

Steps
  1. initialise from the trained FP32 baseline
  2. fuse for QAT (``fuse_modules_qat``: Conv+BN+ReLU -> ConvBnReLU2d keeps BN trainable)
  3. attach the QAT qconfig: FakeQuantize modules with moving-average min/max observers
     (per-channel symmetric INT8 weights, per-tensor affine UINT8 activations)
  4. fine-tune with fake quantization in the forward pass (straight-through estimator)
       * observers are frozen after ``--freeze-observer-frac`` of the steps
       * BatchNorm running statistics are frozen after ``--freeze-bn-frac`` of the steps
  5. ``convert`` -> deployable INT8 model, evaluate, save models/checkpoints/qat_int8.pth
     (the fake-quant model is also saved as qat_fakequant.pth, used for the ONNX QDQ export)

Usage:
    python scripts/qat.py --epochs 2 --lr 1e-3
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.ao.nn.intrinsic.qat as nniqat  # noqa: E402
from torch.ao.quantization import convert, disable_observer, enable_observer  # noqa: E402

from core.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.engine import evaluate, finetune  # noqa: E402
from core.history import reset_stage  # noqa: E402
from core.quant import prepare_qat_model, select_engine  # noqa: E402
from core.utils import CKPT, CKPT_DIR, RESULTS_DIR, add_common_args, setup  # noqa: E402
from models.resnet import count_parameters  # noqa: E402

FAKEQUANT_CKPT = CKPT_DIR / "qat_fakequant.pth"


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze-observer-frac", type=float, default=0.7,
                   help="fraction of QAT steps after which observers (scales/zero-points) are frozen")
    p.add_argument("--freeze-bn-frac", type=float, default=0.5,
                   help="fraction of QAT steps after which BN running stats are frozen")
    p.add_argument("--engine", default="auto")
    p.add_argument("--no-reduce-range", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--output", default=str(CKPT["qat"]))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = setup(args)
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[qat] {out} exists - reusing it (pass --force to redo).")
        return
    reduce_range = not args.no_reduce_range
    engine = select_engine(args.engine)
    print(f"[qat] quantized engine: {engine}, reduce_range={reduce_range}, training device: {device}")

    data = load_cifar10(args.data_dir, seed=args.seed, train_subset=args.train_subset)
    test_loader = data.test_loader(limit=args.test_limit)
    fp32, base_ckpt = load_checkpoint("fp32")
    fp32_test = evaluate(fp32, test_loader)
    print(f"[qat] FP32 baseline test acc {fp32_test['acc']:.2f}%")

    model = prepare_qat_model(copy.deepcopy(fp32).cpu(), reduce_range).to(device)
    # accuracy of the fake-quantized model before any QAT training (observers see one pass first)
    with torch.no_grad():
        model.train()
        for x, _ in data.calibration_loader(1000, 100, args.seed):
            model(x.to(device))
    model.apply(disable_observer)
    pre = evaluate(model, data.val_loader(), device)
    model.apply(enable_observer)
    print(f"[qat] fake-quant val acc before QAT fine-tuning: {pre['acc']:.2f}%")

    steps_total = args.epochs * -(-len(data.train_y) // args.batch_size)
    state = {"step": 0, "obs_frozen": False, "bn_frozen": False}

    def after_step():
        state["step"] += 1
        if not state["bn_frozen"] and state["step"] >= args.freeze_bn_frac * steps_total:
            model.apply(nniqat.freeze_bn_stats)
            state["bn_frozen"] = True
            print(f"  [qat] BN statistics frozen at step {state['step']}", flush=True)
        if not state["obs_frozen"] and state["step"] >= args.freeze_observer_frac * steps_total:
            model.apply(disable_observer)
            state["obs_frozen"] = True
            print(f"  [qat] observers frozen at step {state['step']}", flush=True)

    def eval_no_observer(m, loader, dev):
        # keep validation data from updating the quantization ranges
        m.apply(disable_observer)
        r = evaluate(m, loader, dev)
        if not state["obs_frozen"]:
            m.apply(enable_observer)
        return r

    reset_stage(RESULTS_DIR / "training_history.csv", "qat")
    t0 = time.time()
    finetune(model, data, args, device, args.epochs, args.lr, "qat", RESULTS_DIR / "training_history.csv",
             after_step=after_step, eval_fn=eval_no_observer, weight_decay=1e-4)
    train_time = time.time() - t0

    model = model.cpu().eval()
    model.apply(disable_observer)
    fq_test = evaluate(model, test_loader)
    torch.save({"config": base_ckpt["config"], "state_dict": model.state_dict(),
                "reduce_range": reduce_range, "meta": {"fakequant_test_acc": fq_test["acc"]}},
               FAKEQUANT_CKPT)

    int8 = convert(copy.deepcopy(model), inplace=False).eval()
    val = evaluate(int8, data.val_loader())
    test = evaluate(int8, test_loader)
    print(f"[qat] fake-quant test {fq_test['acc']:.2f}% | INT8 val {val['acc']:.2f}% | INT8 test "
          f"{test['acc']:.2f}% (loss {test['loss']:.4f}) | drop vs FP32 {fp32_test['acc'] - test['acc']:+.2f} pts")
    meta = {"test_acc": test["acc"], "test_loss": test["loss"], "val_acc": val["acc"],
            "fp32_test_acc": fp32_test["acc"], "fakequant_test_acc": fq_test["acc"],
            "pre_qat_fakequant_val_acc": pre["acc"], "epochs": args.epochs, "lr": args.lr,
            "freeze_bn_frac": args.freeze_bn_frac, "freeze_observer_frac": args.freeze_observer_frac,
            "train_time_s": train_time, "params": count_parameters(int8), "seed": args.seed,
            "train_samples": len(data.train_y),
            "smoke_test": args.test_limit is not None
            or base_ckpt["meta"].get("smoke_test", False)}
    save_checkpoint(out, "qat", int8, base_ckpt["config"], meta, quant_engine=engine, reduce_range=reduce_range)
    print(f"[qat] saved {out} (+ fake-quant model {FAKEQUANT_CKPT.name})")


if __name__ == "__main__":
    main()
