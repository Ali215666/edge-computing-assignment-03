"""Q1-A: Post-Training Static Quantization (eager mode, INT8).

Steps
  1. load the trained FP32 baseline
  2. fuse Conv+BN+ReLU / Conv+BN (shortcut) modules. The classifier Linear is not followed by
     ReLU/BN, so it has nothing to fuse with and is quantized on its own.
  3. select the quantized backend (x86 > fbgemm > onednn > qnnpack, whichever this build has)
  4. attach the qconfig (per-channel symmetric INT8 weights, histogram-calibrated UINT8
     activations) and insert observers with ``prepare``
  5. calibrate: forward ``--calib-samples`` training images (no augmentation, no labels used)
  6. ``convert`` -> real INT8 kernels (quantized::conv2d_relu, quantized::add_relu, ...)
  7. evaluate on the test set and save models/checkpoints/ptq_int8.pth

Usage:
    python scripts/ptq.py --calib-samples 2000
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.ao.quantization import convert, prepare  # noqa: E402

from core.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.engine import evaluate  # noqa: E402
from core.quant import ptq_qconfig, select_engine  # noqa: E402
from core.utils import CKPT, add_common_args, setup  # noqa: E402
from models.resnet import count_parameters  # noqa: E402


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--calib-samples", type=int, default=2000, help="training images used for calibration")
    p.add_argument("--calib-batch-size", type=int, default=100)
    p.add_argument("--engine", default="auto", help="auto | x86 | fbgemm | onednn | qnnpack")
    p.add_argument("--no-reduce-range", action="store_true",
                   help="use the full 8-bit activation range (only safe on VNNI CPUs)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--output", default=str(CKPT["ptq"]))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup(args)
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[ptq] {out} exists - reusing it (pass --force to redo).")
        return
    reduce_range = not args.no_reduce_range
    engine = select_engine(args.engine)
    print(f"[ptq] quantized engine: {engine} (available: {torch.backends.quantized.supported_engines}), "
          f"reduce_range={reduce_range}")

    data = load_cifar10(args.data_dir, seed=args.seed)
    test_loader = data.test_loader(limit=args.test_limit)
    fp32, base_ckpt = load_checkpoint("fp32")
    fp32_test = evaluate(fp32, test_loader)
    print(f"[ptq] FP32 baseline test acc {fp32_test['acc']:.2f}%")

    model = copy.deepcopy(fp32).cpu().eval()
    model.fuse_model(qat=False)
    fused_test = evaluate(model, test_loader)
    print(f"[ptq] after Conv-BN(-ReLU) fusion (still FP32): {fused_test['acc']:.2f}%")

    model.qconfig = ptq_qconfig(reduce_range)
    prepare(model, inplace=True)
    t0 = time.time()
    calib = data.calibration_loader(args.calib_samples, args.calib_batch_size, args.seed)
    with torch.no_grad():
        for x, _ in calib:
            model(x)
    calib_time = time.time() - t0
    print(f"[ptq] calibrated on {calib.num_samples} training images in {calib_time:.1f}s")
    convert(model, inplace=True)
    model.eval()

    val = evaluate(model, data.val_loader())
    test = evaluate(model, test_loader)
    print(f"[ptq] INT8 val {val['acc']:.2f}% | test {test['acc']:.2f}% (loss {test['loss']:.4f}) | "
          f"drop vs FP32 {fp32_test['acc'] - test['acc']:+.2f} pts")
    meta = {"test_acc": test["acc"], "test_loss": test["loss"], "val_acc": val["acc"],
            "fp32_test_acc": fp32_test["acc"], "fused_fp32_test_acc": fused_test["acc"],
            "calib_samples": calib.num_samples, "calib_time_s": calib_time,
            "activation_observer": "HistogramObserver (per-tensor affine, quint8)",
            "weight_observer": "PerChannelMinMaxObserver (per-channel symmetric, qint8)",
            "params": count_parameters(model), "seed": args.seed,
            "smoke_test": bool(args.test_limit) or base_ckpt["meta"].get("smoke_test", False)}
    save_checkpoint(out, "ptq", model, base_ckpt["config"], meta, quant_engine=engine,
                    reduce_range=reduce_range)
    print(f"[ptq] saved {out}")


if __name__ == "__main__":
    main()
