"""Q2-B: Structured channel pruning with physical architecture reconstruction.

* importance criterion: L1 norm of each output filter (see core/structured.py for the
  exact residual-group aggregation)
* ``--internal-ratio``: fraction of block-internal channels (conv1 -> conv2) removed
* ``--stream-ratio``  : fraction of residual-stream channels removed, jointly across every
  layer that reads/writes the same residual tensor, so additions stay shape-valid
* the network is rebuilt as a smaller ResNet18 (new channel config) and the surviving
  weights are copied. No zero masks are involved.
* the reconstructed model is fine-tuned, evaluated and saved with its reduced config

Usage:
    python scripts/structured_pruning.py --internal-ratio 0.5 --stream-ratio 0.25 --epochs 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from core.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.engine import evaluate, finetune  # noqa: E402
from core.flops import count_macs  # noqa: E402
from core.history import reset_stage  # noqa: E402
from core.structured import build_pruned_model, compute_keep_indices, layer_shapes, verify_dimensions  # noqa: E402
from core.utils import CKPT, RESULTS_DIR, add_common_args, setup  # noqa: E402
from models.resnet import count_parameters  # noqa: E402


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--internal-ratio", type=float, default=0.5)
    p.add_argument("--stream-ratio", type=float, default=0.25)
    p.add_argument("--round-to", type=int, default=8, help="round kept channel counts up to a multiple")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output", default=str(CKPT["structured"]))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = setup(args)
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[structured] {out} exists - reusing it (pass --force to redo).")
        return
    data = load_cifar10(args.data_dir, seed=args.seed, train_subset=args.train_subset)
    test_loader = data.test_loader(limit=args.test_limit)
    fp32, base_ckpt = load_checkpoint("fp32")
    base = evaluate(fp32, test_loader)
    p0, macs0 = count_parameters(fp32), count_macs(fp32)
    print(f"[structured] baseline: test {base['acc']:.2f}%, {p0:,} params, {macs0 / 1e6:.1f} M MACs")

    keep = compute_keep_indices(fp32, args.internal_ratio, args.stream_ratio, args.round_to)
    model = build_pruned_model(fp32, keep)
    issues = verify_dimensions(model)
    if issues:
        raise RuntimeError(f"pruned model has inconsistent dimensions: {issues}")
    p1, macs1 = count_parameters(model), count_macs(model)
    print(f"[structured] rebuilt model: {p1:,} params ({p0 / p1:.2f}x fewer), "
          f"{macs1 / 1e6:.1f} M MACs ({macs0 / macs1:.2f}x fewer); dimension check passed")
    print(f"[structured] channels  stem {model.config['stem']} | blocks (in,mid,out): "
          + ", ".join(f"({b['in']},{b['mid']},{b['out']})" for b in model.config["blocks"]))
    pre = evaluate(model, test_loader)
    print(f"[structured] test acc right after pruning (no fine-tuning): {pre['acc']:.2f}%")

    hist = RESULTS_DIR / "training_history.csv"
    reset_stage(hist, "structured")
    t0 = time.time()
    model = model.to(device)
    finetune(model, data, args, device, args.epochs, args.lr, "structured", hist)
    model = model.cpu().eval()
    val = evaluate(model, data.val_loader())
    test = evaluate(model, test_loader)
    assert not verify_dimensions(model)
    print(f"[structured] after fine-tuning: val {val['acc']:.2f}% | test {test['acc']:.2f}% "
          f"(loss {test['loss']:.4f}) | drop vs FP32 {base['acc'] - test['acc']:+.2f} pts")
    meta = {"test_acc": test["acc"], "test_loss": test["loss"], "val_acc": val["acc"],
            "test_acc_before_finetune": pre["acc"], "fp32_test_acc": base["acc"],
            "internal_ratio": args.internal_ratio, "stream_ratio": args.stream_ratio,
            "round_to": args.round_to, "params": p1, "fp32_params": p0, "param_compression": p0 / p1,
            "macs": macs1, "fp32_macs": macs0, "mac_reduction": macs0 / macs1,
            "layer_shapes": layer_shapes(model),
            "kept_channels": {"stream": [len(i) for i in keep["stream"]],
                              "internal": [len(i) for i in keep["internal"]]},
            "criterion": "L1 norm of output filters; residual groups: per-layer-normalised sum",
            "epochs": args.epochs, "lr": args.lr, "time_s": time.time() - t0, "seed": args.seed,
            "train_samples": len(data.train_y),
            "smoke_test": args.test_limit is not None
            or base_ckpt["meta"].get("smoke_test", False)}
    save_checkpoint(out, "structured", model, model.config, meta)
    print(f"[structured] saved {out}")


if __name__ == "__main__":
    main()
