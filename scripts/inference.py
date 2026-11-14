"""Unified PyTorch inference for the five models.

    python scripts/inference.py --model fp32 --index 7              # CIFAR-10 test sample #7
    python scripts/inference.py --model all --image path/to/cat.jpg  # any RGB image
    python scripts/inference.py --model ptq --index 0 --index 1 --index 2

Models: fp32 | ptq | qat | unstructured | structured | all
INT8 models (ptq, qat) always run on CPU with the quantized engine stored in the checkpoint
(or the best available one). Float models use --device (auto = CUDA if available).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.checkpoints import load_checkpoint, model_device  # noqa: E402
from core.data import CLASSES  # noqa: E402
from core.preprocess import cifar_test_sample, load_image, normalize_hwc_uint8  # noqa: E402
from core.utils import DATA_DIR, DISPLAY, MODEL_NAMES, get_device, quiet_warnings  # noqa: E402


def predict(model, x: torch.Tensor, device: torch.device, topk: int = 3):
    with torch.no_grad():
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        prob = torch.softmax(logits.float(), 1).cpu()
    conf, idx = prob.topk(topk, 1)
    return [{"pred": int(idx[i, 0]), "class": CLASSES[int(idx[i, 0])], "confidence": float(conf[i, 0]),
             "top": [(CLASSES[int(j)], float(c)) for j, c in zip(idx[i], conf[i])]}
            for i in range(x.shape[0])], dt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    p.add_argument("--checkpoint", default=None, help="override the checkpoint path (single model)")
    p.add_argument("--image", action="append", default=[], help="image file(s)")
    p.add_argument("--index", type=int, action="append", default=[], help="CIFAR-10 test index(es)")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    quiet_warnings()

    inputs, labels, names = [], [], []
    for path in args.image:
        inputs.append(load_image(path)); labels.append(None); names.append(path)
    for i in args.index or ([] if args.image else [0]):
        img, y = cifar_test_sample(args.data_dir, i)
        inputs.append(img); labels.append(y); names.append(f"cifar10-test[{i}]")
    x = torch.from_numpy(normalize_hwc_uint8(np.stack(inputs)))

    kinds = MODEL_NAMES if args.model == "all" else [args.model]
    req_device = get_device(args.device)
    results = []
    for kind in kinds:
        model, ckpt = load_checkpoint(args.checkpoint if (args.checkpoint and len(kinds) == 1) else kind,
                                      model_device(kind, req_device))
        dev = model_device(ckpt["kind"], req_device)
        preds, dt = predict(model, x, dev)
        engine = f", engine={ckpt['runtime_engine']}" if "runtime_engine" in ckpt else ""
        print(f"\n== {DISPLAY[ckpt['kind']]} ({dev}{engine}) - {dt:.2f} ms for {len(preds)} image(s)")
        for name, y, pr in zip(names, labels, preds):
            truth = "" if y is None else f" | true: {CLASSES[y]} {'OK' if y == pr['pred'] else 'WRONG'}"
            top = ", ".join(f"{c} {p:.3f}" for c, p in pr["top"])
            print(f"  {name}: {pr['class']} (confidence {pr['confidence']:.4f}){truth}   top-3: {top}")
            results.append({"model": kind, "input": name, **pr, "label": y})
    return results


if __name__ == "__main__":
    main()
