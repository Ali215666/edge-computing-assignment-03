"""ONNX Runtime (CPU) inference for the exported models - no PyTorch needed at runtime
except for reading CIFAR-10 test samples via torchvision.

    python scripts/onnx_inference.py --model all --index 3
    python scripts/onnx_inference.py --model ptq --image my_dog.png

Preprocessing (core/preprocess.py, numpy only): RGB -> centre crop -> 32x32 bilinear ->
/255 -> (x - mean) / std with the CIFAR-10 statistics -> NCHW float32.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from core.data import CLASSES  # noqa: E402
from core.preprocess import cifar_test_sample, load_image, normalize_hwc_uint8, softmax  # noqa: E402
from core.utils import DATA_DIR, DISPLAY, MODEL_NAMES, ONNX_DIR  # noqa: E402


def make_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    p.add_argument("--onnx", default=None, help="explicit .onnx path (single model)")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--index", type=int, action="append", default=[])
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args(argv)

    imgs, labels, names = [], [], []
    for path in args.image:
        imgs.append(load_image(path)); labels.append(None); names.append(path)
    for i in args.index or ([] if args.image else [0]):
        img, y = cifar_test_sample(args.data_dir, i)
        imgs.append(img); labels.append(y); names.append(f"cifar10-test[{i}]")
    x = normalize_hwc_uint8(np.stack(imgs))

    manifest = json.loads((ONNX_DIR / "manifest.json").read_text()) if (ONNX_DIR / "manifest.json").exists() else {}
    kinds = MODEL_NAMES if args.model == "all" else [args.model]
    results = []
    for kind in kinds:
        path = Path(args.onnx) if (args.onnx and len(kinds) == 1) else ONNX_DIR / f"{kind}.onnx"
        if not path.exists():
            reason = manifest.get(kind, {}).get("attempts", "not exported")
            print(f"\n== {DISPLAY[kind]}: no ONNX file ({path}); export info: {reason}")
            continue
        sess = make_session(path)
        t0 = time.perf_counter()
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        dt = (time.perf_counter() - t0) * 1000
        prob = softmax(logits)
        print(f"\n== {DISPLAY[kind]} [ONNX Runtime CPU, {path.name}] - {dt:.2f} ms for {len(x)} image(s)")
        for name, y, pr in zip(names, labels, prob):
            c = int(pr.argmax())
            truth = "" if y is None else f" | true: {CLASSES[y]} {'OK' if y == c else 'WRONG'}"
            print(f"  {name}: {CLASSES[c]} (confidence {pr[c]:.4f}){truth}")
            results.append({"model": kind, "input": name, "pred": c, "confidence": float(pr[c]), "label": y})
    return results


if __name__ == "__main__":
    main()
