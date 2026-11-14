"""Export the five models to ONNX and validate them with ONNX Runtime (CPU).

Export strategy (decided per model, every attempt and error is recorded):
  * float models (fp32, unstructured, structured): the torch.export-based exporter
    (``dynamo=True``, opset 18), falling back to the TorchScript exporter if it fails.
  * INT8 eager models (ptq, qat): the dynamo exporter does not support eager-mode
    quantized modules (packed params / quantized tensors), so its error is recorded and the
    TorchScript exporter (``dynamo=False``, opset 13) is used. It lowers quantized::conv2d /
    add_relu / linear to QuantizeLinear -> DequantizeLinear (QDQ) + float ops, with the
    weights stored as INT8 initialisers. ONNX Runtime fuses QDQ groups into integer kernels.

Validation: onnx.checker, then ONNX Runtime CPU inference on ``--num-samples`` CIFAR-10 test
images. We compare logits / top-1 predictions with the PyTorch model on the same inputs.
Output: onnx/<model>.onnx, onnx/manifest.json, results/onnx_validation.csv

Usage:
    python scripts/export_onnx.py [--models fp32 ptq ...] [--num-samples 2000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from core.checkpoints import INT8_KINDS, load_checkpoint  # noqa: E402
from core.data import load_cifar10  # noqa: E402
from core.utils import DATA_DIR, DISPLAY, MODEL_NAMES, ONNX_DIR, RESULTS_DIR, quiet_warnings  # noqa: E402


def _export_dynamo(model, path: Path):
    batch = torch.export.Dim("batch", min=1, max=4096)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # exporter prints progress with emoji (cp1252 consoles)
        prog = torch.onnx.export(model, (torch.randn(2, 3, 32, 32),), dynamo=True, opset_version=18,
                                 input_names=["input"], output_names=["logits"],
                                 dynamic_shapes={"x": {0: batch}}, external_data=False, verbose=False)
        prog.save(str(path), external_data=False)


def _export_torchscript(model, path: Path, opset: int):
    torch.onnx.export(model, (torch.randn(1, 3, 32, 32),), str(path), dynamo=False, opset_version=opset,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def error_summary(e: BaseException) -> str:
    """One-line description of an export failure: outer exception plus its root cause."""
    root = e
    while root.__cause__ is not None:
        root = root.__cause__
    first = _ANSI.sub("", str(e)).strip().splitlines()
    msg = f"{type(e).__name__}: {first[0][:200] if first else ''}"
    if root is not e:
        cause = _ANSI.sub("", str(root)).strip().splitlines()
        msg += f" | root cause {type(root).__name__}: {cause[0][:300] if cause else ''}"
    return msg


def export_one(kind: str, model, path: Path) -> dict:
    attempts = []
    order = [("dynamo", None), ("torchscript", 13 if kind in INT8_KINDS else 17)]
    for method, opset in order:
        t0 = time.time()
        try:
            if method == "dynamo":
                _export_dynamo(model, path)
            else:
                _export_torchscript(model, path, opset)
            onnx.checker.check_model(str(path))
            attempts.append({"method": method, "ok": True, "time_s": round(time.time() - t0, 1)})
            m = onnx.load(str(path))
            return {"status": "exported", "method": method,
                    "opset": int(next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))),
                    "file": path.name, "size_mb": path.stat().st_size / 2**20,
                    "op_types": sorted({n.op_type for n in m.graph.node}), "attempts": attempts}
        except Exception as e:  # record, then try the next exporter
            msg = error_summary(e)
            attempts.append({"method": method, "ok": False, "error": msg})
            print(f"  [{kind}] {method} export failed: {msg}")
            if path.exists():
                path.unlink()
    return {"status": "failed", "attempts": attempts}


def validate(kind: str, model, path: Path, x: np.ndarray, y: np.ndarray, threads: int) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ort_logits, pt_logits = [], []
    with torch.no_grad():
        for i in range(0, len(x), 250):
            xb = x[i:i + 250]
            ort_logits.append(sess.run(None, {name: xb})[0])
            pt_logits.append(model(torch.from_numpy(xb)).numpy())
    o, p = np.concatenate(ort_logits), np.concatenate(pt_logits)
    diff = np.abs(o - p)
    return {"samples": len(x), "top1_agreement": float((o.argmax(1) == p.argmax(1)).mean()),
            "max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
            "ort_acc": float((o.argmax(1) == y).mean() * 100), "torch_acc": float((p.argmax(1) == y).mean() * 100),
            "allclose_1e-3": bool(np.allclose(o, p, atol=1e-3, rtol=1e-3))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--num-samples", type=int, default=2000, help="test images used for validation")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    quiet_warnings()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    data = load_cifar10(args.data_dir, seed=args.seed)
    from core.data import normalize_uint8
    n = min(args.num_samples, len(data.test_y))
    x = normalize_uint8(data.test_x[:n]).numpy()
    y = data.test_y[:n].numpy()

    manifest_path = ONNX_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rows = []
    for kind in args.models:
        print(f"[onnx] {DISPLAY[kind]}")
        model, ckpt = load_checkpoint(kind)
        model.eval()
        path = ONNX_DIR / f"{kind}.onnx"
        info = export_one(kind, model, path)
        info["source_checkpoint"] = kind
        if info["status"] == "exported":
            try:
                info["validation"] = validate(kind, model, path, x, y, args.threads)
                v = info["validation"]
                print(f"  [{kind}] {info['method']} opset {info['opset']} -> {path.name} ({info['size_mb']:.2f} MB); "
                      f"ORT top-1 agreement {v['top1_agreement'] * 100:.2f}%, max|dlogit| {v['max_abs_logit_diff']:.2e}, "
                      f"ORT acc {v['ort_acc']:.2f}% vs torch {v['torch_acc']:.2f}% on {v['samples']} images")
            except Exception as e:
                info["status"] = "exported_but_ort_failed"
                info["ort_error"] = f"{type(e).__name__}: {e}"
                print(f"  [{kind}] ONNX Runtime failed: {info['ort_error']}")
                traceback.print_exc()
        manifest[kind] = info
        v = info.get("validation", {})
        rows.append({"model": kind, "display": DISPLAY[kind], "status": info["status"],
                     "export_method": info.get("method", ""), "opset": info.get("opset", ""),
                     "onnx_file": info.get("file", ""), "onnx_size_mb": round(info.get("size_mb", float("nan")), 3),
                     "validation_samples": v.get("samples", ""),
                     "top1_agreement_pct": round(v["top1_agreement"] * 100, 3) if v else "",
                     "max_abs_logit_diff": v.get("max_abs_logit_diff", ""),
                     "mean_abs_logit_diff": v.get("mean_abs_logit_diff", ""),
                     "onnx_acc_subset": v.get("ort_acc", ""), "torch_acc_subset": v.get("torch_acc", ""),
                     "failed_attempts": " | ".join(f"{a['method']}: {a['error']}" for a in info["attempts"]
                                                   if not a["ok"])})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    out = RESULTS_DIR / "onnx_validation.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():  # keep rows of models not re-exported this time
        old = pd.read_csv(out)
        old = old[~old["model"].isin([r["model"] for r in rows])]
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df["_o"] = df["model"].map({k: i for i, k in enumerate(MODEL_NAMES)})
    df.sort_values("_o").drop(columns="_o").to_csv(out, index=False)
    print(f"[onnx] wrote {manifest_path} and {out}")


if __name__ == "__main__":
    main()
