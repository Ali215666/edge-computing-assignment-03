"""Benchmark all five models on the full CIFAR-10 test set (PyTorch and ONNX Runtime).

Identical conditions for every model: the same 10,000 pre-normalised test tensors, the
same CPU thread count (torch.set_num_threads / ORT intra_op_num_threads), the same batch
size, warm-up runs and timing method (time.perf_counter, CUDA synchronize when on GPU).
Model loading, ORT session creation and preprocessing are excluded from all timings.

Measured per model
  * accuracy, cross-entropy loss (batched pass over the whole test set)
  * throughput (images/s) and mean batch latency: ``--repeats`` timed passes over the full
    test set at ``--batch-size``; the median pass is reported
  * per-image latency: batch size 1, one timed forward per image for ``--latency-samples``
    images (mean / median / p95 / p99)
  * serialized model size (MB) and gzip-compressed size, parameter count, non-zero
    parameters, conv/fc weight sparsity, compression ratio vs FP32, MACs

Outputs (results/):
  benchmark_summary.csv       PyTorch: one row per model
  predictions.csv             PyTorch: per test image & model -> label, prediction,
                              confidence, batch-1 inference time
  onnx_benchmark_summary.csv  ONNX Runtime: one row per model
  onnx_predictions.csv        ONNX Runtime: per-image predictions
  model_comparison.csv        side-by-side PyTorch vs ONNX Runtime summary
  benchmark_env.json          hardware / software / settings

Usage:
    python scripts/benchmark.py                       # both backends, all models
    python scripts/benchmark.py --backend pytorch --latency-samples 1000
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from core.checkpoints import INT8_KINDS, load_checkpoint, model_device  # noqa: E402
from core.data import load_cifar10, normalize_uint8  # noqa: E402
from core.flops import count_macs  # noqa: E402
from core.utils import (DATA_DIR, DISPLAY, MODEL_NAMES, ONNX_DIR, RESULTS_DIR, get_device,  # noqa: E402
                        quiet_warnings)
from models.resnet import ResNet18, count_parameters  # noqa: E402


# ----------------------------------------------------------------------------- runners
class TorchRunner:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model, self.device = model.eval(), device

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x).to(self.device))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return out.float().cpu().numpy()


class OrtRunner:
    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.name: x})[0]


# ----------------------------------------------------------------------------- measurements
def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def batched_pass(run, x: np.ndarray, y: np.ndarray, bs: int, warmup: int, repeats: int):
    for i in range(warmup):
        run(x[(i * bs) % len(x):(i * bs) % len(x) + bs])
    times, logits = [], None
    for r in range(repeats):
        outs = []
        t0 = time.perf_counter()
        for i in range(0, len(x), bs):
            outs.append(run(x[i:i + bs]))
        times.append(time.perf_counter() - t0)
        if logits is None:
            logits = np.concatenate(outs)
    t = float(np.median(times))
    nb = -(-len(x) // bs)
    lsm = _log_softmax(logits.astype(np.float64))
    prob = np.exp(lsm)
    return {
        "logits": logits, "pred": logits.argmax(1), "conf": prob.max(1),
        "acc": float((logits.argmax(1) == y).mean() * 100),
        "loss": float(-lsm[np.arange(len(y)), y].mean()),
        "pass_time_s": t, "pass_times_s": times, "throughput_ips": len(x) / t,
        "batch_latency_ms": t / nb * 1000, "amortized_ms_per_image": t / len(x) * 1000,
    }


def single_image_pass(run, x: np.ndarray, n: int, warmup: int):
    for i in range(warmup):
        run(x[i:i + 1])
    times, preds = np.empty(n), np.empty(n, dtype=np.int64)
    for i in range(n):
        xi = x[i:i + 1]
        t0 = time.perf_counter()
        out = run(xi)
        times[i] = (time.perf_counter() - t0) * 1000
        preds[i] = out.argmax(1)[0]
    return times, preds


# ----------------------------------------------------------------------------- model stats
def state_dict_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def weight_stats(model: nn.Module):
    """(non-zero params, zeros in conv/fc weights, total conv/fc weights)."""
    nonzero, zw, tw = 0, 0, 0
    for p in model.parameters():
        nonzero += int((p != 0).sum())
    for m in model.modules():
        w = None
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
        elif callable(getattr(m, "weight", None)) and not isinstance(getattr(m, "weight"), torch.Tensor):
            w = m.weight().dequantize()  # packed INT8 conv/linear
            nonzero += int((w != 0).sum())
            b = m.bias()
            if b is not None:
                nonzero += int((b != 0).sum())
        if w is not None:
            zw += int((w == 0).sum())
            tw += w.numel()
    return nonzero, zw, tw


def macs_of(kind: str, ckpt: dict) -> int:
    # INT8 models have the FP32 graph; count on a float twin with the same config
    return count_macs(ResNet18(ckpt["config"]))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--backend", default="both", choices=["pytorch", "onnx", "both"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10, help="warm-up iterations (batched and single-image)")
    p.add_argument("--repeats", type=int, default=3, help="timed passes over the test set (median used)")
    p.add_argument("--latency-samples", type=int, default=10000, help="images timed one-by-one (batch 1)")
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    p.add_argument("--device", default="cpu", help="device for float PyTorch models (INT8 is CPU-only)")
    p.add_argument("--test-limit", type=int, default=None, help="smoke tests only")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    quiet_warnings()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cifar10(args.data_dir, seed=args.seed)
    xt, yt = data.test_x, data.test_y
    if args.test_limit:
        xt, yt = xt[:args.test_limit], yt[:args.test_limit]
    x = normalize_uint8(xt).numpy()  # preprocessing done once, outside all timings
    y = yt.numpy()
    n_lat = min(args.latency_samples, len(x))
    device = get_device(args.device)
    backends = ["pytorch", "onnx"] if args.backend == "both" else [args.backend]
    env = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
           "processor": platform.processor(), "threads": args.threads, "batch_size": args.batch_size,
           "warmup": args.warmup, "repeats": args.repeats, "latency_samples": n_lat,
           "test_images": len(x), "quantized_engine": torch.backends.quantized.engine,
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import onnxruntime as ort
        env["onnxruntime"] = ort.__version__
    except ImportError:
        pass

    for backend in backends:
        summary, preds_frames = [], []
        fp32_size = None
        manifest = {}
        if backend == "onnx":
            mp = ONNX_DIR / "manifest.json"
            manifest = json.loads(mp.read_text()) if mp.exists() else {}
        for kind in args.models:
            t_load = time.perf_counter()
            model, ckpt = load_checkpoint(kind, model_device(kind, device))
            dev = model_device(kind, device)
            if backend == "pytorch":
                run = TorchRunner(model, dev)
                blob = state_dict_bytes(model)
                size_b = len(blob)
                where = f"{dev}" + (f"/{ckpt['runtime_engine']}" if "runtime_engine" in ckpt else "")
            else:
                path = ONNX_DIR / f"{kind}.onnx"
                if not path.exists() or manifest.get(kind, {}).get("status") != "exported":
                    print(f"[bench/onnx] {kind}: no validated ONNX export - skipped "
                          f"(see onnx/manifest.json: {manifest.get(kind, {}).get('attempts', 'not exported')})")
                    summary.append({"model": kind, "display": DISPLAY[kind], "backend": "onnxruntime",
                                    "status": "not exported"})
                    continue
                run = OrtRunner(path, args.threads)
                blob = path.read_bytes()
                size_b = len(blob)
                where = "cpu/onnxruntime"
            load_s = time.perf_counter() - t_load
            print(f"[bench/{backend}] {DISPLAY[kind]} on {where}: loaded in {load_s:.2f}s", flush=True)

            b = batched_pass(run, x, y, args.batch_size, args.warmup, args.repeats)
            print(f"  acc {b['acc']:.2f}% loss {b['loss']:.4f} | {b['throughput_ips']:.1f} img/s "
                  f"(bs {args.batch_size}, passes {', '.join(f'{t:.1f}s' for t in b['pass_times_s'])})", flush=True)
            lat, pred1 = single_image_pass(run, x, n_lat, args.warmup)
            agree_bs1 = float((pred1 == b["pred"][:n_lat]).mean() * 100)
            print(f"  batch-1 latency mean {lat.mean():.2f} ms, p50 {np.median(lat):.2f}, p95 "
                  f"{np.percentile(lat, 95):.2f} ms over {n_lat} images (bs1/batched agreement {agree_bs1:.2f}%)",
                  flush=True)

            params = count_parameters(model)
            nonzero, zw, tw = weight_stats(model)
            gz = len(gzip.compress(blob, compresslevel=6))
            if kind == "fp32":
                fp32_size = size_b
            row = {
                "model": kind, "display": DISPLAY[kind], "backend": "pytorch" if backend == "pytorch" else "onnxruntime",
                "status": "ok", "device": where, "test_images": len(x),
                "test_acc": round(b["acc"], 3), "test_loss": round(b["loss"], 5),
                "size_mb": round(size_b / 2**20, 3), "size_gzip_mb": round(gz / 2**20, 3),
                "params": params, "nonzero_params": nonzero,
                "weight_sparsity": round(zw / tw, 5) if tw else 0.0,
                "macs_m": round(macs_of(kind, ckpt) / 1e6, 2),
                "compression_ratio": round(fp32_size / size_b, 3) if fp32_size else "",
                "gzip_compression_ratio": round(fp32_size / gz, 3) if fp32_size else "",
                "param_compression": "",
                "batch_size": args.batch_size, "batch_latency_ms": round(b["batch_latency_ms"], 3),
                "amortized_ms_per_image": round(b["amortized_ms_per_image"], 4),
                "throughput_ips": round(b["throughput_ips"], 2),
                "latency_bs1_mean_ms": round(float(lat.mean()), 3),
                "latency_bs1_p50_ms": round(float(np.median(lat)), 3),
                "latency_bs1_p95_ms": round(float(np.percentile(lat, 95)), 3),
                "latency_bs1_p99_ms": round(float(np.percentile(lat, 99)), 3),
                "latency_samples": n_lat, "bs1_vs_batched_agreement_pct": round(agree_bs1, 3),
                "load_time_s": round(load_s, 3), "threads": args.threads,
                "ckpt_smoke_test": bool(ckpt.get("meta", {}).get("smoke_test", False)),
            }
            summary.append(row)
            lat_full = np.full(len(x), np.nan)
            lat_full[:n_lat] = lat
            preds_frames.append(pd.DataFrame({
                "model": kind, "backend": row["backend"], "test_index": np.arange(len(x)), "label": y,
                "pred": b["pred"], "confidence": np.round(b["conf"], 6), "correct": (b["pred"] == y).astype(int),
                "pred_bs1": np.where(np.arange(len(x)) < n_lat, np.pad(pred1, (0, len(x) - n_lat)), -1),
                "inference_time_bs1_ms": np.round(lat_full, 4),
                "amortized_batch_time_ms": round(b["amortized_ms_per_image"], 4)}))
            del model, run

        df = pd.DataFrame(summary)
        if "params" in df and "fp32" in df["model"].values:
            p0 = df.loc[df["model"] == "fp32", "params"].iloc[0]
            df["param_compression"] = (p0 / df["params"]).round(3)
            t0 = df.loc[df["model"] == "fp32", "throughput_ips"].iloc[0]
            l0 = df.loc[df["model"] == "fp32", "latency_bs1_mean_ms"].iloc[0]
            df["throughput_speedup"] = (df["throughput_ips"] / t0).round(3)
            df["latency_bs1_speedup"] = (l0 / df["latency_bs1_mean_ms"]).round(3)
            df["acc_delta_vs_fp32"] = (df["test_acc"] - df.loc[df["model"] == "fp32", "test_acc"].iloc[0]).round(3)
        name = "benchmark_summary.csv" if backend == "pytorch" else "onnx_benchmark_summary.csv"
        df.to_csv(RESULTS_DIR / name, index=False)
        pname = "predictions.csv" if backend == "pytorch" else "onnx_predictions.csv"
        if preds_frames:
            pd.concat(preds_frames, ignore_index=True).to_csv(RESULTS_DIR / pname, index=False)
        print(f"[bench/{backend}] wrote results/{name} and results/{pname}")

    (RESULTS_DIR / "benchmark_env.json").write_text(json.dumps(env, indent=2))
    build_comparison()


def build_comparison():
    """Merge PyTorch and ONNX Runtime summaries into results/model_comparison.csv."""
    pt_path, ox_path = RESULTS_DIR / "benchmark_summary.csv", RESULTS_DIR / "onnx_benchmark_summary.csv"
    if not pt_path.exists():
        return
    pt = pd.read_csv(pt_path)
    cols = ["model", "test_acc", "test_loss", "size_mb", "throughput_ips", "latency_bs1_mean_ms"]
    out = pt[["model", "display", "params", "nonzero_params", "weight_sparsity", "macs_m", "size_mb",
              "size_gzip_mb", "compression_ratio", "gzip_compression_ratio", "param_compression", "test_acc",
              "acc_delta_vs_fp32", "test_loss", "throughput_ips", "throughput_speedup", "latency_bs1_mean_ms",
              "latency_bs1_speedup"]].rename(columns={c: f"torch_{c}" for c in cols[1:]})
    if ox_path.exists():
        ox = pd.read_csv(ox_path)
        ox = ox[ox["status"] == "ok"][cols].rename(columns={c: f"onnx_{c}" for c in cols[1:]})
        out = out.merge(ox, on="model", how="left")
        pp, op = RESULTS_DIR / "predictions.csv", RESULTS_DIR / "onnx_predictions.csv"
        if pp.exists() and op.exists():
            a = pd.read_csv(pp, usecols=["model", "test_index", "pred"])
            b = pd.read_csv(op, usecols=["model", "test_index", "pred"])
            m = a.merge(b, on=["model", "test_index"], suffixes=("_t", "_o"))
            agree = (m["pred_t"] == m["pred_o"]).groupby(m["model"]).mean().mul(100).round(3)
            out["torch_onnx_pred_agreement_pct"] = out["model"].map(agree)
    out.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("[bench] wrote results/model_comparison.csv")


if __name__ == "__main__":
    main()
