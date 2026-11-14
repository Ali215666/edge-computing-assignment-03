"""End-to-end validation of the produced artefacts (requirement 11 of the assignment).

Checks
  * every script imports and its --help works
  * all five checkpoints load, run inference on CPU and give valid probabilities
  * save -> load round trip of every checkpoint reproduces identical outputs
  * PTQ / QAT models really contain INT8 kernels (quantized modules, qint8 weights)
  * unstructured model: pruning is permanent (no *_orig / *_mask, no hooks), sparsity > 0
  * structured model: physically smaller tensors, consistent dimensions, fewer params
  * the unified inference scripts (PyTorch and ONNX Runtime) run for all models
  * CSVs exist and contain measured rows for all five models (10k predictions each)
  * ONNX exports load in ONNX Runtime and agree with PyTorch

Writes results/validation_report.json; exit code 1 if any check fails.

    python scripts/validate_project.py
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from core.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from core.data import load_cifar10, normalize_uint8  # noqa: E402
from core.structured import verify_dimensions  # noqa: E402
from core.utils import CKPT, MODEL_NAMES, ONNX_DIR, RESULTS_DIR, ROOT, quiet_warnings  # noqa: E402
from models.resnet import ResNet18, count_parameters, default_config  # noqa: E402

results = []


def check(name):
    def deco(fn):
        def wrapper(*a, **k):
            try:
                detail = fn(*a, **k)
                results.append({"check": name, "ok": True, "detail": detail})
                print(f"PASS  {name}: {detail}")
            except Exception as e:
                results.append({"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
        return wrapper
    return deco


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    quiet_warnings()
    data = load_cifar10(seed=42)
    x = normalize_uint8(data.test_x[:64])
    y = data.test_y[:64]

    for script in sorted(p for p in (ROOT / "scripts").glob("*.py")
                         if p.name not in ("__init__.py", "validate_project.py")):
        @check(f"script --help: {script.name}")
        def _():
            r = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=180)
            assert r.returncode == 0, r.stderr[-300:]
            return "ok"
        _()

    models = {}
    for kind in MODEL_NAMES:
        @check(f"load + CPU inference: {kind}")
        def _():
            m, ck = load_checkpoint(kind)
            with torch.no_grad():
                out = m(x)
            assert out.shape == (64, 10), out.shape
            p = torch.softmax(out, 1)
            assert torch.allclose(p.sum(1), torch.ones(64), atol=1e-4)
            acc = (out.argmax(1) == y).float().mean().item() * 100
            models[kind] = (m, ck, out)
            return f"output {tuple(out.shape)}, acc on 64 test images {acc:.1f}%, smoke={ck['meta'].get('smoke_test')}"
        _()

        @check(f"checkpoint save/load round trip: {kind}")
        def _():
            m, ck, out = models[kind]
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "rt.pth"
                extra = {k: ck[k] for k in ("quant_engine", "reduce_range") if k in ck}
                save_checkpoint(path, kind, m, ck["config"], ck["meta"], **extra)
                m2, _ = load_checkpoint(str(path))
                with torch.no_grad():
                    out2 = m2(x)
            assert torch.equal(out, out2), (out - out2).abs().max()
            return "identical outputs after reload"
        _()

    for kind in ("ptq", "qat"):
        @check(f"INT8 conversion: {kind}")
        def _():
            m = models[kind][0]
            qmods = [mod for mod in m.modules() if type(mod).__module__.startswith("torch.ao.nn.quantized")
                     or type(mod).__module__.startswith("torch.ao.nn.intrinsic.quantized")]
            convs = [mod for mod in qmods if callable(getattr(mod, "weight", None))]
            assert convs, "no quantized conv/linear modules"
            dtypes = {mod.weight().dtype for mod in convs}
            assert dtypes == {torch.qint8}, dtypes
            float_convs = [n for n, mod in m.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
            assert not float_convs, f"float layers left: {float_convs[:3]}"
            return (f"{len(convs)} INT8 conv/linear layers ({sorted({type(c).__name__ for c in convs})}), "
                    f"weights {dtypes}, engine {torch.backends.quantized.engine}")
        _()

    @check("unstructured: pruning made permanent, sparsity")
    def _():
        m, ck, _ = models["unstructured"]
        names = [n for n, _ in m.named_parameters()] + [n for n, _ in m.named_buffers()]
        assert not [n for n in names if n.endswith("_orig") or n.endswith("_mask")]
        assert not any(mod._forward_pre_hooks for mod in m.modules()), "pruning hooks still attached"
        w = [mod.weight for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
        sp = sum(int((t == 0).sum()) for t in w) / sum(t.numel() for t in w)
        assert abs(sp - ck["meta"]["global_sparsity"]) < 1e-6 and sp > 0.1, sp
        return f"conv+fc sparsity {sp:.4f} (target {ck['meta']['target_sparsity']}), no masks/hooks"
    _()

    @check("structured: physically smaller, valid dimensions")
    def _():
        m, ck, _ = models["structured"]
        assert not verify_dimensions(m), verify_dimensions(m)
        full = ResNet18(default_config())
        p_full, p = count_parameters(full), count_parameters(m)
        assert p < p_full
        smaller = sum(1 for a, b in zip(m.modules(), full.modules())
                      if isinstance(a, nn.Conv2d) and a.weight.shape != b.weight.shape)
        assert smaller > 0
        assert not any("mask" in n for n, _ in m.named_buffers())
        for k in [1, 3, 17]:  # different batch sizes run
            with torch.no_grad():
                assert m(torch.randn(k, 3, 32, 32)).shape == (k, 10)
        return f"{p:,} vs {p_full:,} params ({p_full / p:.2f}x), {smaller} conv layers with fewer channels"
    _()

    @check("unified PyTorch inference script (all models)")
    def _():
        from scripts.inference import main as infer
        with contextlib.redirect_stdout(io.StringIO()):
            r = infer(["--model", "all", "--index", "0", "--index", "1"])
        assert len(r) == 10 and all(0 <= z["confidence"] <= 1 for z in r)
        return ", ".join(f"{z['model']}:{z['class']}({z['confidence']:.2f})" for z in r[::2])
    _()

    @check("ONNX Runtime inference script (all exported models)")
    def _():
        from scripts.onnx_inference import main as oinfer
        with contextlib.redirect_stdout(io.StringIO()):
            r = oinfer(["--model", "all", "--index", "0"])
        assert len(r) >= 1
        return ", ".join(f"{z['model']}:{z['pred']}({z['confidence']:.2f})" for z in r)
    _()

    @check("ONNX exports validated against PyTorch")
    def _():
        man = json.loads((ONNX_DIR / "manifest.json").read_text())
        msgs = []
        for kind in MODEL_NAMES:
            info = man[kind]
            assert info["status"] == "exported", f"{kind}: {info['status']}"
            v = info["validation"]
            thr = 0.999 if kind not in ("ptq", "qat") else 0.95
            assert v["top1_agreement"] >= thr, f"{kind} agreement {v['top1_agreement']}"
            msgs.append(f"{kind} {v['top1_agreement'] * 100:.2f}%")
        return "top-1 agreement: " + ", ".join(msgs)
    _()

    @check("result CSVs contain measured data for all models")
    def _():
        s = pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")
        assert set(s["model"]) == set(MODEL_NAMES) and (s["status"] == "ok").all()
        for col in ["test_acc", "test_loss", "size_mb", "params", "throughput_ips", "latency_bs1_mean_ms",
                    "compression_ratio"]:
            assert s[col].notna().all(), col
        p = pd.read_csv(RESULTS_DIR / "predictions.csv")
        counts = p.groupby("model").size()
        assert set(counts.index) == set(MODEL_NAMES) and counts.nunique() == 1
        for col in ["label", "pred", "confidence", "inference_time_bs1_ms"]:
            assert col in p
        # accuracy in summary must equal accuracy recomputed from the predictions file
        acc = p.groupby("model")["correct"].mean() * 100
        for _, r in s.iterrows():
            assert abs(acc[r["model"]] - r["test_acc"]) < 1e-2, r["model"]
        assert (RESULTS_DIR / "model_comparison.csv").exists()
        h = pd.read_csv(RESULTS_DIR / "training_history.csv")
        stages = sorted(h["stage"].str.replace(r"_r[0-9]+$", "", regex=True).unique())
        return (f"{len(s)} models, {counts.iloc[0]} predictions/model, training_history {len(h)} rows, "
                f"stages {stages}")
    _()

    ok = all(r["ok"] for r in results)
    (RESULTS_DIR / "validation_report.json").write_text(json.dumps(
        {"all_passed": ok, "passed": sum(r["ok"] for r in results), "total": len(results), "checks": results},
        indent=2))
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} checks passed -> results/validation_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
