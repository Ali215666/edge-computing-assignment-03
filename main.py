"""Pipeline orchestrator for the Assignment-03 model-optimisation project.

Stages (run in this order; existing checkpoints are reused unless --force):
  data          download / verify CIFAR-10
  train         FP32 ResNet-18 baseline                    -> models/checkpoints/baseline_fp32.pth
  ptq           post-training static INT8 quantization     -> ptq_int8.pth
  qat           quantization-aware training + INT8 convert -> qat_int8.pth
  unstructured  iterative global L1 unstructured pruning   -> unstructured_pruned.pth
  structured    L1 channel pruning + architecture rebuild  -> structured_pruned.pth
  onnx          ONNX export + ONNX Runtime validation      -> onnx/*.onnx
  benchmark     PyTorch + ONNX Runtime benchmarks          -> results/*.csv
  report        figures + markdown tables                  -> docs/figures, docs/results_tables.md

Examples
  python main.py                                  # full pipeline (default hyper-parameters)
  python main.py --stages ptq qat --force         # redo only the quantization stages
  python main.py --stages benchmark report        # re-measure with existing checkpoints
  python main.py --smoke                          # fast end-to-end check (writes to runs/smoke/)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["data", "train", "ptq", "qat", "unstructured", "structured", "onnx", "benchmark", "report"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run (1k train images, 1 epoch each) into runs/smoke/ - NOT final results")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=0)
    # baseline
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--resume", action="store_true", help="resume interrupted baseline training")
    # quantization
    p.add_argument("--calib-samples", type=int, default=2000)
    p.add_argument("--qat-epochs", type=int, default=2)
    p.add_argument("--qat-lr", type=float, default=1e-3)
    # pruning
    p.add_argument("--target-sparsity", type=float, default=0.8)
    p.add_argument("--prune-iterations", type=int, default=3)
    p.add_argument("--prune-finetune-epochs", type=int, default=1)
    p.add_argument("--prune-final-epochs", type=int, default=1)
    p.add_argument("--internal-ratio", type=float, default=0.5)
    p.add_argument("--stream-ratio", type=float, default=0.25)
    p.add_argument("--structured-epochs", type=int, default=4)
    # benchmark
    p.add_argument("--bench-batch-size", type=int, default=256)
    p.add_argument("--latency-samples", type=int, default=10000)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args(argv)


def run(script: str, args: list, env: dict):
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + [str(a) for a in args]
    print(f"\n{'=' * 80}\n$ {' '.join(cmd[1:])}\n{'=' * 80}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"stage failed ({script}, exit code {r.returncode})")
    print(f"--- {script} finished in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    a = parse_args(argv)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")  # live progress in logs
    common = ["--data-dir", a.data_dir, "--seed", a.seed, "--batch-size", a.batch_size, "--threads", a.threads]
    force = ["--force"] if a.force else []
    smoke = []
    if a.smoke:
        env["A3_OUTPUT_ROOT"] = str(ROOT / "runs" / "smoke")
        smoke = ["--train-subset", 1024, "--test-limit", 1000]
        a.epochs, a.qat_epochs, a.structured_epochs = 2, 1, 1
        a.prune_iterations, a.prune_finetune_epochs, a.prune_final_epochs = 2, 1, 0
        a.calib_samples, a.latency_samples, a.repeats = 256, 200, 1
        print(f"[main] SMOKE TEST -> outputs in {env['A3_OUTPUT_ROOT']} (not final results)")
    t0 = time.time()
    for stage in a.stages:
        if stage == "data":
            sys.path.insert(0, str(ROOT))
            from core.data import load_cifar10
            d = load_cifar10(a.data_dir, seed=a.seed)
            print(f"[main] CIFAR-10 ready: train {len(d.train_y)}, val {len(d.val_y)}, test {len(d.test_y)}")
        elif stage == "train":
            run("train.py", common + smoke + ["--epochs", a.epochs, "--lr", a.lr] + force
                + (["--resume"] if a.resume else []), env)
        elif stage == "ptq":
            run("ptq.py", common + smoke[2:] + ["--calib-samples", a.calib_samples] + force, env)
        elif stage == "qat":
            run("qat.py", common + smoke + ["--epochs", a.qat_epochs, "--lr", a.qat_lr] + force, env)
        elif stage == "unstructured":
            run("unstructured_pruning.py", common + smoke + [
                "--target-sparsity", a.target_sparsity, "--iterations", a.prune_iterations,
                "--finetune-epochs", a.prune_finetune_epochs, "--final-epochs", a.prune_final_epochs] + force, env)
        elif stage == "structured":
            run("structured_pruning.py", common + smoke + [
                "--internal-ratio", a.internal_ratio, "--stream-ratio", a.stream_ratio,
                "--epochs", a.structured_epochs] + force, env)
        elif stage == "onnx":
            run("export_onnx.py", ["--data-dir", a.data_dir, "--num-samples", 500 if a.smoke else 2000], env)
        elif stage == "benchmark":
            run("benchmark.py", ["--data-dir", a.data_dir, "--batch-size", a.bench_batch_size,
                                 "--latency-samples", a.latency_samples, "--repeats", a.repeats]
                + (["--test-limit", 1000] if a.smoke else []), env)
        elif stage == "report":
            run("make_report.py", [], env)
    print(f"\n[main] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
