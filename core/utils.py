"""Paths, seeding, device selection and shared CLI arguments."""
from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
# Smoke tests set A3_OUTPUT_ROOT (e.g. runs/smoke) so their checkpoints / CSVs / ONNX files
# never overwrite the final experiment artefacts.
_OUT = Path(os.environ["A3_OUTPUT_ROOT"]).resolve() if os.environ.get("A3_OUTPUT_ROOT") else None
CKPT_DIR = (_OUT / "checkpoints") if _OUT else ROOT / "models" / "checkpoints"
RESULTS_DIR = (_OUT / "results") if _OUT else ROOT / "results"
ONNX_DIR = (_OUT / "onnx") if _OUT else ROOT / "onnx"
REPORT_DIR = (_OUT / "docs") if _OUT else DOCS_DIR  # generated figures + tables

CKPT = {
    "fp32": CKPT_DIR / "baseline_fp32.pth",
    "ptq": CKPT_DIR / "ptq_int8.pth",
    "qat": CKPT_DIR / "qat_int8.pth",
    "unstructured": CKPT_DIR / "unstructured_pruned.pth",
    "structured": CKPT_DIR / "structured_pruned.pth",
}
MODEL_NAMES = list(CKPT.keys())
DISPLAY = {
    "fp32": "Original FP32",
    "ptq": "PTQ INT8",
    "qat": "QAT INT8",
    "unstructured": "Unstructured-pruned",
    "structured": "Structured-pruned",
}


def quiet_warnings():
    """torch.ao.quantization (eager mode) is deprecated in PyTorch >= 2.10 but still the
    documented way to produce INT8 eager models; silence the repeated deprecation noise."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*")
    warnings.filterwarnings("ignore", message=".*quantize_per_channel.*")
    warnings.filterwarnings("ignore", message=".*Default qconfig of oneDNN.*")
    warnings.filterwarnings("ignore", message=".*must run observer before calling calculate_qparams.*")
    warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data-dir", default=str(DATA_DIR), help="CIFAR-10 root (downloaded if missing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = PyTorch default)")
    p.add_argument("--train-subset", type=int, default=None,
                   help="use only N training images (smoke tests; results are NOT final)")
    p.add_argument("--test-limit", type=int, default=None,
                   help="evaluate on the first N test images only (smoke tests)")
    return p


def setup(args):
    quiet_warnings()
    if getattr(args, "threads", 0):
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    for d in (CKPT_DIR, RESULTS_DIR, ONNX_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return get_device(args.device)
