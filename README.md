# Edge Computing – Assignment 03: Quantization & Pruning of ResNet-18 on CIFAR-10

A complete, reproducible PyTorch pipeline that trains a CIFAR-10 ResNet-18 and compresses it
in four different ways, then benchmarks and deploys all five models:

| # | Model | Method | Script |
|---|---|---|---|
| 1 | Original FP32 | baseline training | `scripts/train.py` |
| 2 | PTQ INT8 | post-training static quantization (eager mode, calibrated) | `scripts/ptq.py` |
| 3 | QAT INT8 | quantization-aware training, then INT8 conversion | `scripts/qat.py` |
| 4 | Unstructured-pruned | iterative global L1-magnitude pruning, made permanent with `prune.remove` | `scripts/unstructured_pruning.py` |
| 5 | Structured-pruned | L1-norm channel pruning with **physical** architecture reconstruction (residual-aware) | `scripts/structured_pruning.py` |

All five are benchmarked on the full CIFAR-10 test set (`scripts/benchmark.py`) and exported to
ONNX and validated with ONNX Runtime (`scripts/export_onnx.py`). Measured results, plots and a
discussion are in **[docs/technical_report.md](docs/technical_report.md)** (PDF version:
**[docs/technical_report.pdf](docs/technical_report.pdf)**, rebuilt with `python scripts/make_pdf.py`).

### Results at a glance (measured, full test set, CPU, from `results/model_comparison.csv`)

| Model | Test acc | Size | Throughput (PyTorch / ONNX Runtime) | Latency, batch 1 (PyTorch / ORT) |
|---|---|---|---|---|
| Original FP32 | 91.56 % | 42.7 MB | 97 / 165 img/s | 22.1 / 9.4 ms |
| PTQ INT8 | 91.56 % | 10.8 MB (3.96×) | 286 / 309 img/s | 12.4 / 4.5 ms |
| QAT INT8 | 91.74 % | 10.8 MB (3.96×) | 284 / 317 img/s | 12.5 / 4.5 ms |
| Unstructured-pruned (80 % zeros) | 91.52 % | 42.7 MB (12.2 MB gzipped) | 94 / 163 img/s | 22.0 / 9.1 ms |
| Structured-pruned (2.64× fewer params) | 91.51 % | 16.2 MB (2.64×) | 208 / 425 img/s | 13.8 / 3.2 ms |

All accuracy differences from FP32 are within noise (McNemar p > 0.05; see report §5.1).
`python scripts/validate_project.py` passes 28/28 checks.

## Project layout

```
├── main.py                      # pipeline orchestrator (all stages, individually selectable)
├── models/
│   ├── resnet.py                # CIFAR ResNet-18 (quantization-ready, width-configurable)
│   └── checkpoints/             # the five .pth checkpoints (+ train_state.pth, qat_fakequant.pth)
├── core/                        # shared library code
│   ├── data.py                  # CIFAR-10 loading, seeded 45k/5k/10k split, fast tensor augmentation
│   ├── engine.py                # train / evaluate / fine-tune loops
│   ├── quant.py                 # backend selection, qconfigs, INT8 model skeleton
│   ├── structured.py            # channel-importance + network surgery for structured pruning
│   ├── checkpoints.py           # save / load for all five model kinds
│   ├── preprocess.py            # inference preprocessing (shared by PyTorch & ONNX paths)
│   ├── flops.py, history.py, utils.py
├── scripts/
│   ├── train.py  ptq.py  qat.py  unstructured_pruning.py  structured_pruning.py
│   ├── inference.py             # unified PyTorch inference for the 5 models
│   ├── export_onnx.py           # ONNX export + ONNX Runtime validation
│   ├── onnx_inference.py        # ONNX Runtime inference (numpy preprocessing)
│   ├── benchmark.py             # PyTorch + ONNX Runtime benchmarks -> CSVs
│   ├── make_report.py           # figures + markdown tables from the CSVs
│   └── validate_project.py      # automated end-to-end checks
├── results/                     # all measured CSV/JSON outputs
├── onnx/                        # *.onnx exports + manifest.json (export method / errors / validation)
├── docs/
│   ├── technical_report.md      # the report
│   ├── results_tables.md        # auto-generated tables
│   └── figures/                 # auto-generated plots
├── runs/                        # logs; runs/smoke/ = smoke-test artefacts, runs/reduced_budget/ = earlier 1-epoch QAT run (both NOT final results)
└── requirements.txt
```

## Installation

Python 3.10–3.12. CPU is enough. A CUDA GPU is used automatically for training and for float
inference if one is available.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
# CPU-only PyTorch wheels, if you need them:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Tested with: torch 2.14.0+cpu, torchvision 0.29.0, onnx 1.23.2, onnxruntime 1.32.0,
onnxscript, numpy 2.5, pandas 3.1, matplotlib 3.11 on Windows 11 / Python 3.12.10.

## Dataset

CIFAR-10 is downloaded automatically by torchvision into `./data` (≈170 MB) the first time any
stage runs. To use an existing copy, pass `--data-dir <dir containing cifar-10-batches-py>`.

Splits (seed 42): **45,000 train / 5,000 validation** (both from the official training set) and the
official **10,000 test** images. Model selection and fine-tuning monitoring use the validation split
only. The test set is used only for final evaluation and benchmarking. PTQ calibration uses
a random subset of the *training* split.

## Running the complete pipeline

```bash
python main.py                       # data → train → ptq → qat → unstructured → structured → onnx → benchmark → report
python scripts/validate_project.py   # automated validation of every artefact
```

Existing checkpoints are reused. Add `--force` to recompute. Run only some stages:

```bash
python main.py --stages ptq qat --force             # redo quantization only
python main.py --stages unstructured --force --target-sparsity 0.9 --prune-iterations 4
python main.py --stages onnx benchmark report       # re-export and re-measure
python main.py --resume --stages train              # resume an interrupted baseline training
python main.py --smoke                              # 10-min end-to-end smoke test -> runs/smoke/
```

Configurable hyper-parameters (see `python main.py --help`): `--epochs --lr --batch-size --seed
--data-dir --calib-samples --qat-epochs --qat-lr --target-sparsity --prune-iterations
--prune-finetune-epochs --prune-final-epochs --internal-ratio --stream-ratio --structured-epochs
--bench-batch-size --latency-samples --repeats --threads`.

## Running individual scripts

Every script has `--help`. Defaults reproduce the reported results.

```bash
python scripts/train.py --epochs 15 --lr 0.1 --batch-size 128      # [--resume] [--force]
python scripts/ptq.py --calib-samples 2000                         # [--engine x86|fbgemm|onednn|qnnpack]
python scripts/qat.py --epochs 2 --lr 1e-3
python scripts/unstructured_pruning.py --target-sparsity 0.8 --iterations 3 --finetune-epochs 1 --final-epochs 1
python scripts/structured_pruning.py --internal-ratio 0.5 --stream-ratio 0.25 --epochs 4
python scripts/export_onnx.py --num-samples 2000
python scripts/benchmark.py --backend both --batch-size 256 --latency-samples 10000 --repeats 3
python scripts/make_report.py
```

Inference on a CIFAR-10 test sample or on any image file:

```bash
python scripts/inference.py --model all --index 7
python scripts/inference.py --model qat --image path/to/picture.jpg
python scripts/onnx_inference.py --model structured --index 7      # ONNX Runtime
```

## Outputs

| What | Where |
|---|---|
| FP32 baseline | `models/checkpoints/baseline_fp32.pth` |
| PTQ INT8 | `models/checkpoints/ptq_int8.pth` |
| QAT INT8 | `models/checkpoints/qat_int8.pth` (fake-quant model: `qat_fakequant.pth`) |
| Unstructured-pruned | `models/checkpoints/unstructured_pruned.pth` |
| Structured-pruned | `models/checkpoints/structured_pruned.pth` (stores its reduced channel config) |
| Training history (all stages) | `results/training_history.csv` |
| PyTorch benchmark summary | `results/benchmark_summary.csv` |
| Per-image predictions / confidences / latencies | `results/predictions.csv` (ONNX: `results/onnx_predictions.csv`) |
| ONNX Runtime benchmark | `results/onnx_benchmark_summary.csv` |
| PyTorch vs ONNX comparison | `results/model_comparison.csv` |
| ONNX export validation | `results/onnx_validation.csv`, `onnx/manifest.json` |
| ONNX models | `onnx/{fp32,ptq,qat,unstructured,structured}.onnx` |
| Validation report | `results/validation_report.json` |
| Logs | `runs/*.txt` |

Checkpoints are dicts `{kind, config, state_dict, meta, [quant_engine, reduce_range]}`. Load any of
them with `core.checkpoints.load_checkpoint("<kind>")`.

## Hardware requirements and limitations

* **Measured on:** AMD Ryzen 5 PRO 5650U laptop CPU (6 cores / 12 threads, AVX2, no AVX-512/VNNI),
  16 GB RAM, no GPU. Measured wall times: one training epoch ≈ 12–19 min. Baseline (15 epochs)
  ≈ 3.4 h, then QAT 51 min, unstructured 94 min, structured 45 min, ONNX 2 min, benchmark 48 min:
  ≈ 7.6 h for the full pipeline.
  A CUDA GPU cuts the training stages to minutes. ~1.5 GB RAM is enough.
* **INT8 models run on CPU only.** PyTorch's eager quantized kernels have no CUDA backend. The
  quantized engine is picked automatically (x86 → fbgemm → onednn → qnnpack). This PyTorch build
  only provides `onednn`. Activations use `reduce_range` (7-bit) to avoid int16 overflow in
  `vpmaddubsw` on CPUs without VNNI, which is the same choice fbgemm makes by default.
* `torch.ao.quantization` (eager mode) is **deprecated** in PyTorch ≥ 2.10 in favour of `torchao`
  but still works in 2.14. The deprecation warnings are silenced.
* **Unstructured sparsity does not shrink the dense file or speed up dense kernels.** Its benefit
  shows in the gzip-compressed size and would need a sparse runtime to turn into speed.
* **ONNX:** the torch.export-based exporter cannot handle eager INT8 modules. INT8 models are
  exported with the TorchScript exporter as QDQ graphs. See the report for agreement numbers and
  details.
* Training budgets (15 baseline epochs, a few fine-tuning epochs) were chosen for a CPU-only
  laptop. Longer schedules would give higher absolute accuracy for every model.
