#!/usr/bin/env bash
# Final experiment run under the 2-hour deadline (reduced fine-tuning budgets, documented in the report).
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "\[train\] saved" runs/train_log.txt; do sleep 15; done
python -u scripts/ptq.py --calib-samples 2000                                          > runs/ptq_log.txt 2>&1
python -u scripts/qat.py --epochs 1 --train-subset 15000 --lr 1e-3                     > runs/qat_log.txt 2>&1
python -u scripts/unstructured_pruning.py --target-sparsity 0.8 --iterations 3 \
       --finetune-epochs 1 --final-epochs 0 --train-subset 15000                       > runs/unstructured_log.txt 2>&1
python -u scripts/structured_pruning.py --internal-ratio 0.5 --stream-ratio 0.25 \
       --epochs 3 --train-subset 22500                                                  > runs/structured_log.txt 2>&1
python -u scripts/export_onnx.py --num-samples 2000                                    > runs/onnx_log.txt 2>&1
python -u scripts/benchmark.py --backend both --batch-size 256 --latency-samples 2000 --repeats 2 > runs/benchmark_log.txt 2>&1
python -u scripts/make_report.py                                                       > runs/report_log.txt 2>&1
echo ALL_STAGES_DONE
