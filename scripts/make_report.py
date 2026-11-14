"""Build report figures (docs/figures/*.png) and markdown tables (docs/results_tables.md)
from the measured CSVs in results/. Nothing here is hand-entered: every number comes from
results/*.csv written by the other scripts.

    python scripts/make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from core.utils import REPORT_DIR as DOCS_DIR, RESULTS_DIR  # noqa: E402

# reference categorical palette (fixed order) + neutral inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})
SHORT = {"fp32": "FP32", "ptq": "PTQ INT8", "qat": "QAT INT8", "unstructured": "Unstr. pruned",
         "structured": "Struct. pruned"}


def fig_training(hist: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    b = hist[hist["stage"] == "baseline"]
    ax = axes[0]
    ax.plot(b["epoch"], b["train_acc"], color=SERIES[0], lw=2, marker="o", ms=4, label="train (augmented)")
    ax.plot(b["epoch"], b["val_acc"], color=SERIES[1], lw=2, marker="o", ms=4, label="validation")
    ax.set(title="Baseline FP32 training", xlabel="epoch", ylabel="accuracy (%)")
    ax.legend(loc="lower right")
    ax = axes[1]
    groups = [("qat", "QAT (fake-quant)"), ("unstructured", "Unstructured pruning"),
              ("structured", "Structured pruning")]
    for i, (prefix, label) in enumerate(groups):
        g = hist[hist["stage"].astype(str).str.startswith(prefix)].reset_index(drop=True)
        if len(g):
            ax.plot(range(1, len(g) + 1), g["val_acc"], color=SERIES[i], lw=2, marker="o", ms=5, label=label)
    if len(b):
        ax.axhline(b["val_acc"].max(), color=INK2, lw=1, ls="--")
        ax.annotate("best FP32 val", (0.02, b["val_acc"].max()), xycoords=ax.get_yaxis_transform(),
                    textcoords="offset points", xytext=(0, -4), ha="left", va="top", color=INK2, fontsize=9)
    ax.set(title="Fine-tuning (validation accuracy)", xlabel="fine-tuning epoch (cumulative)",
           ylabel="accuracy (%)")
    ax.legend(loc="lower right")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_tradeoff(s: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.scatter(s["size_mb"], s["test_acc"], s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    # per-model label placement: FP32/unstructured share a size; PTQ/QAT/structured sit close together
    place = {"fp32": ((9, 3), "left", "bottom"), "unstructured": ((9, -3), "left", "top"),
             "qat": ((0, 9), "center", "bottom"), "ptq": ((0, -10), "center", "top"),
             "structured": ((9, 0), "left", "center")}
    for _, r in s.iterrows():
        off, ha, va = place.get(r["model"], ((8, -4), "left", "top"))
        ax.annotate(f"{SHORT[r['model']]}\n{r['test_acc']:.2f}%, {r['size_mb']:.1f} MB", (r["size_mb"], r["test_acc"]),
                    textcoords="offset points", xytext=off, ha=ha, va=va, fontsize=8.5, color=INK2)
    ax.set(title="Accuracy vs serialized model size (PyTorch)", xlabel="model size (MB)",
           ylabel="test accuracy (%)")
    lo, hi = s["test_acc"].min(), s["test_acc"].max()
    ax.set_ylim(lo - max(0.5, (hi - lo) * 0.3), hi + max(0.5, (hi - lo) * 0.3))
    ax.set_xlim(0, s["size_mb"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_bars(s: pd.DataFrame, out: Path):
    metrics = [("size_mb", "model size (MB)", "{:.1f}"), ("throughput_ips", "throughput (img/s, batch 256)", "{:.0f}"),
               ("latency_bs1_mean_ms", "latency, batch 1 (ms/img)", "{:.2f}"), ("test_acc", "test accuracy (%)", "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    labels = [SHORT[m] for m in s["model"]]
    for ax, (col, title, fmt) in zip(axes, metrics):
        if col == "test_acc":  # small differences: dots on a zoomed axis, not truncated bars
            ax.scatter(labels, s[col], s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
            for i, v in enumerate(s[col]):
                ax.annotate(fmt.format(v), (i, v), textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=8, color=INK2)
        else:
            bars = ax.bar(labels, s[col], color=SERIES[0], width=0.62, edgecolor=SURFACE, linewidth=2)
            for b_, v in zip(bars, s[col]):
                ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fmt.format(v), ha="center",
                        va="bottom", fontsize=8, color=INK2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8.5)
        ax.grid(axis="x", visible=False)
        if col == "test_acc":
            pad = max(0.5, (s[col].max() - s[col].min()) * 0.25)
            ax.set_ylim(s[col].min() - pad, s[col].max() + pad)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_backends(cmp: pd.DataFrame, out: Path):
    if "onnx_throughput_ips" not in cmp:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = range(len(cmp))
    w = 0.38
    for ax, (t, o, title) in zip(axes, [("torch_throughput_ips", "onnx_throughput_ips", "throughput (img/s, batch 256)"),
                                        ("torch_latency_bs1_mean_ms", "onnx_latency_bs1_mean_ms",
                                         "latency, batch 1 (ms/img)")]):
        ax.bar([i - w / 2 for i in x], cmp[t], w, color=SERIES[0], edgecolor=SURFACE, linewidth=2, label="PyTorch")
        ax.bar([i + w / 2 for i in x], cmp[o], w, color=SERIES[1], edgecolor=SURFACE, linewidth=2,
               label="ONNX Runtime")
        ax.set_xticks(list(x), [SHORT[m] for m in cmp["model"]], rotation=20, fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    fig_dir = DOCS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Measured results (auto-generated by scripts/make_report.py)\n"]
    env_p = RESULTS_DIR / "benchmark_env.json"
    if env_p.exists():
        env = json.loads(env_p.read_text())
        md.append("Benchmark environment: " + ", ".join(f"{k}={v}" for k, v in env.items()) + "\n")

    hist_p = RESULTS_DIR / "training_history.csv"
    if hist_p.exists():
        hist = pd.read_csv(hist_p)
        fig_training(hist, fig_dir / "training_curves.png")
        md += ["## Training / fine-tuning history\n", md_table(hist), "\n"]

    s_p = RESULTS_DIR / "benchmark_summary.csv"
    if s_p.exists():
        s = pd.read_csv(s_p)
        s = s[s["status"] == "ok"]
        fig_tradeoff(s, fig_dir / "accuracy_vs_size.png")
        fig_bars(s, fig_dir / "benchmark_bars.png")
        cols = ["display", "test_acc", "acc_delta_vs_fp32", "test_loss", "size_mb", "size_gzip_mb",
                "compression_ratio", "gzip_compression_ratio", "params", "nonzero_params", "param_compression",
                "weight_sparsity", "macs_m", "batch_latency_ms", "amortized_ms_per_image", "throughput_ips",
                "throughput_speedup", "latency_bs1_mean_ms", "latency_bs1_p95_ms", "latency_bs1_speedup"]
        md += ["## PyTorch benchmark (results/benchmark_summary.csv)\n", md_table(s[cols]), "\n"]
    o_p = RESULTS_DIR / "onnx_benchmark_summary.csv"
    if o_p.exists():
        o = pd.read_csv(o_p)
        cols = [c for c in ["display", "status", "test_acc", "test_loss", "size_mb", "compression_ratio",
                            "throughput_ips", "batch_latency_ms", "latency_bs1_mean_ms", "latency_bs1_p95_ms"]
                if c in o]
        md += ["## ONNX Runtime benchmark (results/onnx_benchmark_summary.csv)\n", md_table(o[cols]), "\n"]
    v_p = RESULTS_DIR / "onnx_validation.csv"
    if v_p.exists():
        md += ["## ONNX export validation (results/onnx_validation.csv)\n", md_table(pd.read_csv(v_p)), "\n"]
    c_p = RESULTS_DIR / "model_comparison.csv"
    if c_p.exists():
        c = pd.read_csv(c_p)
        if fig_backends(c, fig_dir / "pytorch_vs_onnx.png"):
            pass
        md += ["## PyTorch vs ONNX Runtime (results/model_comparison.csv)\n", md_table(c), "\n"]
    (DOCS_DIR / "results_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[report] wrote {DOCS_DIR / 'results_tables.md'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
