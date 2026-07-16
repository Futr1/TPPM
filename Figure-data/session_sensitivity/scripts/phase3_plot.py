#!/usr/bin/env python3
"""Phase 3: Plot session count sensitivity figure.

Reads aggregate_results.json from Phase 2 and produces:
  - Accuracy vs N line chart (overall)
  - Per-type breakdown (optional)

Usage:
    python3 phase3_plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
AGGREGATE_PATH = ROOT / "eval_results" / "aggregate_results.json"
FIGURES_DIR = ROOT / "figures"


def plot_overall(aggregate: dict):
    """Plot overall Accuracy vs N."""
    overall = aggregate["overall"]
    ns = sorted(int(k) for k in overall.keys())
    accs = [overall[str(n)]["accuracy"] * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, accs, marker="o", linewidth=2, markersize=8, color="#2196F3")
    ax.set_xlabel("Number of Sessions (N)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("TPPM Session Count Sensitivity on LongMemEval", fontsize=14)
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(accs) * 1.2 if accs else 100)

    for n, acc in zip(ns, accs):
        ax.annotate(f"{acc:.1f}%", (n, acc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "session_sensitivity_overall.pdf"
    fig.savefig(out, dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_by_type(aggregate: dict):
    """Plot per-type Accuracy vs N."""
    by_type = aggregate["by_type"]
    ns = sorted(int(k) for k in by_type.keys())

    # Collect all types
    all_types = set()
    for n_str in by_type:
        all_types.update(by_type[n_str].keys())
    all_types = sorted(all_types)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_types)))

    for qtype, color in zip(all_types, colors):
        type_ns = []
        type_accs = []
        for n in ns:
            n_str = str(n)
            if n_str in by_type and qtype in by_type[n_str]:
                type_ns.append(n)
                type_accs.append(by_type[n_str][qtype]["accuracy"] * 100)
        if type_ns:
            ax.plot(type_ns, type_accs, marker="s", linewidth=1.5,
                    markersize=5, color=color, label=qtype)

    ax.set_xlabel("Number of Sessions (N)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("TPPM Session Count Sensitivity by Question Type", fontsize=14)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "session_sensitivity_by_type.pdf"
    fig.savefig(out, dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    if not AGGREGATE_PATH.exists():
        print(f"ERROR: {AGGREGATE_PATH} not found. Run Phase 2 first.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(AGGREGATE_PATH.read_text())

    plot_overall(aggregate)

    if aggregate.get("by_type"):
        plot_by_type(aggregate)


if __name__ == "__main__":
    main()
