#!/usr/bin/env python3
"""Experiment 4: TPPM Memory Lifecycle Analysis - Visualization Script.

Reads persona traces from run_pipeline.py and generates:
1. Three-layer memory growth (stacked area chart)
2. Conflict handling analysis (grouped bar chart)
3. Information efficiency evolution (dual-axis line chart)

Usage:
    python3 analyze_and_plot.py
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/memconflict_lifecycle")
TRACES_DIR = ROOT / "traces"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def load_traces() -> list[dict[str, Any]]:
    """Load all persona traces."""
    traces = []
    for path in sorted(TRACES_DIR.glob("persona_*.json")):
        with path.open("r", encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces


def analyze_memory_growth(traces: list[dict]) -> dict:
    """Analyze three-layer memory growth across sessions."""
    # Aggregate across personas
    all_sessions = []
    for trace in traces:
        for st in trace["session_traces"]:
            all_sessions.append(st)

    # Group by session_idx
    session_data = {}
    for st in all_sessions:
        idx = st["session_idx"]
        if idx not in session_data:
            session_data[idx] = {"working": [], "short_term": [], "long_term": []}
        session_data[idx]["working"].append(st["layer_sizes"]["working"])
        session_data[idx]["short_term"].append(st["layer_sizes"]["short_term"])
        session_data[idx]["long_term"].append(st["layer_sizes"]["long_term"])

    # Compute averages
    session_indices = sorted(session_data.keys())
    avg_working = [np.mean(session_data[i]["working"]) for i in session_indices]
    avg_short = [np.mean(session_data[i]["short_term"]) for i in session_indices]
    avg_long = [np.mean(session_data[i]["long_term"]) for i in session_indices]

    # Compute promotion and fusion rates
    promotion_counts = []
    fusion_counts = []
    for trace in traces:
        promo = 0
        fuse = 0
        for st in trace["session_traces"]:
            promo += st["events"]["promoted"]
            fuse += st["events"]["fused"]
        promotion_counts.append(promo)
        fusion_counts.append(fuse)

    return {
        "session_indices": session_indices,
        "avg_working": avg_working,
        "avg_short_term": avg_short,
        "avg_long_term": avg_long,
        "avg_promotions_per_persona": np.mean(promotion_counts) if promotion_counts else 0,
        "avg_fusions_per_persona": np.mean(fusion_counts) if fusion_counts else 0,
    }


def plot_memory_growth(growth_data: dict):
    """Plot stacked area chart of three-layer memory growth."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sessions = growth_data["session_indices"]
    working = growth_data["avg_working"]
    short_term = growth_data["avg_short_term"]
    long_term = growth_data["avg_long_term"]

    # Stack the data
    ax.fill_between(sessions, 0, working, alpha=0.6, label="Working Memory", color="#FF6B6B")
    ax.fill_between(sessions, working, [w + s for w, s in zip(working, short_term)],
                    alpha=0.6, label="Short-term Memory", color="#4ECDC4")
    ax.fill_between(sessions,
                    [w + s for w, s in zip(working, short_term)],
                    [w + s + l for w, s, l in zip(working, short_term, long_term)],
                    alpha=0.6, label="Long-term Memory", color="#45B7D1")

    ax.set_xlabel("Session Index", fontsize=12)
    ax.set_ylabel("Average PMU Count", fontsize=12)
    ax.set_title("Three-Layer Memory Growth Across Sessions", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add summary stats
    stats_text = (
        f"Avg Promotions: {growth_data['avg_promotions_per_persona']:.1f}\n"
        f"Avg Fusions: {growth_data['avg_fusions_per_persona']:.1f}"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    out_path = FIGURES_DIR / "memory_growth_stacked.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


def analyze_conflict_handling(traces: list[dict]) -> dict:
    """Analyze conflict handling by type."""
    # Collect question results by conflict type
    by_type = {}
    for trace in traces:
        for qr in trace["question_results"]:
            ctype = qr["conflict_type"]
            if ctype not in by_type:
                by_type[ctype] = {"correct": 0, "total": 0}
            by_type[ctype]["total"] += 1
            if qr.get("is_correct", False):
                by_type[ctype]["correct"] += 1

    # Compute accuracy
    results = {}
    for ctype, data in by_type.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        results[ctype] = {
            "accuracy": acc,
            "correct": data["correct"],
            "total": data["total"],
        }

    # Count contradiction events
    contradiction_counts = []
    for trace in traces:
        count = sum(st["events"]["contradicted"] for st in trace["session_traces"])
        contradiction_counts.append(count)

    return {
        "by_type": results,
        "avg_contradictions_per_persona": np.mean(contradiction_counts) if contradiction_counts else 0,
    }


def plot_conflict_handling(conflict_data: dict):
    """Plot grouped bar chart of conflict handling accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    by_type = conflict_data["by_type"]
    types = sorted(by_type.keys())
    accuracies = [by_type[t]["accuracy"] * 100 for t in types]
    totals = [by_type[t]["total"] for t in types]

    # Create bar chart
    bars = ax.bar(types, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.7)

    # Add value labels on bars
    for bar, acc, total in zip(bars, accuracies, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%\n(n={total})',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Conflict Type", fontsize=12)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_title("Conflict Handling Performance by Type", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add contradiction stats
    stats_text = f"Avg Contradictions Detected: {conflict_data['avg_contradictions_per_persona']:.1f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    out_path = FIGURES_DIR / "conflict_handling_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


def analyze_information_efficiency(traces: list[dict]) -> dict:
    """Analyze information efficiency evolution."""
    # Aggregate compression ratios by session
    session_compression = {}
    session_distillation = {}

    for trace in traces:
        for st in trace["session_traces"]:
            idx = st["session_idx"]
            if idx not in session_compression:
                session_compression[idx] = []
                session_distillation[idx] = []
            session_compression[idx].append(st["compression_ratio"])
            session_distillation[idx].append(st["n_distillation_candidates"])

    # Compute averages
    session_indices = sorted(session_compression.keys())
    avg_compression = [np.mean(session_compression[i]) for i in session_indices]
    avg_distillation = [np.mean(session_distillation[i]) for i in session_indices]

    return {
        "session_indices": session_indices,
        "avg_compression_ratio": avg_compression,
        "avg_distillation_candidates": avg_distillation,
    }


def plot_information_efficiency(efficiency_data: dict):
    """Plot dual-axis line chart of information efficiency."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sessions = efficiency_data["session_indices"]
    compression = efficiency_data["avg_compression_ratio"]
    distillation = efficiency_data["avg_distillation_candidates"]

    # Primary axis: Compression Ratio
    color1 = "#FF6B6B"
    ax1.set_xlabel("Session Index", fontsize=12)
    ax1.set_ylabel("Avg Compression Ratio", fontsize=12, color=color1)
    line1, = ax1.plot(sessions, compression, marker='o', linewidth=2,
                      markersize=4, color=color1, label="Compression Ratio")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Distillation Candidates
    ax2 = ax1.twinx()
    color2 = "#45B7D1"
    ax2.set_ylabel("Avg Distillation Candidates", fontsize=12, color=color2)
    line2, = ax2.plot(sessions, distillation, marker='s', linewidth=2,
                      markersize=4, color=color2, label="Distillation Candidates")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and legend
    ax1.set_title("Information Efficiency Evolution", fontsize=14, fontweight="bold")
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    fig.tight_layout()
    out_path = FIGURES_DIR / "information_efficiency.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


def generate_summary_table(traces: list[dict], growth_data: dict,
                          conflict_data: dict, efficiency_data: dict):
    """Generate summary statistics table."""
    n_personas = len(traces)
    total_sessions = sum(t["n_sessions"] for t in traces)
    total_questions = sum(len(t["question_results"]) for t in traces)

    # Overall accuracy
    correct = sum(1 for t in traces for qr in t["question_results"] if qr.get("is_correct", False))
    overall_acc = correct / total_questions if total_questions > 0 else 0

    # Final memory stats
    final_short = np.mean([t["final_memory_summary"]["short_term"] for t in traces])
    final_long = np.mean([t["final_memory_summary"]["long_term"] for t in traces])

    summary = {
        "n_personas": n_personas,
        "total_sessions": total_sessions,
        "total_questions": total_questions,
        "overall_qa_accuracy": round(overall_acc, 4),
        "avg_final_short_term": round(final_short, 1),
        "avg_final_long_term": round(final_long, 1),
        "avg_promotions_per_persona": round(growth_data["avg_promotions_per_persona"], 1),
        "avg_fusions_per_persona": round(growth_data["avg_fusions_per_persona"], 1),
        "avg_contradictions_per_persona": round(conflict_data["avg_contradictions_per_persona"], 1),
        "conflict_type_accuracy": {
            ctype: round(data["accuracy"], 4)
            for ctype, data in conflict_data["by_type"].items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "summary_statistics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("  Summary Statistics")
    print(f"{'='*60}")
    print(f"  Personas: {n_personas}")
    print(f"  Total Sessions: {total_sessions}")
    print(f"  Total Questions: {total_questions}")
    print(f"  Overall QA Accuracy: {overall_acc:.2%}")
    print(f"  Avg Final Memory: {final_short:.1f} short-term, {final_long:.1f} long-term")
    print(f"  Avg Promotions: {summary['avg_promotions_per_persona']}")
    print(f"  Avg Fusions: {summary['avg_fusions_per_persona']}")
    print(f"  Avg Contradictions: {summary['avg_contradictions_per_persona']}")
    print(f"\n  Conflict Type Accuracy:")
    for ctype, acc in summary["conflict_type_accuracy"].items():
        print(f"    {ctype}: {acc:.2%}")
    print(f"{'='*60}\n")

    print(f"✓ Saved: {output_path}")


def main():
    print("Loading persona traces...")
    traces = load_traces()

    if not traces:
        print(f"ERROR: No traces found in {TRACES_DIR}")
        print("Run run_pipeline.py first.")
        return

    print(f"Loaded {len(traces)} persona traces\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Analysis 1: Memory Growth
    print("Analyzing memory growth...")
    growth_data = analyze_memory_growth(traces)
    plot_memory_growth(growth_data)

    # Analysis 2: Conflict Handling
    print("Analyzing conflict handling...")
    conflict_data = analyze_conflict_handling(traces)
    plot_conflict_handling(conflict_data)

    # Analysis 3: Information Efficiency
    print("Analyzing information efficiency...")
    efficiency_data = analyze_information_efficiency(traces)
    plot_information_efficiency(efficiency_data)

    # Summary table
    print("Generating summary statistics...")
    generate_summary_table(traces, growth_data, conflict_data, efficiency_data)

    print(f"\n{'='*60}")
    print("  Analysis complete!")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
