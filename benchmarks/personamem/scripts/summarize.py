#!/usr/bin/env python3
"""Summarize Phase 3 evaluation results across all configs.

Outputs accuracy tables (per config, per question_type, per topic) and
generates LaTeX-ready tables for the paper.

Usage:
    python3 summarize.py                           # All configs
    python3 summarize.py --output summary.json     # Custom output path
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/personamem'
EVAL_DIR = ROOT / "eval_results"
DEFAULT_OUTPUT = ROOT / "eval_summary.json"

def load_results(config_dir: Path) -> list[dict]:
    """Load all result rows from a config's results CSV."""
    results: list[dict] = []
    results_csv = config_dir / "results.csv"
    if not results_csv.exists():
        return results
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = row.get("score", "").strip().lower() in ("true", "1", "yes")
            results.append(row)
    return results

def compute_accuracy(rows: list[dict], group_by: str | None = None) -> dict:
    """Compute accuracy, optionally grouped by a column.

    Returns:
        {group_value: {"correct": int, "total": int, "accuracy": float}}
        or {"overall": {...}} if group_by is None.
    """
    if group_by is None:
        correct = sum(1 for r in rows if r["score"])
        total = len(rows)
        return {"overall": {"correct": correct, "total": total,
                            "accuracy": round(correct / total * 100, 2) if total > 0 else 0}}

    groups: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key]["total"] += 1
        if row["score"]:
            groups[key]["correct"] += 1

    for key in groups:
        total = groups[key]["total"]
        groups[key]["accuracy"] = round(groups[key]["correct"] / total * 100, 2) if total > 0 else 0

    return dict(groups)

def generate_latex_table(
    summary: dict,
    sweep_name: str,
    configs: list[str],
    by_question_type: bool = False,
) -> str:
    """Generate a LaTeX table for a parameter sweep.

    Args:
        summary: Full summary dict from all configs.
        sweep_name: Display name for the sweep.
        configs: List of config_ids in this sweep.
        by_question_type: If True, break down by question type.

    Returns:
        LaTeX table string.
    """
    if by_question_type:
        # Get all question types from first available config
        qtypes: list[str] = []
        for cid in configs:
            first = summary.get(cid, {}).get("by_question_type", {})
            if first:
                qtypes = list(first.keys())
                break

        header = " & ".join(["Config"] + qtypes + ["Overall"])
        header += " \\\\\n\\hline"

        rows: list[str] = []
        for cid in configs:
            cfg = summary.get(cid, {})
            overall = cfg.get("overall", {}).get("accuracy", 0)
            by_type = cfg.get("by_question_type", {})
            vals = [f"{by_type.get(qt, {}).get('accuracy', 0):.1f}" for qt in qtypes]
            row = f"{cid} & " + " & ".join(vals) + f" & {overall:.1f} \\\\"
            rows.append(row)
    else:
        header = "Config & Overall Accuracy \\\\\n\\hline"
        rows = []
        for cid in configs:
            overall = summary.get(cid, {}).get("overall", {}).get("accuracy", 0)
            rows.append(f"{cid} & {overall:.1f}\\% \\\\")

    ncols = len(header.split("&"))
    return (
        "\\begin{table}[ht]\n"
        f"\\caption{{{sweep_name} — QA Accuracy}}\n"
        "\\begin{tabular}{" + "l" + ("c" * (ncols - 1)) + "}\n"
        "\\hline\n"
        + header + "\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 3 QA evaluation results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--no-latex", action="store_true",
                        help="Skip LaTeX table generation")
    args = parser.parse_args()

    # Discover config directories
    config_dirs = sorted(
        d for d in EVAL_DIR.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    )
    if not config_dirs:
        print("[ERROR] No evaluation results found. Run Phase 3 first.")
        return 1

    # Load and summarize all configs
    summary: dict = {}
    for config_dir in config_dirs:
        config_id = config_dir.name
        rows = load_results(config_dir)
        if not rows:
            continue

        summary[config_id] = {
            "overall": compute_accuracy(rows)["overall"],
            "by_question_type": compute_accuracy(rows, "question_type"),
            "by_topic": compute_accuracy(rows, "topic"),
            "by_persona": compute_accuracy(rows, "persona_id"),
        }

    # Print summary to console
    print(f"\n{'='*70}")
    print(f"Summary — {len(summary)} configs evaluated")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"{'-'*44}")
    for cid in sorted(summary.keys()):
        ov = summary[cid]["overall"]
        print(f"{cid:<20} {ov['correct']:>8} {ov['total']:>6} {ov['accuracy']:>9.2f}%")

    # Save JSON summary
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Summary saved to {args.output}")

    if args.no_latex:
        return 0

    # Generate LaTeX tables for each sweep
    sweep_configs = {
        "Consolidation — write_threshold": [
            "baseline", "write_0.56", "write_0.62", "write_0.68", "write_0.74", "write_0.80"],
        "Consolidation — promote_threshold": [
            "baseline", "promote_0.60", "promote_0.66", "promote_0.72", "promote_0.78", "promote_0.84"],
        "Decay lambda scale": [
            "baseline", "decay_0.25x", "decay_0.5x", "decay_1.0x", "decay_2.0x", "decay_4.0x"],
        "Branching — context_threshold": [
            "baseline", "ctx_0.50", "ctx_0.56", "ctx_0.62", "ctx_0.68", "ctx_0.74"],
    }

    latex_path = ROOT / "eval_summary.tex"
    with latex_path.open("w", encoding="utf-8") as f:
        for sweep_name, configs in sweep_configs.items():
            available = [c for c in configs if c in summary]
            if not available:
                continue
            f.write(generate_latex_table(summary, sweep_name, available))
            f.write("\n\n")
    print(f"[DONE] LaTeX tables saved to {latex_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
