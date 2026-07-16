#!/usr/bin/env python3
"""Summarize ablation evaluation results across all conditions.

Reads Phase 3 CSV results for each condition and outputs:
  - Overall accuracy per condition
  - Per-question-type (Q1, Q4, Q5, Q7, etc.) breakdown
  - Degradation deltas relative to Full TPPM

Usage:
    python3 summarize.py
    python3 summarize.py --condition ablation_consolidation  # single condition
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ROOT = REPO_ROOT / 'benchmarks/ablations'
TABLE3_DATA = REPO_ROOT / 'benchmarks/personamem'
EVAL_DIR = ROOT / "eval_results" / "deepseek"
BASELINE_CSV = TABLE3_DATA / "eval_results" / "deepseek" / "baseline" / "results.csv"

# Conditions in display order
CONDITIONS = [
    ("full_tppm", "Full TPPM", BASELINE_CSV),
    ("ablation_consolidation", "w/o Consolidation", EVAL_DIR / "ablation_consolidation" / "results.csv"),
    ("ablation_branching", "w/o Scene Branching", EVAL_DIR / "ablation_branching" / "results.csv"),
    ("ablation_decay", "w/o Temporal Decay", EVAL_DIR / "ablation_decay" / "results.csv"),
    ("ablation_no_evidence", "w/o Evidence Collection", EVAL_DIR / "ablation_no_evidence" / "results.csv"),
    ("ablation_no_ltm", "w/o Long-term Retrieval", EVAL_DIR / "ablation_no_ltm" / "results.csv"),
]

# Question type → display name mapping
TYPE_NAMES = {
    "recall_user_shared_facts": "Q1 Recall",
    "suggest_new_ideas": "Q2 Suggest",
    "track_full_preference_evolution": "Q4 PrefEvo",
    "recalling_the_reasons_behind_previous_updates": "Q5 Reasons",
    "provide_preference_aligned_recommendations": "Q6 Aligned",
    "generalizing_to_new_scenarios": "Q7 Generalize",
    "recalling_facts_mentioned_by_the_user": "Q3 Facts",
}

# Each condition's diagnostic metric
DIAGNOSTIC = {
    "ablation_consolidation": "track_full_preference_evolution",  # Q4
    "ablation_branching": "generalizing_to_new_scenarios",         # Q7
    "ablation_decay": "track_full_preference_evolution",           # Q4 (best available; true diag is LoCoMo Temporal)
    "ablation_no_evidence": "recalling_the_reasons_behind_previous_updates",  # Q5
    "ablation_no_ltm": "track_full_preference_evolution",          # Q4
}


def load_results(csv_path: Path) -> dict[str, dict[str, int]]:
    """Load results CSV → {question_id: {score, question_type, ...}}."""
    results: dict[str, dict[str, int]] = {}
    if not csv_path.exists():
        return results
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question_id"]
            qtype = row["question_type"]
            score = 1 if row.get("score", "").strip().lower() in ("true", "1") else 0
            results[qid] = {"score": score, "question_type": qtype}
    return results


EXCLUDED_TYPES = {"recalling_facts_mentioned_by_the_user"}  # Q3 not used in paper

def compute_accuracy(results: dict[str, dict[str, int]],
                     filter_type: str | None = None) -> tuple[int, int, float]:
    """Compute accuracy, optionally filtering by question type.
    Overall (filter_type=None) excludes Q3 by default.
    """
    filtered = {
        k: v for k, v in results.items()
        if v["question_type"] not in EXCLUDED_TYPES
        and (filter_type is None or v["question_type"] == filter_type)
    }
    if not filtered:
        return 0, 0, 0.0
    correct = sum(v["score"] for v in filtered.values())
    total = len(filtered)
    return correct, total, correct / total * 100


def main() -> int:
    # Load baseline first
    baseline_results = load_results(BASELINE_CSV)
    if not baseline_results:
        print("[WARN] Baseline results not found. Run Phase 3 for full_tppm first.")
        print(f"  Expected at: {BASELINE_CSV}")

    baseline_overall = compute_accuracy(baseline_results)
    baseline_by_type = {
        t: compute_accuracy(baseline_results, t) for t in TYPE_NAMES
    }

    print("=" * 100)
    print("TPPM benchmarks/ablations Experiment — Results Summary")
    print("=" * 100)

    # Overall accuracy table
    print(f"\n{'Condition':<28} {'Overall':>8} {'#Q':>6} {'Δ':>8}  {'Diagnostic':>20} {'Diag Δ':>8}")
    print("-" * 90)

    for cond_id, label, csv_path in CONDITIONS:
        results = load_results(csv_path)
        if not results:
            print(f"  {label:<26} {'N/A':>8}")
            continue

        overall = compute_accuracy(results)
        delta = overall[2] - baseline_overall[2] if baseline_results else 0.0

        # Diagnostic metric for this condition
        diag_type = DIAGNOSTIC.get(cond_id)
        diag = compute_accuracy(results, diag_type) if diag_type else (0, 0, 0.0)
        diag_baseline = baseline_by_type.get(diag_type, (0, 0, 0.0))
        diag_delta = diag[2] - diag_baseline[2] if baseline_results and diag_type else 0.0

        marker = " ←" if delta < -1.0 else ""
        print(f"  {label:<26} {overall[2]:>7.2f}% {overall[1]:>5}  {delta:>+7.2f}%"
              f"  {TYPE_NAMES.get(diag_type, diag_type or '—'):>20} {diag_delta:>+8.2f}%{marker}")

    # Per-question-type breakdown (for conditions that have results)
    print(f"\n{'=' * 100}")
    print("Per-Question-Type Breakdown")
    print(f"{'=' * 100}")

    types_ordered = ["recall_user_shared_facts", "track_full_preference_evolution",
                     "recalling_the_reasons_behind_previous_updates",
                     "provide_preference_aligned_recommendations",
                     "generalizing_to_new_scenarios",
                     "suggest_new_ideas"]

    header = f"{'Condition':<28}"
    for t in types_ordered:
        header += f"  {TYPE_NAMES.get(t, t):>12}"
    print(header)
    print("-" * len(header))

    for cond_id, label, csv_path in CONDITIONS:
        results = load_results(csv_path)
        if not results:
            continue
        line = f"  {label:<26}"
        for t in types_ordered:
            acc = compute_accuracy(results, t)
            line += f"  {acc[2]:>10.2f}%"
        print(line)

    # Delta breakdown vs baseline
    if baseline_results:
        print(f"\n{'=' * 100}")
        print("Delta vs Full TPPM (per question type)")
        print(f"{'=' * 100}")
        header = f"{'Condition':<28}"
        for t in types_ordered:
            header += f"  {TYPE_NAMES.get(t, t):>12}"
        print(header)
        print("-" * len(header))

        for cond_id, label, csv_path in CONDITIONS:
            if cond_id == "full_tppm":
                continue
            results = load_results(csv_path)
            if not results:
                continue
            line = f"  {label:<26}"
            for t in types_ordered:
                acc = compute_accuracy(results, t)
                bl = compute_accuracy(baseline_results, t)
                delta = acc[2] - bl[2] if bl[1] > 0 else 0.0
                line += f"  {delta:>+11.2f}%"
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
