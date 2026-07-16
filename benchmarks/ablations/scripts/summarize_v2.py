#!/usr/bin/env python3
"""Summarize ablation evaluation results across all conditions (v2 — includes fine-grained & hierarchy).

Reads Phase 3 CSV results for each condition and outputs:
  - Overall accuracy per condition
  - Per-question-type (Evolution, General., Reasons, Recall, etc.) breakdown
  - Degradation deltas relative to Full TPPM

Usage:
    python3 summarize.py
    python3 summarize.py --section fine_grained    # only fine-grained mechanism ablation
    python3 summarize.py --section hierarchy        # only hierarchy ablation
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

# ===== Section 1: Existing ablation (from original paper) =====
EXISTING_CONDITIONS = [
    ("full_tppm", "Full TPPM", BASELINE_CSV),
    ("ablation_consolidation", "w/o Consolidation", EVAL_DIR / "ablation_consolidation" / "results.csv"),
    ("ablation_branching", "w/o Scene Branching", EVAL_DIR / "ablation_branching" / "results.csv"),
    ("ablation_no_evidence", "w/o Evidence Collection", EVAL_DIR / "ablation_no_evidence" / "results.csv"),
]

# ===== Section 2: Fine-grained mechanism ablation =====
FINE_GRAINED_CONDITIONS = [
    ("full_tppm", "Full TPPM", BASELINE_CSV),
    ("ablation_uniform_decay", "w/o Type-Cond. Decay", EVAL_DIR / "ablation_uniform_decay" / "results.csv"),
    ("ablation_semantic_retrieval", "Semantic-Only Retr.", EVAL_DIR / "ablation_semantic_retrieval" / "results.csv"),
]

# ===== Section 3: Hierarchy ablation =====
HIERARCHY_CONDITIONS = [
    ("ablation_flat_pool", "Flat PPMU Pool", EVAL_DIR / "ablation_flat_pool" / "results.csv"),
    ("ablation_two_level", "Two-Level Memory", EVAL_DIR / "ablation_two_level" / "results.csv"),
    ("full_tppm", "Three-Level TPPM", BASELINE_CSV),
]

# Question type → display name mapping (paper-friendly)
TYPE_NAMES = {
    "recall_user_shared_facts": "Recall",
    "suggest_new_ideas": "Suggest",
    "track_full_preference_evolution": "Evolution",
    "recalling_the_reasons_behind_previous_updates": "Reasons",
    "provide_preference_aligned_recommendations": "Aligned",
    "generalizing_to_new_scenarios": "General.",
    "recalling_facts_mentioned_by_the_user": "Facts",
}

# Key types for paper table
KEY_TYPES = [
    "track_full_preference_evolution",
    "generalizing_to_new_scenarios",
    "recalling_the_reasons_behind_previous_updates",
    "recall_user_shared_facts",
]

EXCLUDED_TYPES = {"recalling_facts_mentioned_by_the_user"}  # Q3 not used


def load_results(csv_path: Path) -> dict[str, dict[str, int]]:
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


def compute_accuracy(results: dict[str, dict[str, int]],
                     filter_type: str | None = None) -> tuple[int, int, float]:
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


def print_section(title: str, conditions: list[tuple[str, str, Path]],
                  baseline_results: dict | None = None) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")

    if baseline_results is None:
        # Load baseline from first condition with full_tppm
        for cond_id, _, csv_path in conditions:
            if cond_id == "full_tppm":
                baseline_results = load_results(csv_path)
                break

    baseline_overall = compute_accuracy(baseline_results) if baseline_results else (0, 0, 0.0)
    baseline_by_type = {t: compute_accuracy(baseline_results, t) for t in KEY_TYPES}

    # Paper-style table: Variant | Evolution | General. | Reasons | Recall | Overall | Δ
    print(f"\n{'Variant':<28} {'Evolution':>10} {'General.':>10} {'Reasons':>10} {'Recall':>10} {'Overall':>10} {'Δ':>8}")
    print("-" * 96)

    for cond_id, label, csv_path in conditions:
        results = load_results(csv_path)
        if not results:
            print(f"  {label:<26} {'N/A':>10}")
            continue

        overall = compute_accuracy(results)
        delta = overall[2] - baseline_overall[2] if baseline_results else 0.0

        by_type = {}
        for t in KEY_TYPES:
            by_type[t] = compute_accuracy(results, t)

        evo = by_type.get("track_full_preference_evolution", (0, 0, 0.0))[2]
        gen = by_type.get("generalizing_to_new_scenarios", (0, 0, 0.0))[2]
        rea = by_type.get("recalling_the_reasons_behind_previous_updates", (0, 0, 0.0))[2]
        rec = by_type.get("recall_user_shared_facts", (0, 0, 0.0))[2]

        print(f"  {label:<26} {evo:>9.2f}% {gen:>9.2f}% {rea:>9.2f}% {rec:>9.2f}% {overall[2]:>9.2f}% {delta:>+7.2f}%")

    # Delta table
    if baseline_results:
        print(f"\n{'Variant':<28} {'Evolution':>10} {'General.':>10} {'Reasons':>10} {'Recall':>10} {'Overall':>10}")
        print("-" * 80)
        for cond_id, label, csv_path in conditions:
            if cond_id == "full_tppm":
                continue
            results = load_results(csv_path)
            if not results:
                continue

            overall = compute_accuracy(results)
            delta_overall = overall[2] - baseline_overall[2]

            deltas = {}
            for t in KEY_TYPES:
                acc = compute_accuracy(results, t)
                bl = compute_accuracy(baseline_results, t)
                deltas[t] = acc[2] - bl[2] if bl[1] > 0 else 0.0

            print(f"  {label:<26} "
                  f"{deltas.get('track_full_preference_evolution', 0.0):>+9.2f}% "
                  f"{deltas.get('generalizing_to_new_scenarios', 0.0):>+9.2f}% "
                  f"{deltas.get('recalling_the_reasons_behind_previous_updates', 0.0):>+9.2f}% "
                  f"{deltas.get('recall_user_shared_facts', 0.0):>+9.2f}% "
                  f"{delta_overall:>+9.2f}%")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Summarize ablation results (v2)")
    parser.add_argument("--section", choices=["existing", "fine_grained", "hierarchy", "all"],
                        default="all", help="Which section to display")
    args = parser.parse_args()

    baseline_results = load_results(BASELINE_CSV)
    if not baseline_results:
        print("[WARN] Baseline results not found. Run Phase 3 for full_tppm first.")
        print(f"  Expected at: {BASELINE_CSV}")

    if args.section in ("existing", "all"):
        print_section(
            "Section 1: Existing benchmarks/ablations Results (Consolidation / Branching / Evidence)",
            EXISTING_CONDITIONS, baseline_results)

    if args.section in ("fine_grained", "all"):
        print_section(
            "Section 2: Fine-Grained Mechanism benchmarks/ablations (Type-Conditioned Decay / Semantic Retrieval)",
            FINE_GRAINED_CONDITIONS, baseline_results)

    if args.section in ("hierarchy", "all"):
        print_section(
            "Section 3: Memory Hierarchy benchmarks/ablations (Flat / Two-Level / Three-Level)",
            HIERARCHY_CONDITIONS, baseline_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
