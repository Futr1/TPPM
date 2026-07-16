#!/usr/bin/env python3
"""Summarize LoCoMo ablation QA results across all variants."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ROOT = REPO_ROOT / 'benchmarks/ablations'
EVAL_DIR = ROOT / "eval_results" / "locomo"

# Baseline uses the existing benchmarks/locomo result for reference
TABLE2_BASELINE = REPO_ROOT / 'benchmarks/locomo'/outputs/locomo_qa_results.json")

VARIANTS = [
    ("baseline", "Full TPPM"),
    ("ablation_consolidation", "w/o Consolidation"),
    ("ablation_branching", "w/o Scene Branching"),
    ("ablation_decay", "w/o Temporal Decay"),
    ("ablation_no_evidence", "w/o Evidence Collection"),
    ("ablation_no_ltm", "w/o Long-term Retrieval"),
]

CATEGORIES = ["multi_hop", "single_hop", "temporal", "open_domain", "adversarial", "overall"]


def load_results(variant_id: str) -> dict | None:
    path = EVAL_DIR / variant_id / "qa_results.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print("=" * 90)
    print("LoCoMo benchmarks/ablations — QA Results Summary")
    print("=" * 90)

    # Load Table2 baseline for reference
    tbl2_summary = None
    if TABLE2_BASELINE.exists():
        with TABLE2_BASELINE.open() as f:
            tbl2 = json.load(f)
        tbl2_summary = tbl2.get("summary", {})

    # Header
    header = f"{'Variant':<28}"
    for cat in CATEGORIES:
        label = cat.replace("_", " ").title()
        if cat == "overall":
            label = "Overall"
        header += f"  {label:>12}"
    print(header)
    print("-" * (28 + 14 * len(CATEGORIES)))

    baseline_summary = None

    for vid, label in VARIANTS:
        data = load_results(vid)
        if data is None and vid == "baseline":
            # Use Table2 baseline as reference
            if tbl2_summary:
                line = f"  {label:<26}"
                for cat in CATEGORIES:
                    val = tbl2_summary.get(cat, 0.0)
                    line += f"  {val:>11.1f}%"
                print(line + "  (Table2)")
                baseline_summary = tbl2_summary
            continue

        if data is None:
            print(f"  {label:<26}  {'N/A':>12}" * 1)
            continue

        summary = data.get("summary", {})
        if vid == "baseline":
            baseline_summary = summary

        line = f"  {label:<26}"
        for cat in CATEGORIES:
            val = summary.get(cat, 0.0)
            line += f"  {val:>11.1f}%"
        print(line)

    # Delta table
    if baseline_summary:
        print(f"\n{'=' * 90}")
        print("Delta vs Full TPPM")
        print(f"{'=' * 90}")
        header = f"{'Variant':<28}"
        for cat in CATEGORIES:
            label_cat = cat.replace("_", " ").title()
            if cat == "overall":
                label_cat = "Overall"
            header += f"  {label_cat:>12}"
        print(header)
        print("-" * (28 + 14 * len(CATEGORIES)))

        for vid, label in VARIANTS:
            if vid == "baseline":
                continue
            data = load_results(vid)
            if data is None:
                continue
            summary = data.get("summary", {})
            line = f"  {label:<26}"
            for cat in CATEGORIES:
                val = summary.get(cat, 0.0)
                bl_val = baseline_summary.get(cat, 0.0)
                delta = val - bl_val
                marker = " ←" if cat == "temporal" and vid == "ablation_decay" and delta < -1.0 else ""
                line += f"  {delta:>+11.1f}%{marker}"
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
