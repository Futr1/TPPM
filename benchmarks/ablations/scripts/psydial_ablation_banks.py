#!/usr/bin/env python3
"""Create PsyDial ablation memory banks by post-processing the baseline.

Variants:
    baseline                 — copy existing memory bank as-is
    ablation_consolidation   — all tiers → context_only (consolidation disabled, no promotion)
    ablation_no_evidence     — strip evidence field from all memories
    ablation_no_ltm          — filter out tier=long_term memories

Note: w/o Branching and w/o Temporal Decay reuse baseline
      (PsyDial extraction has no scene branching or temporal decay).

Usage:
    python3 psydial_ablation_banks.py                    # create all variant banks
    python3 psydial_ablation_banks.py --variant ablation_consolidation  # single variant
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# ===== Paths =====
TABLE1 = REPO_ROOT / 'benchmarks/psydial'
ABLATION = REPO_ROOT / 'benchmarks/ablations'

BASELINE_BANK = TABLE1 / "outputs" / "d101_tppm_memory_bank.json"
OUTPUT_DIR = ABLATION / "memory_snapshots" / "psydial"

VARIANTS = [
    "baseline",
    "ablation_consolidation",
    "ablation_no_evidence",
    "ablation_no_ltm",
]

def load_baseline() -> dict:
    if not BASELINE_BANK.exists():
        raise FileNotFoundError(f"Baseline memory bank not found: {BASELINE_BANK}")
    with BASELINE_BANK.open("r", encoding="utf-8") as f:
        return json.load(f)

def create_baseline(bank: dict) -> dict:
    """Copy as-is."""
    return copy.deepcopy(bank)

def create_consolidation(bank: dict) -> dict:
    """w/o Consolidation: all tiers → context_only (consolidation disabled, memories never promote)."""
    result = copy.deepcopy(bank)
    for entry in result.get("memories", []):
        for mem in entry.get("tppm_memory", []):
            mem["tier"] = "context_only"
    result["metadata"]["variant"] = "ablation_consolidation"
    result["metadata"]["note"] = "All tiers forced to context_only (consolidation disabled, no promotion)"
    return result

def create_no_evidence(bank: dict) -> dict:
    """w/o Evidence: strip evidence field from all memories."""
    result = copy.deepcopy(bank)
    for entry in result.get("memories", []):
        for mem in entry.get("tppm_memory", []):
            mem["evidence"] = ""
    result["metadata"]["variant"] = "ablation_no_evidence"
    result["metadata"]["note"] = "Evidence field stripped from all memories"
    return result

def create_no_ltm(bank: dict) -> dict:
    """w/o LTM: filter out tier=long_term memories."""
    result = copy.deepcopy(bank)
    total_before = 0
    total_after = 0
    for entry in result.get("memories", []):
        mems = entry.get("tppm_memory", [])
        total_before += len(mems)
        filtered = [m for m in mems if m.get("tier") != "long_term"]
        entry["tppm_memory"] = filtered
        total_after += len(filtered)
    result["metadata"]["variant"] = "ablation_no_ltm"
    result["metadata"]["note"] = f"Filtered out long_term memories ({total_before} → {total_after})"
    return result

CREATORS = {
    "baseline": create_baseline,
    "ablation_consolidation": create_consolidation,
    "ablation_no_evidence": create_no_evidence,
    "ablation_no_ltm": create_no_ltm,
}

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create PsyDial ablation memory banks from baseline.")
    parser.add_argument("--variant", type=str, default=None,
                        choices=VARIANTS, help="Single variant (default: all)")
    args = parser.parse_args()

    print(f"[INFO] Loading baseline: {BASELINE_BANK}")
    bank = load_baseline()
    n_cases = len(bank.get("memories", []))
    n_mems = sum(len(e.get("tppm_memory", [])) for e in bank.get("memories", []))
    print(f"[INFO] Baseline: {n_cases} cases, {n_mems} total memories")

    variants_to_run = [args.variant] if args.variant else VARIANTS

    for vid in variants_to_run:
        out_dir = OUTPUT_DIR / vid
        out_path = out_dir / "d101_tppm_memory_bank.json"

        if out_path.exists():
            print(f"\n[SKIP] {vid}: already exists at {out_path}")
            continue

        creator = CREATORS[vid]
        result = creator(bank)

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Stats
        n_out = sum(len(e.get("tppm_memory", [])) for e in result.get("memories", []))
        tiers = {}
        for entry in result.get("memories", []):
            for mem in entry.get("tppm_memory", []):
                t = mem.get("tier", "unknown")
                tiers[t] = tiers.get(t, 0) + 1

        print(f"\n[DONE] {vid}: {n_out} memories → {out_path}")
        print(f"  Tiers: {tiers}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
