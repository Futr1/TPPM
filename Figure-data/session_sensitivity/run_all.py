#!/usr/bin/env python3
"""Run all phases of the session count sensitivity analysis.

Usage:
    python3 run_all.py                       # full pipeline
    python3 run_all.py --max-convs 2 -N 1 3 5  # quick test
    python3 run_all.py --skip-phase 1        # skip extraction (use cached)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DEFAULT_N_VALUES = [1, 3, 5, 7, 10, 15, 20]


def run_phase(script: str, extra_args: list[str]) -> int:
    """Run a phase script and return its exit code."""
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full session count sensitivity pipeline.")
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+",
                        default=DEFAULT_N_VALUES)
    parser.add_argument("--skip-phase", type=int, action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    n_args = [f"-N"] + [str(n) for n in args.n_values]
    common_args = []
    if args.max_convs:
        common_args += ["--max-convs", str(args.max_convs)]

    # Phase 1: Extraction
    if 1 not in args.skip_phase:
        rc = run_phase("phase1_extract.py",
                       common_args + n_args + ["--concurrency", str(args.concurrency)])
        if rc != 0:
            print(f"[ERROR] Phase 1 failed with exit code {rc}")
            return rc
    else:
        print("[SKIP] Phase 1 (extraction)")

    # Phase 2: QA Evaluation
    if 2 not in args.skip_phase:
        rc = run_phase("phase2_eval_qa.py", common_args + n_args)
        if rc != 0:
            print(f"[ERROR] Phase 2 failed with exit code {rc}")
            return rc
    else:
        print("[SKIP] Phase 2 (evaluation)")

    # Phase 3: Plotting
    if 3 not in args.skip_phase:
        rc = run_phase("phase3_plot.py")
        if rc != 0:
            print(f"[ERROR] Phase 3 failed with exit code {rc}")
            return rc
    else:
        print("[SKIP] Phase 3 (plotting)")

    print(f"\n{'='*60}")
    print(f"  All phases complete!")
    print(f"  Profiles: {ROOT / 'extracted_profiles'}")
    print(f"  Results:  {ROOT / 'eval_results'}")
    print(f"  Figures:  {ROOT / 'figures'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
