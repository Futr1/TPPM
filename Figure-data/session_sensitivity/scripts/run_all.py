#!/usr/bin/env python3
"""Run full session count sensitivity analysis pipeline (LongMemEval).

Usage:
    python3 run_all.py                       # full pipeline
    python3 run_all.py --max-questions 5 -N 1 3 5  # quick test
    python3 run_all.py --skip-phase 1        # skip extraction
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DEFAULT_N_VALUES = [1, 5, 10, 15, 20, 30, 48]


def run_phase(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Session count sensitivity pipeline")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--skip-phase", type=int, action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    n_args = ["-N"] + [str(n) for n in args.n_values]
    common = []
    if args.max_questions:
        common += ["--max-questions", str(args.max_questions)]

    for phase, script in [(1, "scripts/phase1_extract.py"),
                          (2, "scripts/phase2_eval_qa.py"),
                          (3, "scripts/phase3_plot.py")]:
        if phase in args.skip_phase:
            print(f"[SKIP] Phase {phase}")
            continue
        rc = run_phase(script, common + n_args + (
            ["--concurrency", str(args.concurrency)] if phase == 1 else []))
        if rc != 0:
            print(f"[ERROR] Phase {phase} failed (exit {rc})")
            return rc

    print(f"\n{'='*60}")
    print("  Done! Outputs:")
    print(f"    Profiles: {ROOT / 'extracted_profiles'}")
    print(f"    Results:  {ROOT / 'eval_results'}")
    print(f"    Figures:  {ROOT / 'figures'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
