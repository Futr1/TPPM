#!/usr/bin/env python3
"""DEPRECATED — use llm_judge_scoring.py instead.

This wrapper is kept for backward compatibility only.
It prints a deprecation warning and delegates to llm_judge_scoring.
"""

import sys
import warnings

warnings.warn(
    "gpt4o_scoring.py is deprecated. Use llm_judge_scoring.py instead.",
    DeprecationWarning,
    stacklevel=2,
)
print("[DEPRECATED] gpt4o_scoring.py → delegating to llm_judge_scoring.py", file=sys.stderr)

# Delegate to the new script
from llm_judge_scoring import *  # noqa: E402, F403

if __name__ == "__main__":
    import llm_judge_scoring
    raise SystemExit(llm_judge_scoring.main())
