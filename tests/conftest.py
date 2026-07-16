"""Pytest configuration shared across the test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_TEMP_CANDIDATES = [
    _ROOT / ".pytest-tmp",
    _ROOT / ".tmp-pytest-local",
    _ROOT / ".tmp-smoke",
]


def _select_writable_temp_dir() -> Path:
    for candidate in _TEMP_CANDIDATES:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            return candidate
        except Exception:
            continue
    raise RuntimeError("No writable local pytest temp directory is available.")


_LOCAL_TEMP_DIR = _select_writable_temp_dir()

for env_name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[env_name] = str(_LOCAL_TEMP_DIR)

tempfile.tempdir = str(_LOCAL_TEMP_DIR)
