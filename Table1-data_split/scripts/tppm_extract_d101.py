#!/usr/bin/env python3
"""Offline TPPM memory extraction for PsyDial-D101 BERTScore evaluation.

Extracts psychological profile memories from D101 conversations using
messages[:-1] (all turns before the final user message), then saves a
memory bank JSON for eval_bertscore.py --method tppm_memory.

Pipeline:
    1. Read PsyDial-D101.json (1278 cases)
    2. For each case, use messages[:-1] as extraction material
    3. Call DeepSeek API (async, 8-concurrent) to extract scored candidates
    4. Python-side phi scoring → tiered admission (phi > 0.62)
    5. Save filtered memory bank indexed by case_idx

Usage:
    python3 tppm_extract_d101.py                           # full 1278 cases
    python3 tppm_extract_d101.py --max-cases 30            # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table1-data_split")
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "d101_tppm_memory_bank.json"
DEFAULT_FAILED = ROOT / "logs" / "d101_tppm_failed.jsonl"
DEFAULT_DEBUG = ROOT / "logs" / "d101_tppm_invalid_responses.jsonl"

# ===== Hybrid TPPM Hyperparameters (unchanged from tppm_extract.py) =====
ALPHA_1 = 0.25
ALPHA_2 = 0.30
ALPHA_3 = 0.25
ALPHA_4 = 0.20

# ===== Multi-level Thresholds =====
CONTEXT_THRESHOLD = 0.62   # phi > 0.62 → save for context injection
WRITE_THRESHOLD = 0.68     # phi > 0.68 → tier = "stable"
PROMOTE_THRESHOLD = 0.72   # phi > 0.72 → tier = "long_term"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "REDACTED_DEEPSEEK_KEY"

MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
CONCURRENCY = 8
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

MIN_HISTORY_TURNS = 3  # len(messages) must be >= 3 to have history

VALID_ATTRIBUTES = {"stressor", "affective_state", "coping_style"}


class EmptyLLMResponseError(ValueError):
    pass


class LLMJSONParseError(ValueError):
    pass


class LLMSchemaError(ValueError):
    pass


@dataclass(slots=True)
class CandidateMemory:
    attribute: str
    value: str
    evidence: str
    r_score: float
    e_score: float
    u_score: float
    b_score: float
    phi: float
    tier: str  # "context_only" | "stable" | "long_term"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def compute_phi(r: float, e: float, u: float, b: float) -> float:
    return ALPHA_1 * r + ALPHA_2 * e + ALPHA_3 * u + ALPHA_4 * b


def assign_tier(phi: float) -> str:
    """Assign memory tier based on phi value."""
    if phi > PROMOTE_THRESHOLD:
        return "long_term"
    elif phi > WRITE_THRESHOLD:
        return "stable"
    else:
        return "context_only"


def load_d101(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("D101 must be a JSON array.")
    return data


def format_messages_for_extraction(case: dict[str, Any]) -> tuple[str | None, int]:
    """Extract messages[:-1] as dialogue text. Returns (text, num_messages)."""
    msgs = case.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < MIN_HISTORY_TURNS:
        return None, len(msgs) if isinstance(msgs, list) else 0

    # Use all messages except the last (the final user message)
    history = msgs[:-1]

    lines: list[str] = []
    for m in history:
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    if not lines:
        return None, len(msgs)
    return "\n".join(lines), len(msgs)


def build_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
