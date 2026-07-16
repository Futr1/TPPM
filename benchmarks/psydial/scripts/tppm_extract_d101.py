#!/usr/bin/env python3
"""Offline TPPM memory extraction for PsyDial-D101.

Extracts psychological profile memories from D101 conversations using
messages[:-1] (all turns before the final user message), then saves a
memory bank JSON for generate_responses.py.

Pipeline:
    1. Read PsyDial-D101.json (1278 cases)
    2. For each case, use messages[:-1] as extraction material
    3. Call DeepSeek-V4-Pro API (async, 8-concurrent) to extract scored candidates
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
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/psydial'
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "d101_tppm_memory_bank_v2.json"
DEFAULT_FAILED = ROOT / "logs" / "d101_tppm_failed.jsonl"
DEFAULT_DEBUG = ROOT / "logs" / "d101_tppm_invalid_responses.jsonl"

# ===== Hybrid TPPM Hyperparameters =====
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
API_MODEL = "deepseek-v4-pro"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

MAX_RETRIES = 5
REQUEST_TIMEOUT = 120.0  # v4-pro is slower than v4-flash
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

def build_client(api_key: str = API_KEY, api_base: str = API_BASE) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=REQUEST_TIMEOUT)

# ===== LLM Extraction Core =====

def build_system_prompt() -> str:
    return """
You are a psychological profile memory extractor for a Hybrid TPPM system.

Your only job is feature extraction and scoring. Python code makes the final write decision.

Read the truncated mental-health support dialogue and extract candidate PPMUs for:
- stressor: core pressure source or triggering situation
- affective_state: emotional state or mood pattern
- coping_style: how the user responds, copes, avoids, suppresses, seeks help, etc.

For each candidate, output four scores [0.0, 1.0]:
- r_score: relevance to psychological profile
- e_score: explicitness of evidence
- u_score: utility for future support
- b_score: tendency to persist beyond fleeting utterance

Constraints:
1. Output exactly one JSON object, nothing else.
2. No Markdown, explanations, comments, or trailing commas.
3. Do not copy long raw dialogue. Keep evidence short.
4. No clinical diagnosis labels.
5. If nothing useful, return {"candidates":[]}.

Required schema:
{"candidates":[{"attribute":"stressor|affective_state|coping_style","value":"...","evidence":"...","r_score":0.0,"e_score":0.0,"u_score":0.0,"b_score":0.0}]}
""".strip()

def clean_json_text(content: str) -> str:
    stripped = (content or "").lstrip("﻿").strip()
    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    return stripped

def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)

def first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise LLMJSONParseError("No complete JSON object found.")

def parse_llm_json(content: str) -> dict[str, Any]:
    stripped = clean_json_text(content)
    if not stripped:
        raise EmptyLLMResponseError("Empty content.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        repaired = remove_trailing_commas(stripped)
        if repaired != stripped:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        try:
            return first_json_object(repaired)
        except Exception as e2:
            raise LLMJSONParseError(f"Cannot parse: {str(e)[:200]}") from e2

def normalize_candidate(raw: dict) -> tuple[CandidateMemory | None, str | None]:
    """Validate one raw LLM candidate, compute phi, and assign tier."""
    if not isinstance(raw, dict):
        return None, "not an object"

    attr = str(raw.get("attribute", "")).strip().lower()
    if attr not in VALID_ATTRIBUTES:
        return None, f"invalid attribute {attr!r}"

    value = raw.get("value")
    if not isinstance(value, str) or not value.strip():
        return None, "value must be non-empty string"

    evidence = raw.get("evidence", "")
    if not isinstance(evidence, str):
        return None, "evidence must be string"

    try:
        r = clamp(raw.get("r_score"))
        e = clamp(raw.get("e_score"))
        u = clamp(raw.get("u_score"))
        b = clamp(raw.get("b_score"))
    except Exception as exc:
        return None, str(exc)

    phi = round(compute_phi(r, e, u, b), 6)

    if phi <= CONTEXT_THRESHOLD:
        return None, f"phi={phi:.4f} <= CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}"

    tier = assign_tier(phi)
    return CandidateMemory(attr, value.strip(), evidence.strip(), r, e, u, b, phi, tier), None

def append_invalid_response(*, case_idx: int, model: str, attempt: int,
                            content: str, error: Exception) -> None:
    DEFAULT_DEBUG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": utc_now_iso(),
        "case_idx": case_idx,
        "model": model,
        "attempt": attempt,
        "error": repr(error),
        "content_preview": content[:2000],
    }
    with DEFAULT_DEBUG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

async def get_llm_candidates(
    dialogue_text: str,
    case_idx: int,
    client: AsyncOpenAI,
    model: str = API_MODEL,
    max_retries: int = MAX_RETRIES,
    max_tokens: int = MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Call DeepSeek-V4-Pro API asynchronously to extract scored PPMU candidates."""
    if not dialogue_text.strip():
        return []

    system_prompt = build_system_prompt()
    user_prompt = (
        "Extract scored TPPM candidate memories from the following truncated dialogue.\n"
        "Remember: output JSON only.\n\n"
        f"{dialogue_text}"
    )

    max_token_cap = max(max_tokens, 4096)
    for attempt in range(1, max_retries + 1):
        attempt_max_tokens = min(max_tokens * (2 ** (attempt - 1)), max_token_cap)
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=attempt_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                raise EmptyLLMResponseError("Empty response.")

            payload = parse_llm_json(content)
            if not isinstance(payload, dict) or "candidates" not in payload:
                raise LLMSchemaError("Missing 'candidates' key.")
            return payload["candidates"]
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"LLM extraction failed after {max_retries} attempts.")

# ===== Async Batch Processing =====

async def process_case(
    case_idx: int,
    dialogue_text: str,
    client: AsyncOpenAI,
    model: str = API_MODEL,
) -> tuple[int, list[CandidateMemory]]:
    """Run TPPM extraction for a single D101 case. Returns (case_idx, memories)."""
    raw_candidates = await get_llm_candidates(
        dialogue_text, case_idx, client, model=model,
        max_retries=MAX_RETRIES, max_tokens=MAX_TOKENS,
    )
    accepted: list[CandidateMemory] = []
    for raw in raw_candidates:
        candidate, _reason = normalize_candidate(raw)
        if candidate is not None:
            accepted.append(candidate)
    return case_idx, accepted

async def run_extraction(
    dataset: list[dict[str, Any]],
    model: str = API_MODEL,
    concurrency: int = CONCURRENCY,
    api_key: str = API_KEY,
    api_base: str = API_BASE,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Run async concurrent extraction over all D101 cases.

    Returns:
        memories_out: list of {"case_idx": int, "tppm_memory": [...]} dicts
        skipped: count of cases skipped (insufficient history)
        failed: count of cases that failed after retries
        empty_memory: count of cases where phi all below threshold
    """
    client = build_client(api_key=api_key, api_base=api_base)
    sem = asyncio.Semaphore(concurrency)

    skipped = 0
    failed = [0]   # mutable container for nested function
    empty_memory = [0]
    tasks: list[asyncio.Task] = []

    async def run_one(case_idx: int, dialogue_text: str) -> dict[str, Any] | None:
        async with sem:
            try:
                _, memories = await process_case(case_idx, dialogue_text, client, model)
                if not memories:
                    empty_memory[0] += 1
                return {"case_idx": case_idx, "tppm_memory": [asdict(m) for m in memories]}
            except Exception as exc:
                failed[0] += 1
                DEFAULT_FAILED.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "timestamp": utc_now_iso(),
                    "case_idx": case_idx,
                    "error": repr(exc),
                    "error_type": type(exc).__name__,
                    "dialogue_length": len(dialogue_text),
                    "model": model,
                    "max_retries": MAX_RETRIES,
                }
                with DEFAULT_FAILED.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return None

    for case in dataset:
        case_idx = case["idx"]
        dialogue_text, num_msgs = format_messages_for_extraction(case)
        if dialogue_text is None:
            skipped += 1
            continue
        tasks.append(asyncio.create_task(run_one(case_idx, dialogue_text)))

    print(f"[INFO] Total D101 cases: {len(dataset)}")
    print(f"[INFO] Skipped (insufficient history, < {MIN_HISTORY_TURNS} messages): {skipped}")
    print(f"[INFO] Extraction candidates: {len(tasks)}")
    print(f"[INFO] Concurrency: {concurrency}")

    results = []
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting TPPM memories")
    for coro in progress:
        result = await coro
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r["case_idx"])
    return results, skipped, failed[0], empty_memory[0]

# ===== CLI Entry Point =====

def main() -> int:
    parser = argparse.ArgumentParser(description="TPPM memory extraction for D101 (DeepSeek-V4-Pro).")
    parser.add_argument("--d101", type=Path, default=D101_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    dataset = load_d101(args.d101)
    if args.max_cases:
        dataset = dataset[:args.max_cases]

    print(f"[INFO] Input: {args.d101}")
    print(f"[INFO] Cases: {len(dataset)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] API base: {args.api_base}")
    print(f"[INFO] Max tokens: {args.max_tokens}")
    print(f"[INFO] phi = {ALPHA_1}*r + {ALPHA_2}*e + {ALPHA_3}*u + {ALPHA_4}*b")
    print(f"[INFO] CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}, "
          f"WRITE_THRESHOLD={WRITE_THRESHOLD}, PROMOTE_THRESHOLD={PROMOTE_THRESHOLD}")

    memories_out, skipped, failed, empty_memory = asyncio.run(
        run_extraction(dataset, model=args.model, concurrency=args.concurrency,
                       api_key=args.api_key, api_base=args.api_base)
    )

    total_memories = sum(len(m["tppm_memory"]) for m in memories_out)
    payload = {
        "metadata": {
            "source": "PsyDial-D101",
            "extraction_range": "messages[:-1]",
            "extractor_model": args.model,
            "alphas": {"r": ALPHA_1, "e": ALPHA_2, "u": ALPHA_3, "b": ALPHA_4},
            "context_threshold": CONTEXT_THRESHOLD,
            "write_threshold": WRITE_THRESHOLD,
            "promote_threshold": PROMOTE_THRESHOLD,
            "tier_labels": {
                "context_only": "0.62 < phi <= 0.68",
                "stable": "0.68 < phi <= 0.72",
                "long_term": "phi > 0.72",
            },
            "total_cases": len(dataset),
            "skipped_short_cases": skipped,
            "failed_cases": failed,
            "extracted_cases": len(memories_out),
            "empty_memory_cases": empty_memory,
            "total_memories": total_memories,
            "min_history_turns": MIN_HISTORY_TURNS,
        },
        "memories": memories_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {args.output}")
    print(f"[DONE] Extracted: {len(memories_out)} cases, {total_memories} memories")
    print(f"[DONE] Skipped (short): {skipped}, Failed: {failed}, Empty memories: {empty_memory}")
    if failed:
        print(f"[DONE] Failure log: {args.failed_output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
