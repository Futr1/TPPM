#!/usr/bin/env python3
"""Phase 1: Session-truncated TPPM profile extraction for LongMemEval.

For each N in {1,3,5,10,15,20,30,48}, truncate each question's haystack
to the first N sessions, then extract user profile via LLM.

Usage:
    python3 phase1_extract.py                       # full run
    python3 phase1_extract.py --max-questions 5 -N 1 3 5  # quick test
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DATA_PATH = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity/sampled_100.json")
OUTPUT_DIR = ROOT / "extracted_profiles"
FAILED_LOG = ROOT / "logs" / "phase1_failed.jsonl"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

CONCURRENCY = 20
MAX_RETRIES = 5
REQUEST_TIMEOUT = 120.0
MAX_TOKENS = 8192

DEFAULT_N_VALUES = [1, 5, 10, 15, 20, 30, 48]

EXTRACTION_PROMPT = """You are a profile extractor. Below are conversation sessions between a user and an assistant, ordered chronologically.

Your task: extract ALL factual information from these conversations, covering BOTH the USER and the ASSISTANT. For each fact, record:
- What the fact is (attribute and value)
- Which session it appeared in (session index, 1-based)
- Who the information is about: "user" or "assistant"
- Your confidence (0-1)

## USER-side information to extract:
- Personal details (name, age, education, job, location, etc.)
- Preferences and opinions
- Events with dates/times
- Knowledge or facts the user has shared
- Emotional states, mental health conditions, stressors
- Changes in any of the above over time (note when a fact updates across sessions)

## ASSISTANT-side information to extract:
- Therapeutic advice, suggestions, or recommendations given
- Assessments or observations the assistant made about the user
- Homework, exercises, or tasks assigned to the user
- Information or psychoeducation provided by the assistant
- Diagnoses, treatment plans, or referrals mentioned
- Any factual statements the assistant made that could later be queried

## CRITICAL: Temporal tracking for facts that change over time
When the SAME attribute changes value across sessions (e.g., medication, mood, job status),
extract EACH version separately with its own session number, so the timeline is preserved.

Output as JSON array:
[{{"attribute": "...", "value": "...", "session": N, "source": "user"|"assistant", "confidence": 0.X}}, ...]

Be thorough — extract EVERY piece of information, no matter how small.
If a session contains no extractable information, skip it.

Conversations:
{conversations}"""


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_sessions(haystack_sessions: list[list[dict]], n: int) -> str:
    """Format first N sessions as readable text."""
    parts = []
    for i, session in enumerate(haystack_sessions[:n]):
        parts.append(f"\n=== Session {i+1} ===")
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            # Truncate very long turns
            if len(content) > 2000:
                content = content[:2000] + "..."
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


async def extract_profile(
    client: AsyncOpenAI,
    question_id: str,
    haystack_sessions: list[list[dict]],
    n: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Extract profile from first N sessions."""
    output_path = OUTPUT_DIR / f"{question_id}_N{n}.json"
    if output_path.exists():
        with output_path.open("r") as f:
            return json.load(f)

    conv_text = format_sessions(haystack_sessions, n)
    prompt = EXTRACTION_PROMPT.format(conversations=conv_text)

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=MAX_TOKENS,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            raw = resp.choices[0].message.content.strip()

            # Try to parse JSON from response
            profile_items = _parse_json(raw)
            result = {
                "question_id": question_id,
                "N": n,
                "n_sessions_total": len(haystack_sessions),
                "n_sessions_used": n,
                "profile_items": profile_items,
                "raw_response": raw,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result

        except Exception as e:
            delay = min(2 ** attempt, 30)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
            else:
                FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
                with FAILED_LOG.open("a") as f:
                    f.write(json.dumps({
                        "question_id": question_id, "N": n,
                        "error": str(e),
                    }) + "\n")
                return None


def _parse_json(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling code blocks."""
    import re
    # Strip markdown code blocks
    cleaned = text
    for fence in ['```json', '```']:
        if fence in cleaned:
            # Extract content between code fences
            parts = cleaned.split(fence)
            if len(parts) >= 3:
                cleaned = parts[1]
                break
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "error" not in result:
            return [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try to fix truncated JSON by closing open brackets
    truncated = cleaned.rstrip()
    if truncated and truncated[0] == '[':
        # Count brackets
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        # Try to close it
        fixed = truncated
        # Remove trailing incomplete item
        last_brace = fixed.rfind('}')
        if last_brace > 0:
            fixed = fixed[:last_brace+1]
        fixed += ']'
        try:
            result = json.loads(fixed)
            if isinstance(result, list) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            pass

    # Return raw text
    return [{"raw": cleaned}]


async def main_async(n_values: list[int], max_questions: int | None, concurrency: int):
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    sem = asyncio.Semaphore(concurrency)

    data = load_dataset(DATA_PATH)
    if max_questions:
        data = data[:max_questions]

    print(f"Loaded {len(data)} questions, N values: {n_values}")
    print(f"Total tasks: {len(data) * len(n_values)}")

    tasks = []
    for entry in data:
        qid = entry["question_id"]
        sessions = entry["haystack_sessions"]
        for n in n_values:
            if n > len(sessions):
                continue  # skip N larger than available sessions
            tasks.append(extract_profile(client, qid, sessions, n, sem))

    results = []
    with tqdm(total=len(tasks), desc="Phase 1: Extract") as pbar:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            pbar.update(1)

    succeeded = sum(1 for r in results if r is not None)
    print(f"\nDone: {succeeded}/{len(tasks)} succeeded, {len(tasks)-succeeded} failed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: TPPM profile extraction")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    asyncio.run(main_async(args.n_values, args.max_questions, args.concurrency))


if __name__ == "__main__":
    main()
