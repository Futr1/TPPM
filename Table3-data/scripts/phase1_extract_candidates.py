#!/usr/bin/env python3
"""Phase 1: Extract ProfileCandidates from PersonaMem shared contexts via DeepSeek API.

Session boundary detection: role=system messages split the flat message list
into chronological sessions.

Usage:
    python3 phase1_extract_candidates.py                           # all contexts
    python3 phase1_extract_candidates.py --max-contexts 2          # smoke test
    python3 phase1_extract_candidates.py --context-id <hash>       # single context
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# Allow importing Mini-Agent-5-1 TPPM modules
_AGENT_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.models import ProfileCandidate

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table3-data")
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
SHARED_CONTEXTS_PATH = DATASETS / "shared_contexts_32k.jsonl"
CANDIDATES_DIR = ROOT / "candidates"

# ===== DeepSeek API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "REDACTED_DEEPSEEK_KEY"

CONCURRENCY = 8
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 4096
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0


# ===== Data loading =====

def load_shared_contexts(path: Path) -> list[tuple[str, list[dict]]]:
    """Load all shared contexts from JSONL.

    Returns:
        list of (context_hash, messages_list) tuples.
    """
    contexts: list[tuple[str, list[dict]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Each line is {hash: [messages]}
            for key, msgs in obj.items():
                contexts.append((key, msgs))
    return contexts


def detect_sessions(messages: list[dict]) -> list[tuple[int, list[dict]]]:
    """Split flat message list into sessions at role=system boundaries.

    Each system message marks the start of a new session. The system message
    itself is included as the first message of the session.

    Returns:
        list of (session_idx, session_messages) sorted chronologically.
    """
    sessions: list[tuple[int, list[dict]]] = []
    current_session: list[dict] = []

    for msg in messages:
        if msg.get("role") == "system" and current_session:
            # System message starts a new session — save previous
            sessions.append((len(sessions), current_session))
            current_session = [msg]
        else:
            current_session.append(msg)

    # Don't forget the last session
    if current_session:
        sessions.append((len(sessions), current_session))

    return sessions


def format_session_for_extraction(session_messages: list[dict]) -> str:
    """Format session messages into a single dialogue text block for LLM extraction."""
    lines: list[str] = []
    for msg in session_messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            # Skip persona description prefix — keep it brief
            if "persona:" in content.lower() and len(content) > 300:
                continue
            lines.append(f"[Context] {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


# ===== Async LLM extraction =====

def build_extraction_payload(dialogue_text: str, scene: str = "general") -> dict[str, Any]:
    """Build DeepSeek API payload for profile candidate extraction."""
    schema_hint = {
        "candidates": [
            {
                "attribute": "short_attribute_name",
                "value": "profile_value",
                "context": "supporting_span_or_short_reason",
                "profile_type": "background|preference|goal|style|interest|general",
                "scene": scene,
                "confidence": 0.0,
                "stability": 0.0,
                "recency": 1.0,
                "explicitness": 0.0,
                "user_relevance": 0.0,
                "source": "llm_deepseek",
            }
        ]
    }
    system_prompt = (
        "You are a profile candidate extractor for Temporal Profile Memory (TPM). "
        "Extract stable, reusable, and scene-conditioned user profile information "
        "from the latest conversation session. "
        "Return ONLY valid JSON, no markdown, no explanation."
    )
    user_prompt = (
        "Task: extract profile candidates for TPM.\n"
        f"Current scene: {scene}\n"
        f"Latest conversation session:\n{dialogue_text}\n\n"
        "Extraction rules:\n"
        "1. Extract user profile facts, preferences, goals, style tendencies, identity, "
        "or stable context from the USER speaker. Ignore assistant-only content.\n"
        "2. Extract AT LEAST 1-2 candidates per session — every conversation reveals "
        "something about the user's personality, habits, or situation, even if implicit. "
        "Look for: hobbies, taste, habits, emotional patterns, life situations, values.\n"
        "3. Attribute names MUST be compound and specific. Use domain+category format:\n"
        "   GOOD: music_taste, food_preference, social_style, career_goal, exercise_habit\n"
        "   BAD: interest, preference, goal, style (these are profile_types, not attributes)\n"
        "4. profile_type MUST be one of: background, preference, goal, style, interest, general.\n"
        "5. Score calibration (CRITICAL — use the FULL 0-1 range):\n"
        "   confidence: 0.3-0.5=ambiguous hint, 0.5-0.7=moderate signal, 0.7-0.9=clear statement, 0.9-1.0=explicit self-description.\n"
        "   stability: 0.2-0.4=one-time event/mood, 0.4-0.6=short-term goal, 0.6-0.8=pattern across sessions, 0.8-1.0=core identity.\n"
        "   recency: 0.3-0.5=mentioned long ago, 0.5-0.7=mentioned earlier in session, 0.7-0.9=recently discussed, 0.9-1.0=actively being discussed.\n"
        "   explicitness: 0.2-0.4=implied from context, 0.4-0.7=indirect expression, 0.7-0.9=stated preference, 0.9-1.0=explicitly declared.\n"
        "   user_relevance: 0.2-0.5=minor detail, 0.5-0.7=notable trait, 0.7-0.9=central to profile, 0.9-1.0=defining characteristic.\n"
        "6. VARY scores across candidates — not everything should be 0.8-1.0. Use lower scores for subtle or one-off observations.\n\n"
        f"Output JSON schema example:\n{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    return {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def parse_candidates_from_response(
    content: str, scene: str, original_text: str
) -> list[dict[str, Any]]:
    """Parse LLM JSON response into candidate dicts.

    Returns list of dicts with ProfileCandidate-compatible fields.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, dict):
        raw_list = parsed.get("candidates", [])
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        return []

    def _clamp(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _default_stability(ptype: str) -> float:
        defaults = {
            "background": 0.9, "style": 0.78, "preference": 0.72,
            "interest": 0.7, "goal": 0.56, "general": 0.6,
        }
        return defaults.get(ptype, 0.6)

    candidates: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        attr = str(item.get("attribute", "")).strip()
        val = str(item.get("value", "")).strip()
        if not attr or not val:
            continue
        ptype = str(item.get("profile_type", "general")).strip().lower()
        if ptype not in {"background", "preference", "goal", "style", "interest", "general"}:
            ptype = "general"
        candidates.append({
            "attribute": attr,
            "value": val,
            "context": str(item.get("context") or original_text).strip() or original_text,
            "profile_type": ptype,
            "scene": str(item.get("scene") or scene).strip() or scene,
            "confidence": _clamp(item.get("confidence"), 0.72),
            "stability": _clamp(item.get("stability"), _default_stability(ptype)),
            "recency": _clamp(item.get("recency"), 1.0),
            "explicitness": _clamp(item.get("explicitness"), 0.8),
            "user_relevance": _clamp(item.get("user_relevance"), 0.82),
            "source": str(item.get("source") or "llm_deepseek").strip() or "llm_deepseek",
        })
    return candidates


async def extract_candidates_async(
    client: AsyncOpenAI,
    dialogue_text: str,
    scene: str = "general",
    context_hash: str = "",
    session_idx: int = 0,
) -> list[dict[str, Any]]:
    """Async call to DeepSeek for profile extraction with retries."""
    if not dialogue_text.strip():
        return []

    payload = build_extraction_payload(dialogue_text, scene)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=API_MODEL,
                temperature=0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=payload["messages"],
            )
            content = resp.choices[0].message.content or ""
            if not content.strip():
                continue
            candidates = parse_candidates_from_response(content, scene, dialogue_text)
            return candidates[:8]
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    return []


# ===== Per-context processing =====

async def process_context(
    context_hash: str,
    messages: list[dict],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Extract candidates for all sessions in one shared context.

    Returns:
        (context_hash, num_sessions, num_candidates_extracted)
    """
    sessions = detect_sessions(messages)
    if not sessions:
        return context_hash, 0, 0

    output_dir = CANDIDATES_DIR / context_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    total_candidates = 0

    for session_idx, session_msgs in sessions:
        dialogue_text = format_session_for_extraction(session_msgs)
        scene = f"session_{session_idx}"

        try:
            async with sem:
                candidates = await extract_candidates_async(
                    client, dialogue_text, scene=scene,
                    context_hash=context_hash, session_idx=session_idx,
                )
        except Exception as exc:
            tqdm.write(f"[ERROR] {context_hash[:8]} session {session_idx}: {exc}")
            candidates = []

        output_path = output_dir / f"session_{session_idx:03d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "context_hash": context_hash,
                "session_idx": session_idx,
                "session_id": f"{context_hash}_session_{session_idx}",
                "scene": scene,
                "dialogue_text": dialogue_text,
                "candidates": candidates,
            }, f, ensure_ascii=False, indent=2)

        total_candidates += len(candidates)

    return context_hash, len(sessions), total_candidates


# ===== Main runner =====

async def run_extraction(
    contexts: list[tuple[str, list[dict]]],
    concurrency: int = CONCURRENCY,
) -> tuple[int, int, int]:
    """Run candidate extraction across all contexts concurrently."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        process_context(ctx_hash, msgs, client, sem)
        for ctx_hash, msgs in contexts
    ]

    total_contexts = 0
    total_sessions = 0
    total_candidates = 0

    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Extracting TPPM candidates")
    for coro in progress:
        ctx_hash, num_sessions, num_candidates = await coro
        total_contexts += 1
        total_sessions += num_sessions
        total_candidates += num_candidates
        progress.set_postfix({
            "ctx": ctx_hash[:8],
            "sessions": num_sessions,
            "cands": num_candidates,
        })

    return total_contexts, total_sessions, total_candidates


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract TPPM candidates from PersonaMem shared contexts")
    parser.add_argument("--input", type=Path, default=SHARED_CONTEXTS_PATH)
    parser.add_argument("--max-contexts", type=int, default=None,
                        help="Limit number of contexts (for smoke testing)")
    parser.add_argument("--context-id", type=str, default=None,
                        help="Process a single context by hash prefix")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    contexts = load_shared_contexts(args.input)
    if args.context_id:
        contexts = [(h, m) for h, m in contexts if h.startswith(args.context_id)]
    if args.max_contexts:
        contexts = contexts[:args.max_contexts]

    total_sessions = sum(len(detect_sessions(msgs)) for _, msgs in contexts)
    print(f"[INFO] Contexts: {len(contexts)}")
    print(f"[INFO] Total sessions: {total_sessions}")
    print(f"[INFO] Model: {API_MODEL}")
    print(f"[INFO] Concurrency: {args.concurrency}")

    n_ctx, n_sessions, n_cands = asyncio.run(
        run_extraction(contexts, concurrency=args.concurrency)
    )

    print(f"\n[DONE] Processed {n_ctx} contexts, {n_sessions} sessions, {n_cands} candidates")
    print(f"[DONE] Output: {CANDIDATES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
