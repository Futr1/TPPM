#!/usr/bin/env python3
"""Phase 1: Session-truncated TPPM memory extraction for LoCoMo.

For each N in {1,3,5,7,10,15,20}, truncate each conversation to the
first N sessions, then run TPPM extraction.  The output is one JSON
file per (conv_id, N) pair.

Usage:
    python3 phase1_extract.py                       # full run
    python3 phase1_extract.py --max-convs 2 -N 1 3 5  # quick test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# Allow importing Mini-Agent-5-1 TPPM modules
_AGENT_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.memory import TemporalProfileMemory, TPMConfig
from mini_agent.tpm.models import ProfileCandidate

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
OUTPUT_DIR = ROOT / "extracted_profiles"
FAILED_LOG = ROOT / "logs" / "phase1_failed.jsonl"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "REDACTED_DEEPSEEK_KEY"

CONCURRENCY = 8
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

DEFAULT_N_VALUES = [1, 3, 5, 7, 10, 15, 20]


# ===== Data loading =====

def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("locomo10.json must be a JSON array.")
    return data


def get_sorted_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict], str]]:
    """Extract sorted sessions from a conversation.

    Returns:
        list of (session_num, session_key, turns, date_time) sorted ascending.
    """
    conv_data = conv["conversation"]
    sessions: list[tuple[int, str, list[dict], str]] = []
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.replace("session_", ""))
            except ValueError:
                continue
            turns = conv_data[key]
            dt_key = f"session_{num}_date_time"
            date_time = conv_data.get(dt_key, "")
            sessions.append((num, key, turns, date_time))
    sessions.sort(key=lambda x: x[0])
    return sessions


def format_turns_for_extraction(turns: list[dict]) -> str:
    """Format a session's turn list into a single text block."""
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if speaker and text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# ===== LLM extraction (reused from locomo_tppm_extract.py) =====

def build_extraction_payload(dialogue_text: str, scene: str = "general") -> dict[str, Any]:
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
        "1. Keep only user-related profile facts, preferences, goals, style tendencies, "
        "identity, or stable context. Focus on information about the speakers (not the assistant).\n"
        "2. Ignore generic conversational filler and greetings.\n"
        "3. Use concise attribute names like identity, interest, preference, "
        "current_goal, style, project_focus, personal_background.\n"
        "4. profile_type must be one of: background, preference, goal, style, interest, general.\n"
        "5. confidence, stability, recency, explicitness, user_relevance must be numbers in [0,1].\n"
        "6. user_relevance measures how central this fact is to the user's enduring profile.\n"
        "7. Prefer higher stability for repeated or enduring traits; "
        "lower stability for short-term goals.\n"
        "8. If there is no useful profile memory candidate, return {\"candidates\": []}.\n\n"
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
) -> list[ProfileCandidate]:
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

    candidates: list[ProfileCandidate] = []
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
        candidates.append(ProfileCandidate(
            attribute=attr,
            value=val,
            context=str(item.get("context") or original_text).strip() or original_text,
            profile_type=ptype,
            scene=str(item.get("scene") or scene).strip() or scene,
            confidence=_clamp(item.get("confidence"), 0.72),
            stability=_clamp(item.get("stability"), _default_stability(ptype)),
            recency=_clamp(item.get("recency"), 1.0),
            explicitness=_clamp(item.get("explicitness"), 0.8),
            user_relevance=_clamp(item.get("user_relevance"), 0.82),
            source=str(item.get("source") or "llm_deepseek").strip() or "llm_deepseek",
        ))
    return candidates


async def extract_candidates_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    dialogue_text: str,
    scene: str = "general",
) -> list[ProfileCandidate]:
    """Async call to LLM for profile extraction with retries."""
    if not dialogue_text.strip():
        return []

    payload = build_extraction_payload(dialogue_text, scene)
    max_token_cap = max(MAX_TOKENS, 4096)

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_max_tokens = min(MAX_TOKENS * (2 ** (attempt - 1)), max_token_cap)
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=API_MODEL,
                    temperature=0,
                    max_tokens=attempt_max_tokens,
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


# ===== Core: process one (conv, N) pair =====

async def process_conv_at_n(
    conv: dict[str, Any],
    n_sessions: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run TPPM extraction on the first n_sessions of a conversation."""
    conv_id = conv.get("sample_id", "unknown")
    all_sessions = get_sorted_sessions(conv)

    # Truncate to first N sessions
    truncated = all_sessions[:n_sessions]
    if not truncated:
        return {"conv_id": conv_id, "n_sessions": 0, "error": "no sessions"}

    tpm = TemporalProfileMemory(TPMConfig())

    for session_num, session_key, turns, date_time in truncated:
        dialogue_text = format_turns_for_extraction(turns)
        scene = f"session_{session_num}"

        tpm.start_session(scene=scene, session_id=f"{conv_id}_{session_key}")

        try:
            candidates = await extract_candidates_async(
                client, sem, dialogue_text, scene=scene,
            )
        except Exception as exc:
            FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FAILED_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "conv_id": conv_id, "n": n_sessions,
                    "session_num": session_num, "error": repr(exc),
                }, ensure_ascii=False) + "\n")
            candidates = []

        if candidates:
            tpm.ingest_candidates(
                candidates, scene=scene,
                session_id=f"{conv_id}_{session_key}",
            )

        tpm.finish_session(scene=scene)

    # Run long-term decay after all sessions
    tpm.decay_long_term()

    memory_dict = tpm.to_dict()
    memory_dict["conv_id"] = conv_id
    memory_dict["n_sessions"] = n_sessions
    memory_dict["actual_sessions_processed"] = len(truncated)
    return memory_dict


# ===== Orchestration =====

async def run_all(
    conversations: list[dict[str, Any]],
    n_values: list[int],
    concurrency: int = CONCURRENCY,
) -> None:
    """Run extraction for all (conv, N) pairs."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(concurrency)

    # Build task list: one per (conv_idx, N)
    tasks: list[tuple[int, int]] = []
    for ci, conv in enumerate(conversations):
        all_sessions = get_sorted_sessions(conv)
        max_n = len(all_sessions)
        for n in n_values:
            actual_n = min(n, max_n)
            tasks.append((ci, actual_n))

    print(f"[INFO] Total tasks: {len(tasks)} "
          f"({len(conversations)} conversations × {len(n_values)} N values)")

    # Process sequentially per conversation (memory state is sequential),
    # but use sem for global API concurrency
    for ci, conv in enumerate(conversations):
        conv_id = conv.get("sample_id", f"conv-{ci}")
        all_sessions = get_sorted_sessions(conv)
        max_n = len(all_sessions)

        for n in n_values:
            actual_n = min(n, max_n)
            out_path = OUTPUT_DIR / f"{conv_id}_N{actual_n}.json"

            if out_path.exists():
                print(f"  [SKIP] {conv_id} N={actual_n} (exists)")
                continue

            print(f"  [RUN ] {conv_id} N={actual_n} ...", end=" ", flush=True)
            try:
                result = await process_conv_at_n(conv, actual_n, client, sem)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                total_pmus = (
                    len(result.get("working_memory", []))
                    + len(result.get("short_term_memory", []))
                    + len(result.get("long_term_memory", []))
                )
                print(f"OK ({total_pmus} PMUs)")
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"\n[DONE] Phase 1 extraction complete. Output: {OUTPUT_DIR}")


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: Session-truncated TPPM extraction for LoCoMo.")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+",
                        default=DEFAULT_N_VALUES)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Conversations: {len(conversations)}")
    print(f"[INFO] N values: {args.n_values}")
    print(f"[INFO] Model: {API_MODEL}")

    for i, conv in enumerate(conversations):
        sessions = get_sorted_sessions(conv)
        print(f"  [{i}] {conv['sample_id']}: {len(sessions)} sessions")

    asyncio.run(run_all(conversations, args.n_values, args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
