#!/usr/bin/env python3
"""Stage 1: Cross-session TPPM memory extraction for LoCoMo Experiment 1 Layer 2.

Pipeline per conversation:
    1. Load LoCoMo 10 conversations from locomo10.json
    2. For each conversation, iterate sessions 1..N sequentially
    3. Per session: async DeepSeek API call to extract ProfileCandidates
    4. Feed candidates into TemporalProfileMemory engine:
       ingest → align/fuse → finish_session (promote) → decay
    5. Save full memory bank JSON for downstream QA/Event eval.

Usage:
    python3 locomo_tppm_extract.py                           # full 10 conversations
    python3 locomo_tppm_extract.py --max-convs 1             # smoke test
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
from mini_agent.tpm.models import ProfileCandidate, utc_now

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table2-data")
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "locomo_memory_bank.json"
DEFAULT_FAILED = ROOT / "logs" / "locomo_extract_failed.jsonl"

# ===== DeepSeek API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-pro"
API_KEY = "REDACTED_DEEPSEEK_KEY"

CONCURRENCY = 8
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0


# ===== Data loading =====

def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("locomo10.json must be a JSON array.")
    return data


def get_sorted_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict], str]]:
    """Extract sorted session info from a conversation.

    Returns:
        list of (session_num, session_key, turns, date_time)
        sorted by session_num ascending.
    """
    conv_data = conv["conversation"]
    sessions: list[tuple[int, str, list[dict], str]] = []
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            num_str = key.replace("session_", "")
            try:
                num = int(num_str)
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


# ===== Async LLM extraction =====

def build_extraction_payload(dialogue_text: str, scene: str = "general") -> dict[str, Any]:
    """Build the extraction prompt matching LLMProfileExtractor schema."""
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
    """Parse LLM JSON response into ProfileCandidate list."""
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
    dialogue_text: str,
    scene: str = "general",
    conv_id: str = "",
    session_num: int = 0,
) -> list[ProfileCandidate]:
    """Async call to DeepSeek for profile extraction with retries."""
    if not dialogue_text.strip():
        return []

    payload = build_extraction_payload(dialogue_text, scene)
    max_token_cap = max(MAX_TOKENS, 4096)

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_max_tokens = min(MAX_TOKENS * (2 ** (attempt - 1)), max_token_cap)
        try:
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


# ===== Cross-session TPPM engine =====

async def process_conversation(
    conv: dict[str, Any],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Run TPPM extraction across all sessions of one conversation.

    Sessions are processed sequentially (memory state evolves), but each session's
    LLM call goes through the shared semaphore for global concurrency control.

    Returns:
        (conv_id, memory_dict, error_message)
    """
    conv_id = conv.get("sample_id", "unknown")
    sessions = get_sorted_sessions(conv)

    if not sessions:
        return conv_id, None, f"no sessions found for {conv_id}"

    tpm = TemporalProfileMemory(TPMConfig())
    error_msg: str | None = None

    for session_num, session_key, turns, date_time in sessions:
        dialogue_text = format_turns_for_extraction(turns)
        scene = f"session_{session_num}"

        tpm.start_session(scene=scene, session_id=f"{conv_id}_{session_key}")

        try:
            async with sem:
                candidates = await extract_candidates_async(
                    client, dialogue_text, scene=scene,
                    conv_id=conv_id, session_num=session_num,
                )
        except Exception as exc:
            DEFAULT_FAILED.parent.mkdir(parents=True, exist_ok=True)
            with DEFAULT_FAILED.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "conv_id": conv_id,
                    "session_num": session_num,
                    "session_key": session_key,
                    "error": repr(exc),
                }, ensure_ascii=False) + "\n")
            error_msg = f"extraction failed at session {session_num}: {exc}"
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
    memory_dict["num_sessions"] = len(sessions)
    return conv_id, memory_dict, error_msg


async def run_extraction(
    conversations: list[dict[str, Any]],
    concurrency: int = CONCURRENCY,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run TPPM extraction across all LoCoMo conversations concurrently."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        process_conversation(conv, client, sem)
        for conv in conversations
    ]

    memory_entries: list[dict[str, Any]] = []
    failed = 0
    total_pmus = 0

    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Extracting TPPM across conversations")
    for coro in progress:
        conv_id, memory_dict, error = await coro
        if memory_dict is not None:
            memory_entries.append(memory_dict)
            total_pmus += (
                len(memory_dict.get("working_memory", []))
                + len(memory_dict.get("short_term_memory", []))
                + len(memory_dict.get("long_term_memory", []))
            )
        if error:
            failed += 1
            tqdm.write(f"[WARN] {conv_id}: {error}")

    return memory_entries, failed, total_pmus


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TPPM memory extraction for LoCoMo Experiment 1 Layer 2.")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Conversations: {len(conversations)}")
    print(f"[INFO] Model: {API_MODEL}")
    print(f"[INFO] Concurrency: {args.concurrency}")

    for i, conv in enumerate(conversations):
        sessions = get_sorted_sessions(conv)
        print(f"  [{i}] {conv['sample_id']}: {len(sessions)} sessions, "
              f"{sum(len(s[2]) for s in sessions)} turns")

    memory_entries, failed, total_pmus = asyncio.run(
        run_extraction(conversations, concurrency=args.concurrency)
    )

    payload = {
        "metadata": {
            "source": "LoCoMo-10",
            "extractor_model": API_MODEL,
            "tpm_config": {
                "write_threshold": TPMConfig().write_threshold,
                "context_threshold": TPMConfig().context_threshold,
                "promote_threshold": TPMConfig().promote_threshold,
                "write_weights": list(TPMConfig().write_weights),
                "decay_lambdas": TPMConfig().decay_lambdas,
            },
            "total_conversations": len(conversations),
            "extracted_conversations": len(memory_entries),
            "failed_conversations": failed,
            "total_pmus": total_pmus,
        },
        "conversations": memory_entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {args.output}")
    print(f"[DONE] Extracted: {len(memory_entries)} conversations, {total_pmus} total PMUs")
    print(f"[DONE] Failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
