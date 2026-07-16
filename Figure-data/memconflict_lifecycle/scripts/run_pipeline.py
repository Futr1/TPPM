#!/usr/bin/env python3
"""Experiment 4: TPPM Memory Lifecycle Longitudinal Analysis on MemConflict.

Script 1: Process all sessions for each persona sequentially.
For each session: LLM extract candidates -> TPPM ingest/evolve -> dump trace -> answer questions.

Usage:
    python3 run_pipeline.py                    # all personas
    python3 run_pipeline.py --max-personas 1   # quick test
    python3 run_pipeline.py --persona-idx 0    # specific persona
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

# ===== Add TPPM library to path =====
TPPM_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
sys.path.insert(0, str(TPPM_ROOT))

from mini_agent.tpm.memory import TemporalProfileMemory, TPMConfig
from mini_agent.tpm.models import ProfileCandidate

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/memconflict_lifecycle")
DATA_PATH = Path("/root/autodl-tmp/wangqihao/datasets/MemConflict/Data/Step4_4.jsonl")
TRACES_DIR = ROOT / "traces"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

CONCURRENCY = 40
MAX_RETRIES = 5
REQUEST_TIMEOUT = 120.0
MAX_TOKENS = 8192  # Increased from 4096 to avoid truncation
RETRIEVAL_TOP_K = 15

# ===== Extraction prompt =====
SESSION_EXTRACTION_PROMPT = """You are a profile extractor. Below is ONE conversation session between a user and an assistant.

Extract ALL factual information from this session about BOTH the user and the assistant.

## USER-side facts to extract:
- Personal details, preferences, opinions, events, knowledge shared
- Emotional states, mental health conditions, stressors
- Any fact that could later help answer questions about the user

## ASSISTANT-side facts to extract:
- Advice, suggestions, recommendations given
- Assessments or observations about the user
- Homework, exercises, or tasks assigned

## IMPORTANT: Include supporting evidence
For each fact, include the EXACT sentence or phrase from the conversation that supports it.

## Output format (JSON array):
[{{"attribute": "short_name", "value": "fact_value", "source": "user"|"assistant", "confidence": 0.X, "evidence": "exact supporting quote from the conversation"}}, ...]

Be thorough — extract every fact, no matter how small.
If the session contains no extractable facts, return [].

Session:
{session_text}"""

# ===== QA Prompt =====
QA_PROMPT = """You are an assistant with access to conversation history and long-term profile memory.
Below is the current conversation session, followed by relevant profile memories retrieved from past sessions.
Answer the question based on ALL provided context.
IMPORTANT: Answer in a few words ONLY. Do NOT explain your reasoning. Output the answer directly.
If the context does not contain enough information to answer, say "I don't know."

Today's date: {question_date}

=== Current Conversation (Session {session_n}, {session_n_date}) ===
{session_n_text}

=== TPPM Retrieved Memories (from sessions 1..{session_n}) ===
{retrieved_memories}

Question: {question}
Answer:"""


def load_dataset() -> list[dict[str, Any]]:
    personas = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            personas.append(json.loads(line))
    return personas


def format_session(session: dict) -> str:
    """Format session dialogue as readable text."""
    lines = [f"=== Session {session['Session_ID']} ({session['Date']}) ==="]
    dialogue = session.get("Session_Dialogue", {})
    # Sort by turn number
    turn_keys = sorted(dialogue.keys(), key=lambda k: int(k.split("_")[-1]))
    for turn_key in turn_keys:
        turns = dialogue[turn_key]
        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 1500:
                content = content[:1500] + "..."
            lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def make_tppm_config() -> TPMConfig:
    return TPMConfig(
        write_threshold=0.68,
        context_threshold=0.62,
        promote_threshold=0.72,
        promotion_min_sessions=2,
        distill_stability_threshold=0.82,
        distill_quality_threshold=0.76,
        distill_session_threshold=3,
    )


async def extract_from_session(
    client: AsyncOpenAI,
    session_text: str,
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Extract profile facts from ONE session via LLM."""
    prompt = SESSION_EXTRACTION_PROMPT.format(session_text=session_text)

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=MAX_TOKENS,
                        extra_body={"thinking": {"type": "disabled"}},
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            raw = resp.choices[0].message.content.strip()
            return _parse_json(raw)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
            else:
                print(f"  [WARN] Extraction failed after {MAX_RETRIES} retries: {e}")
    return []


def _parse_json(text: str) -> list[dict]:
    """Parse JSON array from LLM response, handling truncation and fences."""
    import re
    cleaned = text

    # Remove markdown fences
    for fence in ['```json', '```']:
        if fence in cleaned:
            parts = cleaned.split(fence)
            if len(parts) >= 3:
                cleaned = parts[1]
                break
            elif len(parts) == 2:
                # Response starts with fence but may be truncated
                cleaned = parts[1]
                break
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Try to fix truncated JSON by closing open brackets
    # Count unmatched braces and brackets
    open_braces = cleaned.count('{') - cleaned.count('}')
    open_brackets = cleaned.count('[') - cleaned.count(']')

    # Try to close the JSON
    if open_braces > 0 or open_brackets > 0:
        # Remove trailing incomplete value
        last_complete = cleaned.rfind('}')
        if last_complete > 0:
            cleaned = cleaned[:last_complete + 1]
            # Recount
            open_braces = cleaned.count('{') - cleaned.count('}')
            open_brackets = cleaned.count('[') - cleaned.count(']')

        # Close open structures
        cleaned += '}' * open_braces + ']' * open_brackets

        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

    # Fallback: extract array with regex
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            # Try to fix the extracted array
            arr_text = m.group()
            last_brace = arr_text.rfind('}')
            if last_brace > 0:
                arr_text = arr_text[:last_brace + 1] + ']'
                try:
                    return json.loads(arr_text)
                except json.JSONDecodeError:
                    pass

    return []


def facts_to_candidates(facts: list[dict], session_idx: int) -> list[ProfileCandidate]:
    """Convert extracted facts to ProfileCandidate objects."""
    candidates = []
    for f in facts:
        attr = str(f.get("attribute", "")).strip()
        val = str(f.get("value", "")).strip()
        if not attr or not val:
            continue

        source = f.get("source", "user")
        conf = float(f.get("confidence", 0.7))
        evidence_text = str(f.get("evidence", "")).strip()
        if not evidence_text:
            evidence_text = f"[session_{session_idx + 1}] {attr}: {val}"

        if source == "assistant":
            ptype = "general"
            relevance = 0.7
        else:
            ptype = "general"
            relevance = 0.85

        candidates.append(ProfileCandidate(
            attribute=attr,
            value=val,
            context=evidence_text,
            profile_type=ptype,
            scene=f"session_{session_idx + 1}",
            confidence=conf,
            stability=0.6,
            recency=1.0,
            explicitness=0.8,
            user_relevance=relevance,
            source=f"llm_deepseek_{source}",
        ))
    return candidates


def count_events(memory: TemporalProfileMemory, prev_state: dict) -> dict:
    """Count memory events by comparing current state to previous snapshot."""
    curr_short = len(memory.short_term_memory)
    curr_long = len(memory.long_term_memory)
    prev_short = prev_state.get("short_term", 0)
    prev_long = prev_state.get("long_term", 0)

    # Count events from units
    created = 0
    fused = 0
    contradicted = 0
    promoted = 0

    for unit in memory.short_term_memory + memory.long_term_memory:
        if unit.session_count == 1 and unit.reinforcement_count == 1:
            created += 1
        elif unit.reinforcement_count > 1:
            fused += 1
        if unit.contradiction_count > 0:
            contradicted += 1
        if unit.memory_level == "long_term" and unit.session_count >= 2:
            promoted += 1

    return {
        "created": created,
        "fused": fused,
        "contradicted": contradicted,
        "promoted": promoted,
        "short_term": curr_short,
        "long_term": curr_long,
    }


def format_retrieved_memories(memory: TemporalProfileMemory, question: str) -> str:
    """Format retrieved memories for QA context."""
    retrieved = memory.retrieve(question, scene="general", top_k=RETRIEVAL_TOP_K)

    if not retrieved:
        return "(no relevant memories retrieved from past sessions)"

    # Sort by memory_level priority
    level_priority = {"long_term": 3, "short_term": 2, "working": 1}
    retrieved.sort(key=lambda u: (level_priority.get(u.memory_level, 0), u.stability_score),
                   reverse=True)

    lines = [f"(top-{len(retrieved)} memories retrieved)"]
    for unit in retrieved:
        evidence_items = sorted(unit.evidence, key=lambda e: e.timestamp, reverse=True)[:3]
        line = f"- [{unit.memory_level}] {unit.attribute}: {unit.value} "
        line += f"(stability={unit.stability_score:.2f}, confidence={unit.confidence_score:.2f}, "
        line += f"sessions={unit.session_count}, reinforcements={unit.reinforcement_count})"
        lines.append(line)

        for ev in evidence_items:
            ev_text = ev.content[:200]
            lines.append(f"    evidence[{ev.scene}]: \"{ev_text}\"")

    return "\n".join(lines)


async def answer_question(
    client: AsyncOpenAI,
    memory: TemporalProfileMemory,
    session: dict,
    question: dict,
    session_text: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Answer a single question using TPPM retrieval."""
    session_n = session["Session_ID"] + 1
    session_date = session["Date"]

    retrieved_text = format_retrieved_memories(memory, question["question"])

    qa_prompt = QA_PROMPT.format(
        question_date=session_date,
        session_n=session_n,
        session_n_date=session_date,
        session_n_text=session_text,
        retrieved_memories=retrieved_text,
        question=question["question"],
    )

    # Get model answer
    model_answer = ""
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": qa_prompt}],
                        temperature=0.0,
                        max_tokens=256,
                        extra_body={"thinking": {"type": "disabled"}},
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            model_answer = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
            else:
                return {
                    "question_id": question["question_id"],
                    "conflict_type": question["conflict_type"],
                    "error": f"qa_failed: {e}",
                }

    # Simple exact match judge (case-insensitive, strip punctuation)
    gold = question["answer"].strip().lower()
    pred = model_answer.strip().lower()

    # Check if gold answer is contained in prediction or vice versa
    is_correct = (
        gold in pred or
        pred in gold or
        gold.replace(".", "") == pred.replace(".", "") or
        (gold in ["yes", "no"] and pred.startswith(gold))
    )

    return {
        "question_id": question["question_id"],
        "conflict_type": question["conflict_type"],
        "ability_target": question.get("ability_target", ""),
        "difficulty": question.get("difficulty", ""),
        "model_answer": model_answer,
        "gold_answer": question["answer"],
        "is_correct": is_correct,
    }


async def process_persona(
    client: AsyncOpenAI,
    persona: dict[str, Any],
    persona_idx: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Process one persona through all sessions."""
    persona_id = persona["ID"]
    sessions = persona["Full_Session_Chain"]
    n_sessions = len(sessions)

    print(f"\n{'='*60}")
    print(f"  Persona {persona_idx}: {persona_id[:20]}... ({n_sessions} sessions)")
    print(f"{'='*60}")

    # Initialize TPPM
    config = make_tppm_config()
    memory = TemporalProfileMemory(config=config)

    session_traces = []
    question_results = []

    # Extract all sessions in parallel (preserving order)
    print(f"  Extracting candidates from {n_sessions} sessions...")
    session_texts = [format_session(s) for s in sessions]

    extraction_results = [None] * n_sessions

    async def extract_with_idx(idx: int):
        facts = await extract_from_session(client, session_texts[idx], sem)
        return idx, facts

    tasks = [extract_with_idx(i) for i in range(n_sessions)]
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"  Extraction",
        leave=False,
    ):
        idx, facts = await coro
        extraction_results[idx] = facts

    # Now process sessions sequentially through TPPM
    print(f"  Processing sessions through TPPM...")
    prev_state = {"short_term": 0, "long_term": 0}

    for session_idx in tqdm(range(n_sessions), desc="  Sessions", leave=False):
        session = sessions[session_idx]
        session_id = session["Session_ID"]
        session_date = session["Date"]
        scene = f"session_{session_idx + 1}"

        # Get extracted facts
        facts = extraction_results[session_idx] or []
        candidates = facts_to_candidates(facts, session_idx)

        # TPPM ingest - use unique session_id for each session
        unique_session_id = f"{persona_id}_session_{session_idx}"
        memory.start_session(scene, session_id=unique_session_id)
        accepted = memory.ingest_candidates(candidates, scene=scene, session_id=unique_session_id)

        # Capture working memory BEFORE finish_session clears it
        working_size = len(memory.working_memory)

        memory.finish_session(scene)

        # Count events
        events = count_events(memory, prev_state)

        # Calculate compression
        total_input_chars = len(session_texts[session_idx])
        all_memories = memory.all_memories()
        memory_chars = sum(len(u.value) + len(u.context) for u in all_memories)
        compression_ratio = total_input_chars / max(1, memory_chars)

        # Distillation candidates
        n_distill = len(memory.distillation_candidates())

        # Record session trace
        trace = {
            "session_id": session_id,
            "session_idx": session_idx,
            "date": session_date,
            "n_candidates_extracted": len(candidates),
            "n_candidates_accepted": len(accepted),
            "layer_sizes": {
                "working": working_size,
                "short_term": len(memory.short_term_memory),
                "long_term": len(memory.long_term_memory),
            },
            "events": {
                "created": events["created"],
                "fused": events["fused"],
                "contradicted": events["contradicted"],
                "promoted": events["promoted"],
            },
            "total_input_chars": total_input_chars,
            "memory_chars": memory_chars,
            "compression_ratio": round(compression_ratio, 2),
            "n_distillation_candidates": n_distill,
        }
        session_traces.append(trace)

        # Update prev state
        prev_state = {"short_term": events["short_term"], "long_term": events["long_term"]}

        # Answer questions for this session
        questions = session.get("Session_Questions", [])
        if questions:
            session_text = session_texts[session_idx]
            qa_tasks = [
                answer_question(client, memory, session, q, session_text, sem)
                for q in questions
            ]

            for coro in asyncio.as_completed(qa_tasks):
                result = await coro
                result["session_id"] = session_id
                result["session_idx"] = session_idx
                question_results.append(result)

    # Build final trace
    final_trace = {
        "persona_id": persona_id,
        "persona_idx": persona_idx,
        "n_sessions": n_sessions,
        "session_traces": session_traces,
        "question_results": question_results,
        "final_memory_summary": {
            "short_term": len(memory.short_term_memory),
            "long_term": len(memory.long_term_memory),
            "total_evidence": len(memory.evidence_store),
        },
    }

    # Save trace
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRACES_DIR / f"persona_{persona_idx}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_trace, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved trace to {output_path}")
    print(f"    Sessions: {n_sessions}, Questions: {len(question_results)}")
    print(f"    Final memory: {len(memory.short_term_memory)} short-term, {len(memory.long_term_memory)} long-term")

    return final_trace


async def main_async(max_personas: int | None, persona_idx: int | None, concurrency: int):
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    sem = asyncio.Semaphore(concurrency)

    personas = load_dataset()
    print(f"Loaded {len(personas)} personas from {DATA_PATH}")

    if persona_idx is not None:
        # Process single persona
        if persona_idx < len(personas):
            await process_persona(client, personas[persona_idx], persona_idx, sem)
        else:
            print(f"ERROR: persona_idx {persona_idx} out of range (max {len(personas)-1})")
    else:
        # Process all personas
        if max_personas:
            personas = personas[:max_personas]

        for idx, persona in enumerate(personas):
            # Check if already processed
            trace_path = TRACES_DIR / f"persona_{idx}.json"
            if trace_path.exists():
                print(f"  [SKIP] Persona {idx} already processed")
                continue

            await process_persona(client, persona, idx, sem)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print(f"  Traces saved to: {TRACES_DIR}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="TPPM Memory Lifecycle Pipeline")
    parser.add_argument("--max-personas", type=int, default=None, help="Max personas to process")
    parser.add_argument("--persona-idx", type=int, default=None, help="Process specific persona")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="API concurrency limit")
    args = parser.parse_args()

    asyncio.run(main_async(args.max_personas, args.persona_idx, args.concurrency))


if __name__ == "__main__":
    main()
