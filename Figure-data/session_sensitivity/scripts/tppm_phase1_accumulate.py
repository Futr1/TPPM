#!/usr/bin/env python3
"""Phase 1 (TPPM): Sequential memory accumulation through TPPM pipeline.

For each N in {1,5,10,15,20,30,48} and each question:
  - Process sessions 1..N ONE BY ONE through TPPM
  - Each session: LLM extracts ProfileCandidates → memory.ingest → memory.evolve
  - After N sessions: save full TPPM memory state

KEY DIFFERENCE from old phase1_extract.py:
  - Old: all N sessions stuffed into ONE prompt → LLM outputs flat JSON
  - New: N sessions processed sequentially through TPPM → memory accumulates/evolves

N here = number of sessions fed through the TPPM pipeline (= memory evolution rounds).

Usage:
    python3 tppm_phase1_accumulate.py
    python3 tppm_phase1_accumulate.py --max-questions 5 -N 1 5 10
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DATA_PATH = ROOT / "sampled_100.json"
OUTPUT_DIR = ROOT / "tppm_memory_states"
FAILED_LOG = ROOT / "logs" / "tppm_phase1_failed.jsonl"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

CONCURRENCY = 80
MAX_RETRIES = 5
REQUEST_TIMEOUT = 120.0
MAX_TOKENS = 4096

DEFAULT_N_VALUES = [10, 15, 20, 25, 30, 35, 40, 48]

# ===== Extraction prompt: ONE session at a time =====
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
- Information or psychoeducation provided
- Any fact that could later help answer questions about what the assistant said

## IMPORTANT: Include supporting evidence
For each fact, include the EXACT sentence or phrase from the conversation that supports it.
This evidence is CRITICAL for downstream QA — without it, the fact is useless.

## Output format (JSON array):
[{{"attribute": "short_name", "value": "fact_value", "source": "user"|"assistant", "confidence": 0.X, "evidence": "exact supporting quote from the conversation"}}, ...]

Be thorough — extract every fact, no matter how small.
If the session contains no extractable facts, return [].

Session:
{session_text}"""


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_session(session: list[dict], session_idx: int,
                    session_date: str = "") -> str:
    """Format ONE session as readable text."""
    date_info = f" ({session_date})" if session_date else ""
    lines = [f"=== Session {session_idx + 1}{date_info} ==="]
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "..."
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def make_tppm_config() -> TPMConfig:
    """Create TPPM config with standard parameters."""
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
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
    return []


def _parse_json(text: str) -> list[dict]:
    """Parse JSON array from LLM response."""
    import re
    cleaned = text
    for fence in ['```json', '```']:
        if fence in cleaned:
            parts = cleaned.split(fence)
            if len(parts) >= 3:
                cleaned = parts[1]
                break
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return []


def facts_to_candidates(facts: list[dict], session_idx: int) -> list[ProfileCandidate]:
    """Convert extracted facts to ProfileCandidate objects for TPPM ingestion.

    The 'evidence' field from LLM extraction becomes the candidate's context,
    which TPPM stores as EvidenceItem.content — making it available for downstream QA.
    """
    candidates = []
    for f in facts:
        attr = str(f.get("attribute", "")).strip()
        val = str(f.get("value", "")).strip()
        if not attr or not val:
            continue

        source = f.get("source", "user")
        conf = float(f.get("confidence", 0.7))
        # Use LLM-provided evidence as context; fall back to session label
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
            context=evidence_text,           # ← stored as EvidenceItem.content in TPPM
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


def _build_state(question_id: str, n: int, n_available: int, n_total: int,
                 total_candidates: int, total_accepted: int, extraction_failures: int,
                 memory: TemporalProfileMemory, haystack_dates: list[str]) -> dict[str, Any]:
    """Build a state dict for saving."""
    all_units = memory.all_memories()
    saved_dates = haystack_dates[:n_available] if n_available <= len(haystack_dates) else haystack_dates
    return {
        "question_id": question_id,
        "N": n,
        "n_sessions_available": n_total,
        "n_sessions_processed": n_available,
        "total_candidates_extracted": total_candidates,
        "total_accepted_into_memory": total_accepted,
        "extraction_failures": extraction_failures,
        "haystack_dates": saved_dates,
        "tppm_summary": {
            "short_term_count": len(memory.short_term_memory),
            "long_term_count": len(memory.long_term_memory),
            "total_memories": len(all_units),
            "evidence_store_size": len(memory.evidence_store),
        },
        "memory_state": memory.to_dict(),
    }


async def accumulate_all_checkpoints(
    client: AsyncOpenAI,
    question_id: str,
    haystack_sessions: list[list[dict]],
    haystack_dates: list[str],
    checkpoints: list[int],
    sem: asyncio.Semaphore,
) -> dict[int, dict[str, Any]]:
    """Extract all sessions in parallel, then accumulate through TPPM sequentially.

    Parallel extraction means the question-level runtime is bounded by the
    slowest single session (~30s), not the sum of all sessions (~45×30s).
    """
    checkpoint_set = set(checkpoints)
    config = make_tppm_config()
    memory = TemporalProfileMemory(config=config)

    n_max = len(haystack_sessions)
    n_dates = len(haystack_dates)

    # Phase A: Extract ALL sessions in parallel
    async def _extract_one(idx: int) -> tuple[int, list[dict]]:
        session = haystack_sessions[idx]
        session_date = haystack_dates[idx] if idx < n_dates else ""
        session_text = format_session(session, idx, session_date)
        facts = await extract_from_session(client, session_text, sem)
        return idx, facts

    extraction_results: dict[int, list[dict]] = {}
    tasks = [_extract_one(i) for i in range(n_max)]
    for coro in asyncio.as_completed(tasks):
        idx, facts = await coro
        extraction_results[idx] = facts

    # Phase B: Feed through TPPM in session order (fast, no LLM calls)
    total_candidates = 0
    total_accepted = 0
    extraction_failures = 0
    saved_states: dict[int, dict[str, Any]] = {}

    for session_idx in range(n_max):
        scene = f"session_{session_idx + 1}"

        facts = extraction_results.get(session_idx, [])
        if not facts:
            extraction_failures += 1

        candidates = facts_to_candidates(facts, session_idx)
        total_candidates += len(candidates)

        memory.start_session(scene, session_id=question_id)
        accepted = memory.ingest_candidates(candidates, scene=scene, session_id=question_id)
        total_accepted += len(accepted)

        memory.finish_session(scene)

        # Save checkpoint if current N is a target
        n_current = session_idx + 1
        if n_current in checkpoint_set:
            state = _build_state(
                question_id, n_current, n_current, n_max,
                total_candidates, total_accepted, extraction_failures,
                memory, haystack_dates,
            )
            output_path = OUTPUT_DIR / f"{question_id}_N{n_current}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            saved_states[n_current] = state

    return saved_states


async def main_async(n_values: list[int], max_questions: int | None, concurrency: int):
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    sem = asyncio.Semaphore(concurrency)

    data = load_dataset(DATA_PATH)
    if max_questions:
        data = data[:max_questions]

    checkpoints = sorted(n_values)
    print(f"Loaded {len(data)} questions, checkpoints: {checkpoints}")

    # ===== Gather all (qid, session_idx, session_text) tuples =====
    class ExtractionTask:
        __slots__ = ('qid', 'idx', 'session_text')
        def __init__(self, qid, idx, session_text):
            self.qid = qid
            self.idx = idx
            self.session_text = session_text

    all_tasks: list[ExtractionTask] = []
    qid_info: dict[str, dict] = {}

    for entry in data:
        qid = entry["question_id"]
        sessions = entry["haystack_sessions"]
        n_sessions = len(sessions)
        dates = entry.get("haystack_dates", [])
        n_dates = len(dates)
        needed = [n for n in checkpoints if n <= n_sessions]
        existing = [n for n in needed if (OUTPUT_DIR / f"{qid}_N{n}.json").exists()]
        if len(existing) == len(needed) and len(needed) > 0:
            continue  # Already complete

        qid_info[qid] = {
            "sessions": sessions,
            "dates": dates,
            "n_sessions": n_sessions,
            "needed": needed,
        }

        for idx in range(n_sessions):
            session_date = dates[idx] if idx < n_dates else ""
            session_text = format_session(sessions[idx], idx, session_date)
            all_tasks.append(ExtractionTask(qid, idx, session_text))

    total_sessions = len(all_tasks)
    print(f"Questions to process: {len(qid_info)}, total sessions: {total_sessions}")
    print(f"Concurrency: {concurrency}")

    # ===== Phase 1a: Flat batch extraction of ALL sessions =====
    async def _extract_wrapped(task: ExtractionTask) -> tuple[str, int, list[dict]]:
        facts = await extract_from_session(client, task.session_text, sem)
        return task.qid, task.idx, facts

    # Check for cached extraction results
    CACHE_PATH = OUTPUT_DIR / "_extraction_cache.json"
    extraction_results: dict[str, dict[int, list[dict]]] = {qid: {} for qid in qid_info}

    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r") as f:
                cached_raw = json.load(f)
            for qid, sessions_dict in cached_raw.items():
                if qid in extraction_results:
                    extraction_results[qid] = {int(k): v for k, v in sessions_dict.items()}
            already = sum(len(v) for v in extraction_results.values())
            # Filter tasks: only extract sessions not already cached
            all_tasks = [t for t in all_tasks
                        if t.idx not in extraction_results.get(t.qid, {})]
            print(f"Loaded {already} cached extractions, {len(all_tasks)} remaining")
        except Exception:
            pass

    t_extract = 0
    if all_tasks:
        t0 = time.time()
        with tqdm(total=len(all_tasks), desc="Phase 1a: Extract") as pbar:
            coros = [_extract_wrapped(t) for t in all_tasks]
            for coro in asyncio.as_completed(coros):
                qid, idx, facts = await coro
                extraction_results[qid][idx] = facts
                pbar.update(1)
        t_extract = time.time() - t0
        print(f"\nExtraction done in {t_extract/60:.1f} min ({len(all_tasks)/t_extract:.1f} sessions/s)")

        # Cache to disk
        serializable = {qid: {str(k): v for k, v in sessions.items()}
                       for qid, sessions in extraction_results.items()
                       if sessions}
        with CACHE_PATH.open("w") as f:
            json.dump(serializable, f, ensure_ascii=False)
        print(f"Extraction cache saved to {CACHE_PATH}")
    else:
        print(f"All extractions cached, skipping Phase 1a")

    # ===== Phase 1b: Parallel multi-question TPPM accumulation =====
    # Each question is independent → use ThreadPoolExecutor for parallelism
    TPPM_THREADS = 10  # CPU-bound, moderate parallelism

    def _run_tppm_for_question(qid: str) -> int:
        """Run TPPM for one question; return number of checkpoints saved."""
        info = qid_info[qid]
        n_max = info["n_sessions"]
        dates = info["dates"]
        needed = info["needed"]
        checkpoint_set = set(needed)

        config = make_tppm_config()
        memory = TemporalProfileMemory(config=config)

        total_candidates = 0
        total_accepted = 0
        extraction_failures = 0
        local_saved = 0

        for session_idx in range(n_max):
            scene = f"session_{session_idx + 1}"
            facts = extraction_results[qid].get(session_idx, [])
            if not facts:
                extraction_failures += 1

            candidates = facts_to_candidates(facts, session_idx)
            total_candidates += len(candidates)

            memory.start_session(scene, session_id=qid)
            accepted = memory.ingest_candidates(candidates, scene=scene, session_id=qid)
            total_accepted += len(accepted)
            memory.finish_session(scene)

            n_current = session_idx + 1
            if n_current in checkpoint_set:
                state = _build_state(
                    qid, n_current, n_current, n_max,
                    total_candidates, total_accepted, extraction_failures,
                    memory, dates,
                )
                output_path = OUTPUT_DIR / f"{qid}_N{n_current}.json"
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                local_saved += 1

        return local_saved

    t1 = time.time()
    saved_count = 0
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=TPPM_THREADS) as pool, \
         tqdm(total=len(qid_info), desc="Phase 1b: TPPM") as pbar:
        futures = [loop.run_in_executor(pool, _run_tppm_for_question, qid)
                   for qid in qid_info]
        for coro in asyncio.as_completed(futures):
            n_saved = await coro
            saved_count += n_saved
            pbar.update(1)

    t_tppm = time.time() - t1
    print(f"\nTPPM done in {t_tppm/60:.1f} min ({saved_count} checkpoints saved)")
    print(f"Total: {t_extract/60 + t_tppm/60:.1f} min")

    return


def main():
    parser = argparse.ArgumentParser(description="TPPM Phase 1: Sequential memory accumulation")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    asyncio.run(main_async(args.n_values, args.max_questions, args.concurrency))


if __name__ == "__main__":
    main()
