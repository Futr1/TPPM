#!/usr/bin/env python3
"""Phase 2 (TPPM): QA evaluation using TPPM memory retrieval.

For each (question_id, N) from Phase 1:
  1. Load TPPM memory state
  2. Retrieve relevant memories using memory.retrieve(question)
  3. Format retrieved memories as profile text
  4. Answer question + judge correctness

KEY DIFFERENCE from old phase2_eval_qa.py:
  - Old: stuffs ALL profile items into QA prompt (noise at high N)
  - New: uses TPPM's retrieve() to fetch top-K relevant memories

Usage:
    python3 tppm_phase2_qa.py
    python3 tppm_phase2_qa.py --max-questions 5 -N 1 5 10
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Add TPPM library to path =====
TPPM_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
sys.path.insert(0, str(TPPM_ROOT))

from mini_agent.tpm.memory import TemporalProfileMemory

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DATA_PATH = ROOT / "sampled_100.json"
MEMORY_DIR = ROOT / "tppm_memory_states"        # Phase 1 output
EVAL_DIR = ROOT / "eval_results"
RESULTS_PATH = EVAL_DIR / "tppm_qa_results.json"
AGGREGATE_PATH = EVAL_DIR / "tppm_aggregate_results.json"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

CONCURRENCY = 8
MAX_RETRIES = 5
REQUEST_TIMEOUT = 120.0
RETRIEVAL_TOP_K = 15  # Number of memories to retrieve for QA context

DEFAULT_N_VALUES = [10, 15, 20, 25, 30, 35, 40, 48]

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

=== Session Date Reference ===
{session_date_table}

Question: {question}
Answer:"""


def get_judge_prompt(task: str, question: str, answer: str, response: str,
                     abstention: bool = False) -> str:
    """LongMemEval judge templates."""
    if not abstention:
        if task in ('single-session-user', 'single-session-assistant', 'multi-session'):
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == 'temporal-reasoning':
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "In addition, do not penalize off-by-one errors for the number of days/weeks/months. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes "
                "off-by-one errors (e.g., predicting 19 days when the answer is 18), "
                "the model's response is still correct.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == 'knowledge-update':
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == 'single-session-preference':
            return (
                "I will give you a question, a rubric for desired personalized response, "
                "and a response from a model. Please answer yes if the response satisfies "
                "the desired response. Otherwise, answer no. The model does not need to reflect "
                "all the points in the rubric. The response is correct as long as it recalls "
                "and utilizes the user's personal information correctly.\n\n"
                f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        else:
            raise ValueError(f"Unknown task type: {task}")
    else:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information "
            "is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_state(question_id: str, n: int) -> dict[str, Any] | None:
    path = MEMORY_DIR / f"{question_id}_N{n}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_session_text(session: list[dict], session_idx: int) -> str:
    """Format ONE session as readable conversation text (mirrors Mini-Agent-5-1 messages)."""
    lines = []
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + "..."
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def format_tppm_context(memory: TemporalProfileMemory, question: str,
                         session_n_text: str, session_n: int,
                         question_date: str,
                         session_dates: list[str],
                         top_k: int = RETRIEVAL_TOP_K) -> str:
    """Build full QA context: session N conversation + TPPM retrieved memories.

    This mirrors Mini-Agent-5-1's inference pattern:
      - session_n_text ← self.messages (current session's full dialogue history)
      - retrieved memories ← augment_user_message() (cross-session TPPM memories)

    Now includes session dates for temporal reasoning questions.
    """
    retrieved = memory.retrieve(question, scene="general", top_k=top_k)

    if not retrieved:
        memories_text = "(no relevant memories retrieved from past sessions)"
    else:
        # Sort by memory_level priority (long_term > short_term) then stability
        level_priority = {"long_term": 3, "short_term": 2, "working": 1}
        retrieved.sort(key=lambda u: (level_priority.get(u.memory_level, 0), u.stability_score),
                       reverse=True)

        user_blocks = []
        asst_blocks = []

        for unit in retrieved:
            evidence_items = sorted(unit.evidence, key=lambda e: e.timestamp, reverse=True)[:3]

            lines = [f"- [{unit.memory_level}] {unit.attribute}: {unit.value} "
                     f"(stability={unit.stability_score:.2f}, confidence={unit.confidence_score:.2f}, "
                     f"sessions={unit.session_count}, reinforcements={unit.reinforcement_count})"]

            for ev in evidence_items:
                ev_text = ev.content[:200]
                # Resolve scene label to date if available
                scene_label = ev.scene or ""
                ev_date = _resolve_scene_date(scene_label, session_dates)
                date_str = f" [{ev_date}]" if ev_date else ""
                lines.append(f"    evidence[{scene_label}{date_str}]: \"{ev_text}\"")

            block = "\n".join(lines)

            is_asst = any("assistant" in (e.source if hasattr(e, 'source') else "").lower()
                          for e in unit.evidence)
            if is_asst:
                asst_blocks.append(block)
            else:
                user_blocks.append(block)

        parts = [f"(top-{len(retrieved)} memories retrieved)"]
        if user_blocks:
            parts.append("--- User Profile ---")
            parts.extend(user_blocks)
        if asst_blocks:
            parts.append("--- Assistant Interventions ---")
            parts.extend(asst_blocks)
        memories_text = "\n".join(parts)

    # Build session date reference table
    session_n_date = session_dates[session_n - 1] if session_n <= len(session_dates) else "unknown"
    date_lines = []
    # Show dates for sessions 1..session_n that have evidence in retrieved memories
    referenced_sessions = set()
    for unit in retrieved:
        for ev in unit.evidence:
            if ev.scene and ev.scene.startswith("session_"):
                try:
                    referenced_sessions.add(int(ev.scene.split("_")[1]))
                except ValueError:
                    pass
    for s_idx in sorted(referenced_sessions):
        if s_idx <= len(session_dates):
            date_lines.append(f"  Session {s_idx}: {session_dates[s_idx - 1]}")
    # Always include session N date
    if session_n not in referenced_sessions and session_n <= len(session_dates):
        date_lines.append(f"  Session {session_n}: {session_dates[session_n - 1]}")
    session_date_table = "\n".join(date_lines) if date_lines else "(no date information available)"

    return QA_PROMPT.format(
        session_n=session_n,
        session_n_date=session_n_date,
        session_n_text=session_n_text,
        retrieved_memories=memories_text,
        question_date=question_date,
        session_date_table=session_date_table,
        question=question,
    )


def _resolve_scene_date(scene_label: str, session_dates: list[str]) -> str:
    """Map scene label like 'session_20' to actual date string."""
    if not scene_label or not scene_label.startswith("session_"):
        return ""
    try:
        s_idx = int(scene_label.split("_")[1]) - 1  # 0-based
        if 0 <= s_idx < len(session_dates):
            return session_dates[s_idx]
    except (ValueError, IndexError):
        pass
    return ""


async def answer_and_judge(
    client: AsyncOpenAI,
    question_id: str,
    question_type: str,
    question: str,
    answer: str,
    n: int,
    haystack_sessions: list[list[dict]],
    question_date: str,
    haystack_dates: list[str],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Answer using session N context + TPPM retrieval, then judge.

    Context mirrors Mini-Agent-5-1 inference pattern:
      - session N text = self.messages (current dialogue history)
      - TPPM memories = augment_user_message() (cross-session profile)
    """
    state = load_memory_state(question_id, n)
    if state is None:
        return {"question_id": question_id, "N": n, "error": "memory state not found"}

    # Reconstruct memory from saved state
    try:
        memory = TemporalProfileMemory.from_dict(state["memory_state"])
    except Exception as e:
        return {"question_id": question_id, "N": n, "error": f"failed to load memory: {e}"}

    # Get session N text as "current conversation history"
    n_available = min(n, len(haystack_sessions))
    n_dates = min(n, len(haystack_dates))
    session_n_idx = n_available - 1  # 0-based
    session_n_text = format_session_text(haystack_sessions[session_n_idx], session_n_idx)

    # Build session dates slice for this N
    session_dates = haystack_dates[:n_available] if n_available <= len(haystack_dates) else haystack_dates

    qa_prompt = format_tppm_context(
        memory, question, session_n_text, n_available,
        question_date, session_dates,
    )

    # Step 1: Get answer
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
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return {"question_id": question_id, "N": n, "error": "qa failed"}
            await asyncio.sleep(min(2 ** attempt, 30))

    # Step 2: Judge
    is_abstention = "_abs" in question_id
    judge_prompt = get_judge_prompt(question_type, question, answer, model_answer, is_abstention)
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=0.0,
                        max_tokens=256,
                        extra_body={"thinking": {"type": "disabled"}},
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            judge_raw = resp.choices[0].message.content.strip().lower()
            is_correct = "yes" in judge_raw
            break
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return {"question_id": question_id, "N": n, "error": "judge failed"}
            await asyncio.sleep(min(2 ** attempt, 30))

    return {
        "question_id": question_id,
        "question_type": question_type,
        "N": n,
        "model_answer": model_answer,
        "correct_answer": answer,
        "judge_raw": judge_raw,
        "is_correct": is_correct,
        "retrieval_size": len(qa_prompt),
    }


def get_min_n_required(haystack_sessions: list[list[dict]]) -> int:
    """Find the maximum session index (1-based) that has answer-related turns.

    Only sessions with has_answer=True contain information needed to answer
    the question. All such sessions must be ≤ N for the question to be answerable.
    """
    max_session = 0
    for i, session in enumerate(haystack_sessions):
        for turn in session:
            if turn.get("has_answer"):
                max_session = max(max_session, i + 1)
    return max_session


async def main_async(n_values: list[int], max_questions: int | None, concurrency: int):
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    sem = asyncio.Semaphore(concurrency)

    data = load_dataset(DATA_PATH)
    if max_questions:
        data = data[:max_questions]

    tasks = []
    skipped_count = 0
    for entry in data:
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer = entry["answer"]
        sessions = entry["haystack_sessions"]
        question_date = entry.get("question_date", "unknown")
        dates = entry.get("haystack_dates", [])
        min_n = get_min_n_required(sessions)
        for n in n_values:
            if n > len(sessions):
                skipped_count += 1
                continue
            if n < min_n:
                skipped_count += 1
                continue  # answer info not yet in memory
            if load_memory_state(qid, n) is not None:
                tasks.append(answer_and_judge(
                    client, qid, qtype, question, answer, n, sessions,
                    question_date, dates, sem,
                ))

    print(f"Found {len(tasks)} valid (question, N) pairs ({skipped_count} skipped: N < min_required or N > sessions)")

    results = []
    with tqdm(total=len(tasks), desc="TPPM Phase 2: QA + Judge") as pbar:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            pbar.update(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Aggregate
    valid = [r for r in results if "is_correct" in r]
    n2correct = {}
    n2total = {}
    n2type_correct = {}
    n2type_total = {}

    for r in valid:
        n = r["N"]
        n2correct[n] = n2correct.get(n, 0) + (1 if r["is_correct"] else 0)
        n2total[n] = n2total.get(n, 0) + 1

        qtype = r["question_type"]
        key = (n, qtype)
        n2type_correct[key] = n2type_correct.get(key, 0) + (1 if r["is_correct"] else 0)
        n2type_total[key] = n2type_total.get(key, 0) + 1

    aggregate = {"overall": {}, "by_type": {}}
    for n in sorted(n2total.keys()):
        acc = n2correct[n] / n2total[n] if n2total[n] > 0 else 0
        aggregate["overall"][str(n)] = {
            "accuracy": round(acc, 4),
            "correct": n2correct[n], "total": n2total[n],
        }

    for (n, qtype), correct in sorted(n2type_correct.items()):
        total = n2type_total[(n, qtype)]
        acc = correct / total if total > 0 else 0
        key_n = str(n)
        if key_n not in aggregate["by_type"]:
            aggregate["by_type"][key_n] = {}
        aggregate["by_type"][key_n][qtype] = {
            "accuracy": round(acc, 4),
            "correct": correct, "total": total,
        }

    with AGGREGATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"\nAggregate results saved to {AGGREGATE_PATH}")
    print("\n=== TPPM Accuracy by N ===")
    for n in sorted(n2total.keys()):
        acc = n2correct[n] / n2total[n] if n2total[n] > 0 else 0
        print(f"  N={n:>3}: {acc:.4f} ({n2correct[n]}/{n2total[n]})")

    return results


def main():
    parser = argparse.ArgumentParser(description="TPPM Phase 2: QA with retrieval")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    asyncio.run(main_async(args.n_values, args.max_questions, args.concurrency))


if __name__ == "__main__":
    main()
