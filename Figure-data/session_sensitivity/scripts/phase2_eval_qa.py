#!/usr/bin/env python3
"""Phase 2: QA evaluation with session-truncated TPPM profiles.

For each (question_id, N) from Phase 1:
  1. Load TPPM profile
  2. Answer question using profile as context
  3. Judge answer correctness (LongMemEval per-type templates)
  4. Aggregate accuracy per N

Usage:
    python3 phase2_eval_qa.py                       # full run
    python3 phase2_eval_qa.py --max-questions 5 -N 1 3 5  # quick test
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

import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DATA_PATH = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity/sampled_100.json")
PROFILES_DIR = ROOT / "extracted_profiles"
EVAL_DIR = ROOT / "eval_results"
RESULTS_PATH = EVAL_DIR / "qa_results.json"
AGGREGATE_PATH = EVAL_DIR / "aggregate_results.json"

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

DEFAULT_N_VALUES = [1, 5, 10, 15, 20, 30, 48]

QA_PROMPT = """You are given a user profile extracted from conversation history.
Answer the following question based on the profile. Be concise (a few words).
If the profile does not contain enough information to answer, say "I don't know."

Profile:
{profile}

Question: {question}
Answer:"""


def get_judge_prompt(task: str, question: str, answer: str, response: str,
                     abstention: bool = False) -> str:
    """LongMemEval judge templates, adapted from evaluate_qa.py."""
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


def load_profile(question_id: str, n: int) -> dict[str, Any] | None:
    path = PROFILES_DIR / f"{question_id}_N{n}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_profile(profile: dict[str, Any]) -> str:
    """Format profile items as readable text, sorted by session and grouped by source.

    Key improvements for temporal reasoning and assistant-side coverage:
    1. Items are sorted by session (chronological order)
    2. Items are grouped by source: USER profile first, then ASSISTANT interventions
    3. Same-attribute items across sessions are kept in timeline order so the QA
       model can infer changes over time (e.g., medication X at session 3, discontinued at session 7)
    """
    items = profile.get("profile_items", [])
    if not items:
        return "(empty profile)"

    # Sort by session (chronological), then by source (user first)
    def _sort_key(item):
        sess = item.get("session", 999)
        if isinstance(sess, str):
            try:
                sess = int(sess)
            except ValueError:
                sess = 999
        source = item.get("source", "user")
        return (sess, 0 if source == "user" else 1)

    sorted_items = sorted(items, key=_sort_key)

    # Separate by source
    user_items = [it for it in sorted_items if it.get("source", "user") == "user"]
    assistant_items = [it for it in sorted_items if it.get("source", "") == "assistant"]

    lines = []

    # --- USER profile section ---
    lines.append("=== USER PROFILE (chronological by session) ===")
    if user_items:
        for item in user_items:
            attr = item.get("attribute", "")
            val = item.get("value", "")
            sess = item.get("session", "?")
            if attr and val:
                lines.append(f"- [Session {sess}] {attr}: {val}")
            elif "raw" in item:
                lines.append(f"- {item['raw']}")
    else:
        lines.append("(no user facts extracted)")

    # --- ASSISTANT intervention section ---
    lines.append("\n=== ASSISTANT INTERVENTIONS (chronological by session) ===")
    if assistant_items:
        for item in assistant_items:
            attr = item.get("attribute", "")
            val = item.get("value", "")
            sess = item.get("session", "?")
            if attr and val:
                lines.append(f"- [Session {sess}] {attr}: {val}")
            elif "raw" in item:
                lines.append(f"- {item['raw']}")
    else:
        lines.append("(no assistant facts extracted)")

    return "\n".join(lines)


async def answer_and_judge(
    client: AsyncOpenAI,
    question_id: str,
    question_type: str,
    question: str,
    answer: str,
    n: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Answer question using profile, then judge correctness."""
    profile = load_profile(question_id, n)
    if profile is None:
        return {"question_id": question_id, "N": n, "error": "profile not found"}

    profile_text = format_profile(profile)
    is_abstention = "_abs" in question_id
    qa_prompt = QA_PROMPT.format(profile=profile_text, question=question)

    # Step 1: Get model answer
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": qa_prompt}],
                        temperature=0.0,
                        max_tokens=256,
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
    judge_prompt = get_judge_prompt(
        question_type, question, answer, model_answer, is_abstention
    )
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=API_MODEL,
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=0.0,
                        max_tokens=256,
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
    }


async def main_async(n_values: list[int], max_questions: int | None, concurrency: int):
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    sem = asyncio.Semaphore(concurrency)

    data = load_dataset(DATA_PATH)
    if max_questions:
        data = data[:max_questions]

    # Collect all (question_id, N) pairs that have profiles
    tasks = []
    for entry in data:
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer = entry["answer"]
        for n in n_values:
            if load_profile(qid, n) is not None:
                tasks.append(
                    answer_and_judge(client, qid, qtype, question, answer, n, sem)
                )

    print(f"Found {len(tasks)} profile-question pairs to evaluate")

    results = []
    with tqdm(total=len(tasks), desc="Phase 2: QA + Judge") as pbar:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            pbar.update(1)

    # Save raw results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved to {RESULTS_PATH}")

    # Aggregate by N
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

    aggregate = {
        "overall": {},
        "by_type": {},
    }
    for n in sorted(n2total.keys()):
        acc = n2correct[n] / n2total[n] if n2total[n] > 0 else 0
        aggregate["overall"][str(n)] = {
            "accuracy": round(acc, 4),
            "correct": n2correct[n],
            "total": n2total[n],
        }

    for (n, qtype), correct in sorted(n2type_correct.items()):
        total = n2type_total[(n, qtype)]
        acc = correct / total if total > 0 else 0
        if str(n) not in aggregate["by_type"]:
            aggregate["by_type"][str(n)] = {}
        aggregate["by_type"][str(n)][qtype] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total,
        }

    with AGGREGATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"\nAggregate results saved to {AGGREGATE_PATH}")
    print("\n=== Accuracy by N ===")
    for n in sorted(n2total.keys()):
        acc = n2correct[n] / n2total[n] if n2total[n] > 0 else 0
        print(f"  N={n:>3}: {acc:.4f} ({n2correct[n]}/{n2total[n]})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: QA evaluation")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    asyncio.run(main_async(args.n_values, args.max_questions, args.concurrency))


if __name__ == "__main__":
    main()
