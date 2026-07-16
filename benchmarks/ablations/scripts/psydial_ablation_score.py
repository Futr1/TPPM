#!/usr/bin/env python3
"""PsyDial ablation: score generated responses using DeepSeek API (9-dimension).

Async concurrent version — scores all 9 dimensions per case in parallel.

Usage:
    python3 psydial_ablation_score.py --variant baseline
    python3 psydial_ablation_score.py --variant ablation_consolidation --concurrency 16
    python3 psydial_ablation_score.py --variant baseline --resume
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

import aiohttp

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/ablations'
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
EVAL_DIR = ROOT / "eval_results" / "psydial"

# ===== API Config =====
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )
API_BASE = "https://api.deepseek.com"
API_URL = f"{API_BASE}/chat/completions"
MODEL_NAME = "deepseek-v4-flash"
CONCURRENCY = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=90, connect=30)

# ===== Scoring dimensions =====
DIMENSION_KEYS = [
    "empathy", "active_listening", "issue_clarification",
    "open_ended_questioning", "encouraging_self_exploration",
    "cognitive_restructuring", "guided_questioning",
    "non_judgmental_accepting_attitude", "overall_assessment",
]

METRICS_DEFINITIONS = {
    "empathy": {
        "name": "Empathy", "abbr": "Emp",
        "definition": "Empathy refers to the counselor's ability to understand, resonate with, and validate the client's emotions and experiences.",
        "criteria": "- 5: Deep empathy, consistently validating.\n- 4: Shows empathy, occasionally lacks depth.\n- 3: Basic empathy, somewhat distant.\n- 2: Weak empathy, superficial.\n- 1: No empathy, indifferent.",
    },
    "active_listening": {
        "name": "Active Listening", "abbr": "AL",
        "definition": "Active listening ensures thorough understanding of the client's problems and emotions through attentive listening to verbal and non-verbal cues.",
        "criteria": "- 5: Listens attentively, accurately reflects feelings.\n- 4: Listens well, occasionally misses details.\n- 3: Basic listening, misses key details.\n- 2: Partial listening, misses important cues.\n- 1: Does not listen actively.",
    },
    "issue_clarification": {
        "name": "Issue Clarification", "abbr": "IC",
        "definition": "Clarification involves seeking additional details when the client's communication is unclear.",
        "criteria": "- 5: Actively seeks clarification with precise questions.\n- 4: Seeks clarification in most situations.\n- 3: Some clarifying questions, misses key aspects.\n- 2: Rarely asks for clarification.\n- 1: No clarification, poor understanding.",
    },
    "open_ended_questioning": {
        "name": "Open-ended Questioning", "abbr": "OEQ",
        "definition": "Open-ended questions encourage the client to elaborate and explore their thoughts and feelings in depth.",
        "criteria": "- 5: Consistently uses effective open-ended questions.\n- 4: Uses open-ended questions with minor gaps.\n- 3: Some open-ended questions, often closed.\n- 2: Rarely uses open-ended questions.\n- 1: No open-ended questions.",
    },
    "encouraging_self_exploration": {
        "name": "Encouraging Self-Exploration", "abbr": "ESE",
        "definition": "Encouraging self-exploration involves guiding the client to reflect on their own thoughts, feelings, and behaviors.",
        "criteria": "- 5: Consistently encourages deep self-reflection.\n- 4: Encourages self-exploration with minor gaps.\n- 3: Some encouragement, often directive.\n- 2: Rarely encourages self-exploration.\n- 1: Does not encourage self-exploration.",
    },
    "cognitive_restructuring": {
        "name": "Cognitive Restructuring", "abbr": "CR",
        "definition": "Cognitive restructuring helps clients identify and challenge negative or irrational thought patterns.",
        "criteria": "- 5: Expertly guides client to reframe thoughts.\n- 4: Good cognitive restructuring with minor gaps.\n- 3: Basic restructuring, misses opportunities.\n- 2: Rarely applies cognitive restructuring.\n- 1: No cognitive restructuring.",
    },
    "guided_questioning": {
        "name": "Guided Questioning", "abbr": "GQ",
        "definition": "Guided questioning uses strategic questions to lead the client toward insights and solutions.",
        "criteria": "- 5: Strategic questions lead to deep insights.\n- 4: Good questioning with minor gaps.\n- 3: Some guided questions, often unfocused.\n- 2: Rarely uses guided questioning.\n- 1: No guided questioning.",
    },
    "non_judgmental_accepting_attitude": {
        "name": "Non-judgmental Accepting Attitude", "abbr": "NJAA",
        "definition": "A non-judgmental accepting attitude creates a safe space where the client feels accepted without criticism.",
        "criteria": "- 5: Fully accepting, no judgment whatsoever.\n- 4: Mostly accepting with very minor judgmental hints.\n- 3: Generally accepting but some judgmental moments.\n- 2: Noticeably judgmental at times.\n- 1: Highly judgmental.",
    },
    "overall_assessment": {
        "name": "Overall Assessment", "abbr": "OA",
        "definition": "Overall assessment evaluates the counselor's response holistically, considering all counseling skills and therapeutic effectiveness.",
        "criteria": "- 5: Excellent overall counseling response.\n- 4: Good response with minor areas for improvement.\n- 3: Average response with notable gaps.\n- 2: Poor response with several issues.\n- 1: Very poor response, hindering therapy.",
    },
}

SCORING_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing counseling skills in psychological counseling dialogues. "
    "Evaluate the counselor's response based on the given dimension and criteria."
)

# ===== Helpers =====

def format_dialogue_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "Client" if msg.get("role") == "user" else "Counselor"
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "[No dialogue history]"

def build_scoring_prompt(ctx: str, response: str, metric_key: str) -> str:
    m = METRICS_DEFINITIONS[metric_key]
    return (
        f"The following is a counseling context.\n"
        f"Dialogue history: {ctx}\n"
        f"Counselor's response: {response}\n\n"
        f"{m['name']} ({m['abbr']})\n\n"
        f"Definition: {m['definition']}\n\n"
        f"Rating Criteria:\n{m['criteria']}\n\n"
        f"Provide a brief reasoning for your rating based on these criteria, "
        f"and then assign a numerical rating. Provide your answer in the following format.\n\n"
        f"- Reasoning: (Your explanation here)\n"
        f"- Rating: (Ranging from 1 to 5)"
    )

def parse_response(content: str) -> dict | None:
    reason_match = re.search(r'- Reasoning:\s*(.+?)(?=\n- Rating:|\Z)', content, re.DOTALL)
    reasoning = reason_match.group(1).strip() if reason_match else ""
    rating_match = re.search(r'- Rating:\s*(\d+)', content)
    if rating_match:
        rating = int(rating_match.group(1))
        if 1 <= rating <= 5:
            return {"reasoning": reasoning, "rating": rating}
    return None

# ===== Async scoring =====

async def score_one_dimension(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    metric_key: str,
) -> dict:
    """Score one dimension asynchronously."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
    }

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        result = parse_response(content)
                        if result:
                            return result
                        return {"reasoning": "PARSE_FAILED", "rating": 0}
                    elif resp.status == 429:
                        wait = min(2 ** attempt * 5, 60)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        text = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {"reasoning": f"API_ERROR_{resp.status}", "rating": 0}
            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"reasoning": "TIMEOUT", "rating": 0}

    return {"reasoning": "MAX_RETRIES", "rating": 0}

async def score_case(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    ctx: str,
    response: str,
) -> dict[str, dict]:
    """Score all 9 dimensions for one case concurrently."""
    tasks = []
    for metric_key in DIMENSION_KEYS:
        prompt = build_scoring_prompt(ctx, response, metric_key)
        tasks.append(score_one_dimension(session, sem, SCORING_SYSTEM_PROMPT, prompt, metric_key))

    results = await asyncio.gather(*tasks)
    return {DIMENSION_KEYS[i]: r for i, r in enumerate(results)}

async def score_all_cases(
    results: list[dict],
    d101_messages: dict[int, list[dict]],
    existing: dict[int, dict],
    concurrency: int,
    checkpoint_path: Path,
    metadata: dict,
) -> list[dict]:
    """Score all cases with async concurrency."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 2)

    scores_list = list(existing.values())
    scored = 0
    errors = 0

    # Filter cases that need scoring
    to_score = [case for case in results if case["idx"] not in existing]
    print(f"[INFO] Cases to score: {len(to_score)} (existing: {len(existing)})")

    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches for checkpointing
        batch_size = 50
        for batch_start in range(0, len(to_score), batch_size):
            batch = to_score[batch_start:batch_start + batch_size]

            # Build tasks for this batch
            batch_tasks = []
            for case in batch:
                idx = case["idx"]
                dialogue = d101_messages.get(idx, [])
                ctx = format_dialogue_history(dialogue)
                response = case.get("generated", "")
                batch_tasks.append(score_case(session, sem, ctx, response))

            # Run batch concurrently
            batch_results = await asyncio.gather(*batch_tasks)

            # Collect results
            for case, scores_for_case in zip(batch, batch_results):
                idx = case["idx"]
                entry = {
                    "idx": idx,
                    "golden": case.get("golden", ""),
                    "generated": case.get("generated", ""),
                    "fallback_reason": case.get("fallback_reason"),
                }
                entry.update(scores_for_case)
                scores_list.append(entry)
                scored += 1

                # Count errors
                for k in DIMENSION_KEYS:
                    if scores_for_case[k]["rating"] == 0:
                        errors += 1

            # Progress
            total_done = len(existing) + scored
            total_cases = len(results)
            print(f"  [PROGRESS] {total_done}/{total_cases} cases scored | Errors: {errors}")

            # Checkpoint
            save_checkpoint(checkpoint_path, metadata, scores_list)

    return scores_list

# ===== Checkpoint / Summary =====

def save_checkpoint(output_path: Path, metadata: dict, scores: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "scores": sorted(scores, key=lambda x: x["idx"]),
        }, f, ensure_ascii=False, indent=2)

def compute_summary(scores: list[dict]) -> dict:
    summary = {}
    for key in DIMENSION_KEYS:
        ratings = [s[key]["rating"] for s in scores if s.get(key, {}).get("rating", 0) > 0]
        if ratings:
            mean = sum(ratings) / len(ratings)
            summary[key] = {
                "mean": round(mean, 3),
                "std": round((sum((r - mean) ** 2 for r in ratings) / len(ratings)) ** 0.5, 3),
                "count": len(ratings),
            }
        else:
            summary[key] = {"mean": 0, "std": 0, "count": 0}
    return summary

# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PsyDial ablation scoring (9-dimension, async concurrent).")
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--generations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Resolve paths
    gen_path = args.generations or (EVAL_DIR / args.variant / "generations.json")
    out_path = args.output or (EVAL_DIR / args.variant / "scores.json")

    # Load generations
    with gen_path.open("r", encoding="utf-8") as f:
        gen_data = json.load(f)
    results = gen_data["results"]
    if args.max_cases:
        results = results[:args.max_cases]

    # Load D101 for dialogue history
    with D101_PATH.open("r", encoding="utf-8") as f:
        d101_data = json.load(f)
    d101_messages = {item["idx"]: item["messages"] for item in d101_data}

    # Load checkpoint
    existing = {}
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        existing = {item["idx"]: item for item in data.get("scores", [])}

    print(f"[INFO] Variant: {args.variant}")
    print(f"[INFO] Generations: {gen_path} ({len(results)} cases)")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Existing scores: {len(existing)}")
    print(f"[INFO] Concurrency: {args.concurrency}")

    # Run async scoring
    t0 = time.time()
    scores_list = asyncio.run(score_all_cases(
        results, d101_messages, existing, args.concurrency,
        out_path, gen_data.get("metadata", {}),
    ))
    elapsed = time.time() - t0

    # Final save
    save_checkpoint(out_path, gen_data.get("metadata", {}), scores_list)

    # Summary
    if scores_list:
        summary = compute_summary(scores_list)
        print(f"\n{'=' * 60}")
        print(f"PsyDial Scoring — {args.variant} (n={len(scores_list)}, {elapsed:.0f}s)")
        print(f"{'=' * 60}")
        print(f"{'Dimension':<35} {'Mean':>6}  {'Std':>6}")
        print("-" * 50)
        for key in DIMENSION_KEYS:
            m = METRICS_DEFINITIONS[key]
            s = summary[key]
            print(f"{m['name']:<35} {s['mean']:6.3f}  {s['std']:6.3f}")
        print(f"{'=' * 60}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
