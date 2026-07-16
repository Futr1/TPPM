#!/usr/bin/env python3
"""TPPM-enhanced response generation for PsyDial-D101 using DeepSeek-V4-Pro API.

Loads TPPM psychological profiles from the memory bank, injects them into
the system prompt alongside conversation history, and generates counselor
responses via the DeepSeek API.

Pipeline:
    1. Load D101 test set (1278 cases)
    2. Load TPPM memory bank (from tppm_extract_d101.py)
    3. For each case, build messages with TPPM profile injection
    4. Call DeepSeek-V4-Pro API (async, 8-concurrent) to generate responses
    5. Save per-case generations (golden vs generated, with fallback metadata)

Usage:
    python3 generate_responses.py                              # full 1278 cases
    python3 generate_responses.py --max-cases 5                # smoke test
    python3 generate_responses.py --dry-run                    # print prompts only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table1-data")
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
DEFAULT_MEMORY_BANK = ROOT / "outputs" / "d101_tppm_memory_bank_v2.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "eval" / "d101_full" / "tppm_memory_generations_v2.json"

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
REQUEST_TIMEOUT = 120.0
MAX_TOKENS = 256       # same as original vLLM config
CONCURRENCY = 8
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# ===== Prompt Constants =====
ATTRIBUTE_LABELS = {
    "stressor": "压力来源",
    "affective_state": "情绪状态",
    "coping_style": "应对方式",
}

SYSTEM_PROMPT = (
    "你是一名经验丰富的专业心理咨询师。"
    "请根据对话历史，直接给出下一句回复。"
    "只输出回复文本本身，不要输出思考过程、分析、解释或任何额外内容。"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===== Memory Bank Loading =====


def load_d101(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_bank(path: Path) -> dict[int, list[dict]]:
    """Load D101 TPPM memory bank and index by case_idx.

    Returns:
        dict mapping case_idx (int) -> list of memory dicts.
    """
    if not path.exists():
        print(f"[WARN] Memory bank not found: {path}. All cases will fall back.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        bank = json.load(f)

    indexed: dict[int, list[dict]] = {}
    for entry in bank.get("memories", []):
        case_idx = entry.get("case_idx")
        memories = entry.get("tppm_memory", [])
        if case_idx is not None and isinstance(memories, list):
            indexed[int(case_idx)] = [m for m in memories if isinstance(m, dict)]
    return indexed


def format_memory_background(memories: list[dict]) -> str:
    """Format TPPM memories as structured psychological profile text block."""
    if not memories:
        return "暂无可用的长期画像背景。"

    lines = []
    for i, mem in enumerate(memories, 1):
        attr = str(mem.get("attribute", "profile")).strip()
        label = ATTRIBUTE_LABELS.get(attr, attr)
        value = str(mem.get("value", "")).strip()
        evidence = str(mem.get("evidence", "")).strip()
        phi = mem.get("phi")
        if not value:
            continue
        suffix = ""
        if isinstance(phi, (int, float)):
            suffix += f";显著性={float(phi):.3f}"
        if evidence:
            suffix += f";简要依据={evidence[:120]}"
        lines.append(f"{i}. {label}: {value}{suffix}")
    return "\n".join(lines) if lines else "暂无可用的长期画像背景。"


# ===== Prompt Construction =====


def build_messages_no_memory(case: dict) -> list[dict]:
    """Fallback: only the last user turn."""
    msgs = case["messages"]
    last_user = None
    for m in reversed(msgs):
        if m["role"] == "user":
            last_user = m
            break
    if last_user is None:
        raise ValueError(f"No user message found in case {case['idx']}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": last_user["content"]},
    ]


def build_messages_long_context(case: dict) -> list[dict]:
    """Fallback: full conversation history."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *case["messages"],
    ]


def build_messages_tppm_memory(
    case: dict,
    memory_index: dict[int, list[dict]],
) -> tuple[list[dict], str | None]:
    """Build messages with TPPM psychological profile as context.

    Returns:
        (messages, fallback_reason)
        - messages: list of {"role": ..., "content": ...} for the model.
        - fallback_reason: None if TPPM memories were used, or a string
          describing why a fallback was triggered.
    """
    case_idx = case["idx"]
    msgs = case["messages"]

    # Fallback 1: extremely short dialogue
    if len(msgs) <= 2:
        return build_messages_no_memory(case), "insufficient_history"

    # Look up TPPM memories
    memories = memory_index.get(case_idx)

    # Fallback 2: extraction failed (not in bank at all)
    if memories is None:
        return build_messages_long_context(case), "extraction_failed"

    # Fallback 3: no memories above threshold
    if not memories:
        return build_messages_long_context(case), "no_memories_above_threshold"

    # Normal path: TPPM memories available
    memory_text = format_memory_background(memories)

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"【来访者长期画像 — 内部参考】\n"
        f"{memory_text}\n\n"
        f"注意：请自然运用画像信息理解来访者，"
        f"不要在回复中直接复述画像内容或提及记忆系统。"
    )

    return [
        {"role": "system", "content": system_content},
        *case["messages"],
    ], None


# ===== DeepSeek API Generation =====


def build_client(api_key: str = API_KEY, api_base: str = API_BASE) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=REQUEST_TIMEOUT)


async def generate_one_response(
    messages: list[dict],
    case_idx: int,
    client: AsyncOpenAI,
    model: str = API_MODEL,
    max_retries: int = MAX_RETRIES,
    max_tokens: int = MAX_TOKENS,
) -> str | None:
    """Call DeepSeek-V4-Pro API to generate a counselor response."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=messages,
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if not content:
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
                return None
            return content
        except Exception as exc:
            if attempt >= max_retries:
                print(f"  [FAIL] case_idx={case_idx} after {max_retries} attempts: {exc}")
                return None
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    return None


async def run_generation(
    test_cases: list[dict],
    memory_index: dict[int, list[dict]],
    model: str = API_MODEL,
    concurrency: int = CONCURRENCY,
    api_key: str = API_KEY,
    api_base: str = API_BASE,
    min_turns: int = 1,
) -> tuple[list[dict], int]:
    """Run async concurrent generation over all test cases.

    Returns:
        results: list of per-case result dicts
        skipped: count of cases skipped (min_turns)
    """
    client = build_client(api_key=api_key, api_base=api_base)
    sem = asyncio.Semaphore(concurrency)

    skipped = 0
    tasks: list[asyncio.Task] = []

    for case in test_cases:
        if len(case["messages"]) < min_turns:
            skipped += 1
            continue

        messages, fallback_reason = build_messages_tppm_memory(case, memory_index)

        async def gen_one(msgs=messages, fb=fallback_reason, c=case):
            async with sem:
                generated = await generate_one_response(
                    msgs, c["idx"], client, model=model,
                )
                entry = {
                    "idx": c["idx"],
                    "golden": c["golden"]["content"],
                    "generated": generated or "",
                }
                if fb:
                    entry["fallback_reason"] = fb
                return entry

        tasks.append(asyncio.create_task(gen_one()))

    print(f"[INFO] Total test cases: {len(test_cases)}")
    print(f"[INFO] Skipped (min_turns={min_turns}): {skipped}")
    print(f"[INFO] Generation tasks: {len(tasks)}")
    print(f"[INFO] Concurrency: {concurrency}")

    results = []
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses")
    for coro in progress:
        result = await coro
        results.append(result)

    results.sort(key=lambda r: r["idx"])
    return results, skipped


# ===== Checkpoint Support =====


def load_checkpoint(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["idx"]: r for r in data.get("results", [])}


def save_checkpoint(output_path: Path, metadata: dict, results: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": sorted(results, key=lambda x: x["idx"]),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ===== Main =====


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TPPM-enhanced response generation (DeepSeek-V4-Pro)."
    )
    parser.add_argument("--d101", type=Path, default=D101_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-bank", type=Path, default=DEFAULT_MEMORY_BANK)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without making API calls",
    )
    args = parser.parse_args()

    # Load data
    test_cases = load_d101(args.d101)
    if args.max_cases:
        test_cases = test_cases[:args.max_cases]
    print(f"[INFO] Loaded {len(test_cases)} test cases from {args.d101}")

    # Load memory bank
    memory_index = load_memory_bank(args.memory_bank)
    print(f"[INFO] Memory bank: {len(memory_index)} cases indexed")

    # Dry run: print prompts for first few cases
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — Printing prompts (no API calls)")
        print("=" * 60)
        for case in test_cases[:3]:
            messages, fallback = build_messages_tppm_memory(case, memory_index)
            print(f"\n--- Case idx={case['idx']} (fallback={fallback}) ---")
            for msg in messages:
                role = msg["role"]
                content = msg["content"][:300]
                print(f"  [{role}] {content}")
            print()
        return 0

    # Load checkpoint
    existing = load_checkpoint(args.output)
    print(f"[INFO] Existing results in checkpoint: {len(existing)}")

    # Filter out already-done cases
    remaining = [c for c in test_cases if c["idx"] not in existing]
    print(f"[INFO] Remaining to generate: {len(remaining)}")

    if not remaining:
        print("[INFO] All cases already generated. Nothing to do.")
        return 0

    # Run generation
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] API base: {args.api_base}")
    print(f"[INFO] Max tokens: {args.max_tokens}")

    results, skipped = asyncio.run(
        run_generation(
            remaining, memory_index,
            model=args.model, concurrency=args.concurrency,
            api_key=args.api_key, api_base=args.api_base,
            min_turns=args.min_turns,
        )
    )

    # Merge with existing
    all_results = list(existing.values()) + results
    all_results.sort(key=lambda r: r["idx"])

    # Count stats
    fallback_count = sum(1 for r in all_results if "fallback_reason" in r)
    valid_count = len(all_results) - fallback_count
    empty_count = sum(1 for r in all_results if not r["generated"])
    print(f"\n[STATS] Total: {len(all_results)}, Valid (TPPM): {valid_count}, Fallback: {fallback_count}, Empty: {empty_count}")

    # Save
    metadata = {
        "method": "tppm_memory",
        "model": args.model,
        "test_cases": len(all_results),
        "min_turns": args.min_turns,
        "generated_at": utc_now(),
    }
    save_checkpoint(args.output, metadata, all_results)
    print(f"[SAVED] {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
