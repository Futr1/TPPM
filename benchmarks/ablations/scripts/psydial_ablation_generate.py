#!/usr/bin/env python3
"""PsyDial ablation: generate counselor responses using DeepSeek API.

For each variant, loads the variant memory bank, injects TPPM memory
into system prompt, generates responses for all 1278 D101 test cases.

Context structure (Mini-Agent-5-1 style):
    System prompt (Chinese counselor) + TPPM memory block
    Full dialogue history (user/assistant turns)

Usage:
    python3 psydial_ablation_generate.py --variant baseline
    python3 psydial_ablation_generate.py --variant ablation_consolidation --max-cases 30
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/ablations'
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
SNAPSHOTS_DIR = ROOT / "memory_snapshots" / "psydial"
EVAL_DIR = ROOT / "eval_results" / "psydial"

# ===== DeepSeek API =====
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
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "你是一名经验丰富的专业心理咨询师。"
    "请根据对话历史，直接给出下一句回复。"
    "只输出回复文本本身，不要输出思考过程、分析、解释或任何额外内容。"
)

ATTRIBUTE_LABELS = {
    "stressor": "压力来源",
    "affective_state": "情绪状态",
    "coping_style": "应对方式",
}


# ===== Data loading =====

def load_d101(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_bank(path: Path) -> dict[int, list[dict]]:
    if not path.exists():
        print(f"[WARN] Memory bank not found: {path}")
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


# ===== Memory formatting =====

def format_memory_block(memories: list[dict], include_evidence: bool = True) -> str:
    """Format TPPM memories in Mini-Agent-5-1 modular style (Chinese labels)."""
    if not memories:
        return ""

    lines = []
    for i, mem in enumerate(memories, 1):
        attr = str(mem.get("attribute", "profile")).strip()
        label = ATTRIBUTE_LABELS.get(attr, attr)
        value = str(mem.get("value", "")).strip()
        if not value:
            continue

        phi = mem.get("phi")
        tier = mem.get("tier", "context_only")
        parts = [f"{i}. {label}: {value}"]

        if isinstance(phi, (int, float)):
            parts.append(f"显著性={float(phi):.3f}")
        parts.append(f"level={tier}")

        if include_evidence:
            evidence = str(mem.get("evidence", "")).strip()
            if evidence:
                parts.append(f"依据={evidence[:120]}")

        lines.append("；".join(parts))

    return "\n".join(lines) if lines else ""


# ===== Message building =====

def build_messages(
    case: dict,
    memory_index: dict[int, list[dict]],
    include_evidence: bool = True,
) -> tuple[list[dict], str | None]:
    """Build messages for a single case with TPPM memory.

    Returns (messages, fallback_reason).
    """
    case_idx = case["idx"]
    msgs = case["messages"]

    # Fallback: short dialogue
    if len(msgs) <= 2:
        last_user = msgs[-1] if msgs else {"role": "user", "content": ""}
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": last_user.get("content", "")},
        ], "insufficient_history"

    # Look up memories
    memories = memory_index.get(case_idx)
    if memories is None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *msgs,
        ], "extraction_failed"

    if not memories:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *msgs,
        ], "no_memories_above_threshold"

    # Normal path: TPPM memories available
    memory_text = format_memory_block(memories, include_evidence=include_evidence)

    system_content = SYSTEM_PROMPT
    if memory_text:
        system_content += (
            f"\n\n【来访者心理画像 — 内部参考】\n"
            f"{memory_text}\n\n"
            f"注意：请自然运用画像信息理解来访者，"
            f"不要在回复中直接复述画像内容或提及记忆系统。"
        )

    return [
        {"role": "system", "content": system_content},
        *msgs,
    ], None


# ===== Async generation =====

async def _generate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    messages: list[dict],
    case_idx: int,
) -> tuple[int, str]:
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=API_MODEL,
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = resp.choices[0].message.content or ""
                return case_idx, content.strip()
            except Exception:
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(min(30.0, 2 ** attempt))
    return case_idx, ""


async def generate_all(
    test_cases: list[dict],
    memory_index: dict[int, list[dict]],
    include_evidence: bool = True,
) -> list[dict]:
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)

    # Build all message sets
    tasks_info: list[tuple[list[dict], int, str | None]] = []
    for case in test_cases:
        msgs, fallback = build_messages(case, memory_index, include_evidence)
        tasks_info.append((msgs, case["idx"], fallback))

    # Run generation
    results_map: dict[int, str] = {}
    pending = [(msgs, idx) for msgs, idx, _ in tasks_info]
    round_num = 0

    while pending:
        round_num += 1
        if round_num > 1:
            delay = min(60.0, 2 ** round_num)
            print(f"[INFO] Retry round {round_num}: {len(pending)} failed, waiting {delay:.0f}s...")
            time.sleep(delay)

        tasks = [_generate_one(client, sem, msgs, idx) for msgs, idx in pending]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        next_pending: list[tuple[list[dict], int]] = []
        for item, (msgs, idx) in zip(outputs, pending):
            if isinstance(item, Exception):
                next_pending.append((msgs, idx))
            else:
                results_map[item[0]] = item[1]

        pending = next_pending
        if not pending:
            break

    # Build results
    fallback_map = {idx: fb for _, idx, fb in tasks_info}
    results = []
    for case in test_cases:
        idx = case["idx"]
        generated = results_map.get(idx, "")
        entry = {
            "idx": idx,
            "golden": case["golden"]["content"],
            "generated": generated,
        }
        if idx in fallback_map and fallback_map[idx]:
            entry["fallback_reason"] = fallback_map[idx]
        results.append(entry)

    return results


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PsyDial ablation: generate counselor responses.")
    parser.add_argument("--variant", type=str, default="baseline",
                        help="Variant ID")
    parser.add_argument("--memory-bank", type=Path, default=None,
                        help="Override memory bank path")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-evidence", action="store_true",
                        help="Strip evidence from memory output")
    args = parser.parse_args()

    # Resolve paths
    if args.memory_bank:
        bank_path = args.memory_bank
    else:
        bank_path = SNAPSHOTS_DIR / args.variant / "d101_tppm_memory_bank.json"

    if args.output:
        output_path = args.output
    else:
        output_path = EVAL_DIR / args.variant / "generations.json"

    # Load data
    test_cases = load_d101(D101_PATH)
    if args.max_cases:
        test_cases = test_cases[:args.max_cases]

    memory_index = load_memory_bank(bank_path)

    include_evidence = not args.no_evidence

    print(f"[INFO] Variant: {args.variant}")
    print(f"[INFO] Memory bank: {bank_path} ({len(memory_index)} cases)")
    print(f"[INFO] Test cases: {len(test_cases)}")
    print(f"[INFO] Include evidence: {include_evidence}")
    print(f"[INFO] Output: {output_path}")

    # Generate
    results = asyncio.run(generate_all(
        test_cases, memory_index, include_evidence=include_evidence,
    ))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "variant": args.variant,
                "model": API_MODEL,
                "test_cases": len(results),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    # Stats
    n_fallback = sum(1 for r in results if "fallback_reason" in r)
    n_empty = sum(1 for r in results if not r["generated"])
    print(f"\n[DONE] {output_path}")
    print(f"[DONE] Generated: {len(results)}, fallback: {n_fallback}, empty: {n_empty}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
