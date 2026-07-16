#!/usr/bin/env python3
"""LoCoMo ablation QA eval — Mini-Agent-5-1 context structure.

Context per QA:
    [Current Session full text]  — only the session from evidence field
    [Temporal Profile Memory]    — top-5 retrieved PMUs, Mini-Agent format
    Question

Variants via --memory-bank:
    baseline, ablation_consolidation, ablation_branching, ablation_decay

Evaluation-time modifications:
    --no-ltm      → w/o Long-term Retrieval (exclude long_term_memory)
    --no-evidence → w/o Evidence Collection (strip evidence from output)
    --no-memory   → no TPPM memory at all (conversation-only baseline)

Usage:
    python3 locomo_ablation_eval.py --variant baseline
    python3 locomo_ablation_eval.py --variant ablation_branching --max-questions 20
    python3 locomo_ablation_eval.py --variant baseline --no-ltm
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
from typing import Any

import numpy as np
from openai import AsyncOpenAI

_LOCOMO_ROOT = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo")
if str(_LOCOMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCOMO_ROOT))

from task_eval.evaluation import eval_question_answering

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Ablation")
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
SNAPSHOTS_DIR = ROOT / "memory_snapshots" / "locomo"
EVAL_DIR = ROOT / "eval_results" / "locomo"

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
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 5
MAX_TOKENS = 256

QA_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Your job is to understand the following conversation and answer questions based on it. "
    "A structured speaker profile extracted from earlier conversations is also provided — "
    "use it to infer answers when the conversation text alone is insufficient. "
    "Write a short answer in a few words. Do not write complete sentences. "
    "Answer with exact words from the conversations whenever possible. "
    "If you don't know the answer, please don't share false information."
)


# ===== Data loading =====

def load_memory_bank(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Memory bank not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    indexed: dict[str, dict[str, Any]] = {}
    for entry in bank.get("conversations", []):
        cid = entry.get("conv_id", "")
        if cid:
            indexed[cid] = entry
    return indexed


def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_sorted_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict]]]:
    conv_data = conv["conversation"]
    sessions: list[tuple[int, str, list[dict]]] = []
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.replace("session_", ""))
            except ValueError:
                continue
            sessions.append((num, key, conv_data[key]))
    sessions.sort(key=lambda x: x[0])
    return sessions


# ===== Session mapping from evidence =====

def get_evidence_session(evidence: list[str]) -> int | None:
    """Extract session number from evidence field (D{N}:{turn} format).

    Returns the session number from the first evidence entry.
    Returns None if no valid evidence found.
    """
    if not evidence:
        return None
    for ev in evidence:
        if isinstance(ev, str):
            # Handle "D1:3" or "D1:3; D2:5" formats
            parts = ev.split(";")
            for part in parts:
                part = part.strip()
                m = re.match(r"D(\d+):\d+", part)
                if m:
                    return int(m.group(1))
    return None


def get_session_text(conv: dict[str, Any], session_num: int) -> str:
    """Get full text of a specific session."""
    key = f"session_{session_num}"
    turns = conv.get("conversation", {}).get(key, [])
    if not turns:
        return ""
    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        if speaker and text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# ===== TPPM memory retrieval (Mini-Agent-5-1 style) =====

def _text_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity."""
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r'\w+', a.lower()))
    tokens_b = set(re.findall(r'\w+', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _get_scene_branch_value(pmu: dict[str, Any], scene: str) -> tuple[str, str]:
    """Get the best branch value for a given scene.

    Returns (value, branch_scene).
    """
    branches = pmu.get("scene_branches", {})
    if scene in branches:
        return str(branches[scene].get("value", pmu.get("value", ""))), scene
    if "general" in branches:
        return str(branches["general"].get("value", pmu.get("value", ""))), "general"
    if branches:
        best_scene = max(branches.keys(),
                         key=lambda s: branches[s].get("reinforcement_count", 0))
        return str(branches[best_scene].get("value", pmu.get("value", ""))), best_scene
    return str(pmu.get("value", "")), pmu.get("scene", "general")


def _retrieve_score(pmu: dict[str, Any], question: str, scene: str) -> float:
    """Compute retrieval score for a PMU given a question and scene."""
    branch_value, branch_scene = _get_scene_branch_value(pmu, scene)
    canonical_value = str(pmu.get("value", ""))

    # Relevance
    rel = max(
        _text_similarity(question, branch_value),
        _text_similarity(question, canonical_value),
    )

    # Scene match
    if branch_scene == scene:
        scene_score = 1.0
    elif branch_scene == "general" or scene == "general":
        scene_score = 0.7
    else:
        scene_score = 0.4

    # Context match
    branch_context = ""
    branches = pmu.get("scene_branches", {})
    if scene in branches:
        branch_context = str(branches[scene].get("context", ""))
    ctx_score = max(
        _text_similarity(question, branch_context),
        _text_similarity(question, str(pmu.get("context", ""))),
    )

    # Attribute/type keyword match
    attr = pmu.get("attribute", "").lower()
    ptype = pmu.get("profile_type", "").lower()
    q_lower = question.lower()
    if attr in q_lower or ptype in q_lower:
        ctx_score = max(ctx_score, 0.5)

    stability = float(pmu.get("stability_score", 0))
    quality = float(pmu.get("quality_score", 0))

    # Weights: relevance 0.35, stability 0.20, context 0.15, scene 0.20, quality 0.10
    return (0.35 * rel + 0.20 * stability + 0.15 * ctx_score
            + 0.20 * scene_score + 0.10 * quality)


def retrieve_top_k(
    memory_entry: dict[str, Any],
    question: str,
    scene: str = "general",
    top_k: int = 5,
    no_ltm: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve top-k PMUs by weighted score."""
    working = memory_entry.get("working_memory", [])
    short_term = memory_entry.get("short_term_memory", [])
    long_term = [] if no_ltm else memory_entry.get("long_term_memory", [])

    all_memories = working + short_term + long_term
    if not all_memories:
        return []

    scored = []
    for pmu in all_memories:
        score = _retrieve_score(pmu, question, scene)
        scored.append((score, pmu))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pmu for _, pmu in scored[:top_k]]


# ===== Memory formatting (Mini-Agent-5-1 modular format) =====

def format_tppm_memory_block(
    retrieved_pmus: list[dict[str, Any]],
    include_evidence: bool = True,
) -> str:
    """Format retrieved PMUs as [Temporal Profile Memory] block.

    Mini-Agent-5-1 format:
        - attribute: value (type=..., scene=..., stability=..., level=..., evidence=...)
    """
    if not retrieved_pmus:
        return ""

    lines = ["[Temporal Profile Memory]"]
    for pmu in retrieved_pmus:
        attr = pmu.get("attribute", "?")
        value = str(pmu.get("value", "?")).strip()
        ptype = pmu.get("profile_type", "general")
        scene = pmu.get("scene", "general")
        stability = float(pmu.get("stability_score", 0))
        level = pmu.get("memory_level", "short_term")

        parts = [
            f"- {attr}: {value}",
            f"(type={ptype}",
            f"scene={scene}",
            f"stability={stability:.2f}",
            f"level={level}",
        ]

        if include_evidence:
            evidence_list = pmu.get("evidence", [])
            if evidence_list:
                ev = evidence_list[0]
                ev_content = str(ev.get("content", ""))[:80]
                ev_time = str(ev.get("timestamp", ""))[:19]
                parts.append(f"evidence_time={ev_time}")
                parts.append(f'evidence={ev_content})')
            else:
                parts.append(")")
        else:
            parts.append(")")

        line = ", ".join(parts[:-1]) + parts[-1] if len(parts) > 2 else parts[0] + parts[-1]
        # Clean up formatting
        line = f"- {attr}: {value} (type={ptype}, scene={scene}, stability={stability:.2f}, level={level}"
        if include_evidence:
            evidence_list = pmu.get("evidence", [])
            if evidence_list:
                ev = evidence_list[0]
                ev_content = str(ev.get("content", ""))[:80]
                ev_time = str(ev.get("timestamp", ""))[:19]
                line += f", evidence_time={ev_time}, evidence={ev_content})"
            else:
                line += ")"
        else:
            line += ")"
        lines.append(line)

    return "\n".join(lines)


# ===== Context builder =====

def build_ablation_context(
    conv: dict[str, Any],
    memory_entry: dict[str, Any] | None,
    question: str,
    session_num: int | None,
    no_ltm: bool = False,
    include_evidence: bool = True,
    no_memory: bool = False,
) -> str:
    """Build context in Mini-Agent-5-1 style:
    [Current Session] + [Temporal Profile Memory] + Question
    """
    parts: list[str] = []

    # 1. Current session full text
    if session_num is not None:
        session_text = get_session_text(conv, session_num)
        if session_text:
            # Get date if available
            dt_key = f"session_{session_num}_date_time"
            dt = conv.get("conversation", {}).get(dt_key, "")
            dt_label = f" ({dt})" if dt else ""
            parts.append(f"[Session {session_num}{dt_label} — current conversation]")
            parts.append(session_text)
            parts.append("")

    # 2. TPPM memory (top-5 retrieval)
    if memory_entry and not no_memory:
        scene = f"session_{session_num}" if session_num else "general"
        retrieved = retrieve_top_k(
            memory_entry, question, scene=scene,
            top_k=5, no_ltm=no_ltm,
        )
        if retrieved:
            memory_block = format_tppm_memory_block(
                retrieved, include_evidence=include_evidence,
            )
            parts.append(memory_block)
            parts.append("")

    # 3. Question
    parts.append(f"Based on the above, write a short answer for the following "
                 f"question in a few words. Do not write complete sentences. "
                 f"Answer with exact words from the conversations whenever possible.\n\n"
                 f"Question: {question}")

    return "\n".join(parts)


# ===== Async QA generation =====

async def _generate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    messages: list[dict[str, str]],
    conv_idx: int,
    qa_idx: int,
) -> tuple[int, int, str]:
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
                return conv_idx, qa_idx, content.strip()
            except Exception:
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(min(30.0, 2 ** attempt))
    return conv_idx, qa_idx, ""


async def generate_qa_answers(
    conversations: list[dict[str, Any]],
    memory_bank: dict[str, dict[str, Any]],
    no_ltm: bool = False,
    include_evidence: bool = True,
    no_memory: bool = False,
    max_questions: int | None = None,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)

    all_tasks: list[tuple[list[dict[str, str]], int, int]] = []
    count = 0

    for conv_idx, conv in enumerate(conversations):
        cid = conv.get("sample_id", "")
        memory_entry = memory_bank.get(cid)

        for qa_idx, qa_item in enumerate(conv["qa"]):
            if max_questions and count >= max_questions:
                break
            count += 1

            question = qa_item["question"]
            evidence = qa_item.get("evidence", [])
            session_num = get_evidence_session(evidence)

            prompt_text = build_ablation_context(
                conv, memory_entry, question, session_num,
                no_ltm=no_ltm,
                include_evidence=include_evidence,
                no_memory=no_memory,
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            all_tasks.append((messages, conv_idx, qa_idx))

    total = len(all_tasks)
    print(f"[INFO] Generating answers for {total} QA pairs (concurrency={CONCURRENCY})...")

    # Retry loop
    results_map: dict[tuple[int, int], str] = {}
    pending = all_tasks[:]
    round_num = 0

    while pending:
        round_num += 1
        if round_num > 1:
            delay = min(60.0, 2 ** round_num)
            print(f"[INFO] Retry round {round_num}: {len(pending)} failed, waiting {delay:.0f}s...")
            time.sleep(delay)

        tasks = [_generate_one(client, sem, msgs, ci, qi) for msgs, ci, qi in pending]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        next_pending: list[tuple[list[dict[str, str]], int, int]] = []
        for item, (msgs, ci, qi) in zip(outputs, pending):
            if isinstance(item, Exception):
                next_pending.append((msgs, ci, qi))
            else:
                results_map[(ci, qi)] = item[2]

        pending = next_pending
        if not pending:
            break

    if pending:
        print(f"[WARN] {len(pending)} QA pairs failed after {round_num} rounds")

    # Reconstruct results
    results: list[dict[str, Any]] = []
    for conv_idx, conv in enumerate(conversations):
        while len(results) <= conv_idx:
            results.append({
                "sample_id": conversations[len(results)]["sample_id"],
                "qa": [],
            })
        for qa_idx, qa_item in enumerate(conv["qa"]):
            if (conv_idx, qa_idx) not in results_map and max_questions:
                continue
            generated = results_map.get((conv_idx, qa_idx), "")
            generated = re.sub(r'', '', generated, flags=re.DOTALL).strip()
            gt_answer = qa_item.get("answer") or qa_item.get("adversarial_answer", "")
            results[conv_idx]["qa"].append({
                "question": qa_item["question"],
                "answer": gt_answer,
                "category": qa_item["category"],
                "evidence": qa_item.get("evidence", []),
                "tppm_prediction": generated,
            })

    return results


# ===== Evaluation =====

def evaluate_and_save(
    results: list[dict[str, Any]],
    output_path: Path,
    variant_id: str,
) -> dict[str, float]:
    category_names = {
        1: "multi_hop",
        2: "single_hop",
        3: "temporal",
        4: "open_domain",
        5: "adversarial",
    }

    all_f1s: dict[int, list[float]] = {c: [] for c in category_names}

    for conv_result in results:
        qas = conv_result["qa"]
        if not qas:
            continue
        try:
            scores, _, _ = eval_question_answering(qas, eval_key="tppm_prediction")
        except Exception as exc:
            print(f"[WARN] eval failed for {conv_result['sample_id']}: {exc}")
            continue
        for i, qa in enumerate(qas):
            cat = qa["category"]
            if i < len(scores):
                all_f1s[cat].append(scores[i])

    summary: dict[str, float] = {}
    core_means: list[float] = []
    for cat, name in category_names.items():
        scores = all_f1s[cat]
        avg = round(float(np.mean(scores)) * 100, 1) if scores else 0.0
        summary[name] = avg
        if cat != 5 and scores:
            core_means.append(float(np.mean(scores)) * 100)

    summary["overall"] = round(float(np.mean(core_means)), 1) if core_means else 0.0

    payload = {
        "metadata": {
            "variant": variant_id,
            "model": API_MODEL,
            "context_strategy": "mini_agent_style (current session + top-5 TPPM)",
            "num_conversations": len(results),
            "num_qa_pairs": sum(len(r["qa"]) for r in results),
        },
        "summary": summary,
        "per_category_counts": {
            name: len(all_f1s[cat]) for cat, name in category_names.items()
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return summary


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LoCoMo ablation QA eval — Mini-Agent-5-1 context structure.")
    parser.add_argument("--variant", type=str, default="baseline",
                        help="Variant ID (baseline, ablation_consolidation, etc.)")
    parser.add_argument("--memory-bank", type=Path, default=None,
                        help="Override memory bank path (default: auto from variant)")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path (default: auto from variant)")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-convs", type=int, default=None)
    # Evaluation-time ablation flags
    parser.add_argument("--no-ltm", action="store_true",
                        help="Exclude long_term_memory from retrieval")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Strip evidence from memory output")
    parser.add_argument("--no-memory", action="store_true",
                        help="No TPPM memory at all (conversation-only baseline)")
    args = parser.parse_args()

    # Resolve memory bank path
    if args.memory_bank:
        bank_path = args.memory_bank
    else:
        bank_path = SNAPSHOTS_DIR / args.variant / "locomo_memory_bank.json"

    # Resolve output path
    # Determine effective variant name for output directory
    if args.no_ltm:
        output_variant = "ablation_no_ltm"
    elif args.no_evidence:
        output_variant = "ablation_no_evidence"
    elif args.no_memory:
        output_variant = "no_memory"
    else:
        output_variant = args.variant

    if args.output:
        output_path = args.output
    else:
        output_path = EVAL_DIR / output_variant / "qa_results.json"

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]

    memory_bank = {} if args.no_memory else load_memory_bank(bank_path)

    print(f"[INFO] Variant: {output_variant}")
    print(f"[INFO] Memory bank: {bank_path if not args.no_memory else 'NONE'}")
    print(f"[INFO] Conversations: {len(conversations)}")
    print(f"[INFO] No LTM: {args.no_ltm}")
    print(f"[INFO] No Evidence: {args.no_evidence}")
    print(f"[INFO] No Memory: {args.no_memory}")
    print(f"[INFO] Max questions: {args.max_questions or 'all'}")
    print(f"[INFO] Output: {output_path}")

    results = asyncio.run(generate_qa_answers(
        conversations, memory_bank,
        no_ltm=args.no_ltm,
        include_evidence=not args.no_evidence,
        no_memory=args.no_memory,
        max_questions=args.max_questions,
    ))

    summary = evaluate_and_save(results, output_path, output_variant)

    total_qa = sum(len(r["qa"]) for r in results)
    print(f"\n{'='*60}")
    print(f"LoCoMo QA — {output_variant} (n={total_qa})")
    for name, score in summary.items():
        print(f"  {name}: {score:.1f}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
