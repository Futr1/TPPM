#!/usr/bin/env python3
"""Re-run all 5 ablation variants on Suggest (suggest_new_ideas) questions only.

Uses the same v1 context building logic as phase3_ablation.py:
  - Full conversation history (all messages up to end_index)
  - format_memory_block with stability*quality ranking (except semantic_retrieval)

Variants:
  1. ablation_consolidation  → benchmarks/ablations snapshots, v1 formatting
  2. ablation_branching      → benchmarks/ablations snapshots, v1 formatting
  3. ablation_no_evidence    → t3 baseline snapshots, include_evidence=False
  4. ablation_semantic       → t3 baseline snapshots, sort by semantic similarity (not stability*quality)
  5. ablation_flat_pool      → benchmarks/ablations flat_pool snapshots (all PMUs in long_term_memory)

Usage:
    python3 rerun_ablation_suggest.py
    python3 rerun_ablation_suggest.py --max-questions 5   # smoke test
    python3 rerun_ablation_suggest.py --variant ablation_consolidation  # single variant
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

import tiktoken
from openai import OpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/ablations'
TABLE3_DATA = REPO_ROOT / 'benchmarks/personamem'
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
QUESTIONS_CSV = DATASETS / "questions_32k.csv"
SHARED_CONTEXTS_JSONL = DATASETS / "shared_contexts_32k.jsonl"

SNAPSHOTS_ABL = ROOT / "memory_snapshots"
SNAPSHOTS_T3 = TABLE3_DATA / "memory_snapshots"
OUTPUT_BASE = ROOT / "eval_results" / "deepseek" / "suggest_rerun"

# ===== LLM API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

MAX_CONTEXT_TOKENS = 32768
MEMORY_TOKEN_BUDGET = 2048

TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

# ===== Variant definitions =====
VARIANTS = {
    "ablation_consolidation": {
        "label": "w/o Consolidation",
        "snapshot_dir": SNAPSHOTS_ABL / "ablation_consolidation",
        "no_ltm": False,
        "no_evidence": False,
        "semantic_sort": False,
    },
    "ablation_branching": {
        "label": "w/o Branching",
        "snapshot_dir": SNAPSHOTS_ABL / "ablation_branching",
        "no_ltm": False,
        "no_evidence": False,
        "semantic_sort": False,
    },
    "ablation_no_evidence": {
        "label": "w/o Evidence",
        "snapshot_dir": SNAPSHOTS_T3 / "baseline",
        "no_ltm": False,
        "no_evidence": True,
        "semantic_sort": False,
    },
    "ablation_semantic": {
        "label": "Semantic-Only Retrieval",
        "snapshot_dir": SNAPSHOTS_T3 / "baseline",
        "no_ltm": False,
        "no_evidence": False,
        "semantic_sort": True,  # Sort by similarity to question, not stability*quality
    },
    "ablation_flat_pool": {
        "label": "Flat PPMU Pool",
        "snapshot_dir": SNAPSHOTS_ABL / "ablation_flat_pool",
        "no_ltm": False,
        "no_evidence": False,
        "semantic_sort": False,
    },
}

# ===== JSONL Index =====

def build_jsonl_index(jsonl_path: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index

def load_context_by_id(jsonl_path: Path, offset: int) -> list[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))

# ===== Semantic similarity for semantic_retrieval variant =====

def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())

def semantic_similarity(a: str, b: str) -> float:
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

# ===== Memory formatting (same as phase3_ablation.py v1) =====

def format_memory_block(
    memory_snapshot: dict[str, Any],
    max_tokens: int,
    no_ltm: bool = False,
    include_evidence: bool = True,
    semantic_sort: bool = False,
    question_text: str = "",
) -> str:
    """Format TPPM memory into a compact text block.

    Args:
        memory_snapshot: Phase 2 output dict with memory stores.
        max_tokens: Token budget for the memory block.
        no_ltm: If True, use working_memory + short_term_memory only.
        include_evidence: If True, include truncated evidence/context snippets.
        semantic_sort: If True, sort by semantic similarity to question (not stability*quality).
        question_text: The question text for semantic sorting.
    """
    if no_ltm:
        all_memories = (
            memory_snapshot.get("working_memory", []) +
            memory_snapshot.get("short_term_memory", [])
        )
        header = "[TPPM Memory — current session profile (no long-term memory)]\n"
    else:
        all_memories = memory_snapshot.get("long_term_memory", [])
        header = "[TPPM Memory — structured user profile]\n"

    if not all_memories:
        return ""

    # Score PMUs
    scored: list[tuple[float, dict]] = []
    for pmu in all_memories:
        if semantic_sort and question_text:
            # Semantic retrieval: score by similarity to question
            value = pmu.get("canonical_value", "") or pmu.get("value", "")
            if not value:
                branches = pmu.get("branches", [])
                value = branches[0].get("value", "") if branches else ""
            attr = pmu.get("attribute", "")
            pmu_text = f"{attr}: {value}"
            score = semantic_similarity(question_text, pmu_text)
        else:
            # Default: stability * quality
            stability = float(pmu.get("stability_score", 0))
            quality = float(pmu.get("quality_score", 0))
            score = stability * quality
        scored.append((score, pmu))
    scored.sort(key=lambda x: x[0], reverse=True)

    header_tokens = len(TOKENIZER.encode(header))
    budget = max_tokens - header_tokens

    entries: list[str] = []
    for _, pmu in scored:
        attribute = pmu.get("attribute", "?")
        canonical_value = pmu.get("canonical_value", "")
        if canonical_value and str(canonical_value).strip():
            value = str(canonical_value).strip()
        else:
            branches = pmu.get("branches", [])
            if branches:
                value = str(branches[0].get("value", pmu.get("value", "?"))).strip()
            else:
                value = str(pmu.get("value", "?")).strip()

        profile_type = pmu.get("profile_type", "general")
        stability = float(pmu.get("stability_score", 0))
        quality = float(pmu.get("quality_score", 0))
        scene = pmu.get("scene", "general")

        evidence_snippet = ""
        if include_evidence:
            ctx = pmu.get("context", "")
            if ctx and str(ctx).strip():
                ctx_str = str(ctx).strip()
                if len(ctx_str) > 120:
                    ctx_str = ctx_str[:120] + "..."
                evidence_snippet = f" ; evidence: \"{ctx_str}\""

        entry = (
            f"- {attribute}: {value} "
            f"(type={profile_type}, stability={stability:.2f}, quality={quality:.2f}, "
            f"scene={scene}{evidence_snippet})"
        )
        entry_tokens = len(TOKENIZER.encode(entry))
        if budget - entry_tokens < 0:
            break
        entries.append(entry)
        budget -= entry_tokens

    if not entries:
        return ""

    return header + "\n".join(entries) + "\n"

def build_context_window(
    conversation: list[dict],
    end_index: int,
    memory_snapshot: dict[str, Any] | None,
    question: str,
    all_options: str,
    no_ltm: bool = False,
    include_evidence: bool = True,
    semantic_sort: bool = False,
) -> list[dict]:
    instructions = (
        "Find the most appropriate model response and give your final answer "
        "(a), (b), (c), or (d) after the special token <final_answer>."
    )
    conv = conversation[:end_index]
    question_block = f"{question}\n\n{instructions}\n\n{all_options}"
    question_tokens = len(TOKENIZER.encode(question_block))

    conv_text = _messages_to_text(conv)
    conv_tokens = len(TOKENIZER.encode(conv_text))

    available = MAX_CONTEXT_TOKENS - question_tokens - conv_tokens
    memory_block = ""
    if memory_snapshot is not None and available > 200:
        memory_budget = min(MEMORY_TOKEN_BUDGET, available - 100)
        memory_block = format_memory_block(
            memory_snapshot, memory_budget,
            no_ltm=no_ltm, include_evidence=include_evidence,
            semantic_sort=semantic_sort, question_text=question,
        )

    memory_tokens = len(TOKENIZER.encode(memory_block)) if memory_block else 0
    total_used = conv_tokens + memory_tokens + question_tokens

    if total_used > MAX_CONTEXT_TOKENS:
        excess = total_used - MAX_CONTEXT_TOKENS + 200
        conv_text_chars = len(conv_text)
        trunc_ratio = max(0, (conv_text_chars - excess * 4) / max(1, conv_text_chars))
        conv_text = conv_text[:int(len(conv_text) * trunc_ratio)]
        conv_text += "\n[... conversation truncated to fit context window ...]"

    system_content = (
        "You are a helpful assistant answering questions about a user "
        "based on conversation history and profile memory."
    )

    user_content_parts = []
    if conv_text:
        user_content_parts.append(f"[Conversation History]\n{conv_text}")
    if memory_block:
        user_content_parts.append(memory_block)
    user_content_parts.append(question_block)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(user_content_parts)},
    ]

def _messages_to_text(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            if len(content) > 500:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                content = " ".join(sentences[:2])
                if len(sentences) > 2:
                    content += " [...]"
            lines.append(f"[Context] {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

# ===== Answer extraction =====

def extract_answer(predicted_answer: str, correct_answer: str) -> tuple[bool, str]:
    def _extract_only_options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r'\(([a-d])\)', text)
        if in_parens:
            return set(in_parens)
        return set(re.findall(r'\b([a-d])\b', text))

    correct = correct_answer.lower().strip("() ")
    full_response = predicted_answer
    predicted_answer = predicted_answer.strip()
    if "<final_answer>" in predicted_answer:
        predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
    if predicted_answer.endswith("</final_answer>"):
        predicted_answer = predicted_answer[:-len("</final_answer>")].strip()

    pred_options = _extract_only_options(predicted_answer)
    if pred_options == {correct}:
        return True, predicted_answer

    response_options = _extract_only_options(full_response)
    if response_options == {correct}:
        return True, predicted_answer

    return False, predicted_answer

# ===== Evaluation runner =====

def run_variant(
    variant: str,
    questions: list[dict],
    jsonl_index: dict[str, int],
    client: OpenAI,
    resume: bool = False,
) -> tuple[int, int]:
    """Run Suggest evaluation for a single variant. Returns (correct, total)."""
    cfg = VARIANTS[variant]
    label = cfg["label"]
    snapshot_dir = cfg["snapshot_dir"]
    no_ltm = cfg["no_ltm"]
    no_evidence = cfg["no_evidence"]
    semantic_sort = cfg["semantic_sort"]

    # Load memory snapshots
    memory_cache: dict[str, dict] = {}
    if snapshot_dir.exists():
        for fpath in snapshot_dir.glob("*.json"):
            with fpath.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
            ctx_hash = snapshot.get("context_hash", fpath.stem)
            memory_cache[ctx_hash] = snapshot
    else:
        print(f"  [WARN] No snapshots at {snapshot_dir}")
        return 0, 0

    # Output
    output_dir = OUTPUT_BASE / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    completed_ids: set[str] = set()
    total_correct = 0
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_ids.add(row["question_id"])
                if row.get("score", "").strip().lower() in ("true", "1", "yes"):
                    total_correct += 1
        print(f"  [RESUME] {len(completed_ids)} done, {total_correct} correct")

    total_questions = len(completed_ids)
    prev_sid = None
    prev_context = None

    write_mode = "a" if resume and output_path.exists() else "w"
    with open(output_path, write_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        if write_mode == "w":
            writer.writerow([
                "score", "persona_id", "question_id", "question_type", "topic",
                "correct_answer", "predicted_answer", "model_response",
                "condition", "context_length_in_tokens",
            ])

        pbar = tqdm(questions, desc=f"  {label}", leave=False)
        for row in pbar:
            if resume and row["question_id"] in completed_ids:
                continue

            total_questions += 1
            sid = row["shared_context_id"]
            end_index = int(row["end_index_in_shared_context"])

            if sid != prev_sid:
                if sid in jsonl_index:
                    prev_context = load_context_by_id(SHARED_CONTEXTS_JSONL, jsonl_index[sid])
                else:
                    prev_context = []
                prev_sid = sid

            memory = memory_cache.get(sid)
            question_text = row["user_question_or_message"]
            all_options = row["all_options"]
            correct_answer = row["correct_answer"]

            messages = build_context_window(
                prev_context, end_index, memory,
                question_text, all_options,
                no_ltm=no_ltm,
                include_evidence=not no_evidence,
                semantic_sort=semantic_sort,
            )

            try:
                response = client.chat.completions.create(
                    model=API_MODEL,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                model_response = response.choices[0].message.content or ""
            except Exception as e:
                tqdm.write(f"[ERROR] {label}: {e}")
                model_response = ""

            score, predicted = extract_answer(model_response, correct_answer)
            if score:
                total_correct += 1

            writer.writerow([
                score,
                row["persona_id"],
                row["question_id"],
                row["question_type"],
                row["topic"],
                correct_answer,
                predicted,
                model_response[:500],
                variant,
                row["context_length_in_tokens"],
            ])
            out_f.flush()

    accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
    print(f"  [{label}] Suggest: {total_correct}/{total_questions} = {accuracy:.2f}%")
    return total_correct, total_questions

# ===== Main =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run ablation variants on Suggest questions only")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions for smoke testing")
    parser.add_argument("--variant", type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="Run a single variant only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output CSVs")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # Build JSONL index
    print("[INFO] Building JSONL index...")
    jsonl_index = build_jsonl_index(SHARED_CONTEXTS_JSONL)
    print(f"[INFO] Indexed {len(jsonl_index)} shared contexts")

    # Filter questions: suggest_new_ideas only
    suggest_rows: list[dict] = []
    with open(QUESTIONS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["question_type"] == "suggest_new_ideas":
                suggest_rows.append(row)
    print(f"[INFO] Found {len(suggest_rows)} suggest_new_ideas questions")

    if args.max_questions:
        suggest_rows = suggest_rows[:args.max_questions]
        print(f"[INFO] Limited to {len(suggest_rows)} questions (smoke test)")

    variants_to_run = [args.variant] if args.variant else list(VARIANTS.keys())
    print(f"[INFO] Running {len(variants_to_run)} variant(s): {variants_to_run}")
    print()

    results: dict[str, tuple[int, int]] = {}
    for variant in variants_to_run:
        cfg = VARIANTS[variant]
        print(f"--- {variant} ({cfg['label']}) ---")
        print(f"  Snapshots: {cfg['snapshot_dir']}")
        print(f"  no_ltm={cfg['no_ltm']}, no_evidence={cfg['no_evidence']}, "
              f"semantic_sort={cfg['semantic_sort']}")
        correct, total = run_variant(
            variant, suggest_rows, jsonl_index, client, resume=args.resume)
        results[variant] = (correct, total)
        print()

    # Summary
    print("=" * 70)
    print(f"{'Variant':<30} {'Suggest':>12} {'%':>8}")
    print("-" * 70)
    for variant in variants_to_run:
        correct, total = results[variant]
        label = VARIANTS[variant]["label"]
        pct = correct / total * 100 if total > 0 else 0
        print(f"{label:<30} {correct:>3}/{total:<3}     {pct:>6.2f}%")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
