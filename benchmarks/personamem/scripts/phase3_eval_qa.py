#!/usr/bin/env python3
"""Phase 3: Evaluate QA accuracy for each TPPM config using vLLM + Qwen3.5-9B.

Builds a 32K-token context window: conversation history + TPPM memory + question.
Queries vLLM via OpenAI-compatible HTTP API.

Usage:
    python3 phase3_eval_qa.py --config-id baseline
    python3 phase3_eval_qa.py --config-id baseline --max-questions 10   # smoke test
    python3 phase3_eval_qa.py --config-id baseline --no-tppm             # ablation

vLLM server must be running:
    vllm serve Qwen/Qwen3.5-9B --tensor-parallel-size 2 --max-model-len 32768
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

import tiktoken
from openai import OpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/personamem'
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
QUESTIONS_CSV = DATASETS / "questions_32k.csv"
SHARED_CONTEXTS_JSONL = DATASETS / "shared_contexts_32k.jsonl"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
EVAL_DIR = ROOT / "eval_results"

# ===== vLLM Config =====
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3.5-9B"
MAX_CONTEXT_TOKENS = 32768
MEMORY_TOKEN_BUDGET = 2048  # Max tokens for TPPM memory block

# ===== Tokenizer =====
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")  # Approximate token counter

# ===== JSONL Index =====

def build_jsonl_index(jsonl_path: Path) -> dict[str, int]:
    """Build file-offset index for JSONL: {key: byte_offset}."""
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
    """Load a single shared context from JSONL by byte offset."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))

# ===== Context window builder =====

def format_memory_block(memory_snapshot: dict[str, Any], max_tokens: int) -> str:
    """Format TPPM long-term memory into a compact text block.

    Sorts PMUs by stability_score * quality_score descending,
    then takes top entries that fit within max_tokens budget.

    Args:
        memory_snapshot: Phase 2 output dict with long_term_memory list.
        max_tokens: Token budget for the memory block.

    Returns:
        Formatted memory string.
    """
    long_term = memory_snapshot.get("long_term_memory", [])
    if not long_term:
        return ""

    # Score and sort PMUs
    scored: list[tuple[float, dict]] = []
    for pmu in long_term:
        stability = float(pmu.get("stability_score", 0))
        quality = float(pmu.get("quality_score", 0))
        score = stability * quality
        scored.append((score, pmu))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build entries, tracking token budget
    header = "[TPPM Memory — structured user profile]\n"
    header_tokens = len(TOKENIZER.encode(header))
    budget = max_tokens - header_tokens

    entries: list[str] = []
    for _, pmu in scored:
        attribute = pmu.get("attribute", "?")
        # Use canonical_value if available, otherwise value
        canonical_value = pmu.get("canonical_value", "")
        if canonical_value and str(canonical_value).strip():
            value = str(canonical_value).strip()
        else:
            # Try branches for the value
            branches = pmu.get("branches", [])
            if branches:
                value = str(branches[0].get("value", pmu.get("value", "?"))).strip()
            else:
                value = str(pmu.get("value", "?")).strip()

        profile_type = pmu.get("profile_type", "general")
        stability = float(pmu.get("stability_score", 0))
        quality = float(pmu.get("quality_score", 0))
        scene = pmu.get("scene", "general")

        entry = (
            f"- {attribute}: {value} "
            f"(type={profile_type}, stability={stability:.2f}, quality={quality:.2f}, scene={scene})"
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
) -> list[dict]:
    """Build the 32K-token context window for vLLM.

    Order: conversation (truncated) → memory block → question + options.

    Args:
        conversation: Full shared context message list.
        end_index: Cutoff index from PersonaMem question (exclusive).
        memory_snapshot: Phase 2 memory snapshot, or None for no-TPPM ablation.
        question: The QA question text.
        all_options: Formatted options string "(a) ... (b) ... (c) ... (d) ..."

    Returns:
        List of {"role": ..., "content": ...} dicts for vLLM chat API.
    """
    instructions = (
        "Find the most appropriate model response and give your final answer "
        "(a), (b), (c), or (d) after the special token <final_answer>."
    )

    # Truncate conversation to end_index
    conv = conversation[:end_index]

    # Count tokens for the fixed parts
    question_block = f"{question}\n\n{instructions}\n\n{all_options}"
    question_tokens = len(TOKENIZER.encode(question_block))

    # Build conversation as text, count tokens
    conv_text = _messages_to_text(conv)
    conv_tokens = len(TOKENIZER.encode(conv_text))

    # Calculate memory budget
    available = MAX_CONTEXT_TOKENS - question_tokens - conv_tokens

    memory_block = ""
    if memory_snapshot is not None and available > 200:
        memory_budget = min(MEMORY_TOKEN_BUDGET, available - 100)
        memory_block = format_memory_block(memory_snapshot, memory_budget)

    # Re-count after memory is added, and truncate conversation if needed
    memory_tokens = len(TOKENIZER.encode(memory_block)) if memory_block else 0
    total_used = conv_tokens + memory_tokens + question_tokens

    if total_used > MAX_CONTEXT_TOKENS:
        # Truncate conversation to fit
        excess = total_used - MAX_CONTEXT_TOKENS + 200  # extra margin
        # Crude truncation: drop roughly excess tokens from conversation
        conv_text_chars = len(conv_text)
        trunc_ratio = max(0, (conv_text_chars - excess * 4) / max(1, conv_text_chars))
        conv_text = conv_text[:int(len(conv_text) * trunc_ratio)]
        conv_text += "\n[... conversation truncated to fit context window ...]"

    # Assemble final messages
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

    user_content = "\n\n".join(user_content_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

def _messages_to_text(messages: list[dict]) -> str:
    """Convert message list to compact text format."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            # Truncate persona descriptions — keep only first 2 sentences
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
    """Extract predicted option and compare against correct answer.

    Ported from PersonaMem inference_standalone_openai.py.
    """
    def _extract_only_options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r'\(([a-d])\)', text)
        if in_parens:
            return set(in_parens)
        else:
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

    # Fallback: search full response
    response_options = _extract_only_options(full_response)
    if response_options == {correct}:
        return True, predicted_answer

    return False, predicted_answer

# ===== Evaluation runner =====

def run_evaluation(
    config_id: str,
    no_tppm: bool = False,
    max_questions: int | None = None,
    vllm_url: str = VLLM_BASE_URL,
) -> tuple[Path, int, int]:
    """Run QA evaluation for a single config.

    Args:
        config_id: Config ID from Phase 2 (e.g. 'baseline', 'write_0.56').
        no_tppm: If True, skip TPPM memory (ablation baseline).
        max_questions: Limit questions for smoke testing.
        vllm_url: vLLM server URL.

    Returns:
        (output_path, num_correct, num_total)
    """
    client = OpenAI(base_url=vllm_url, api_key="not-needed")

    # Build JSONL index for shared contexts
    jsonl_index = build_jsonl_index(SHARED_CONTEXTS_JSONL)

    # Load memory snapshots for this config (unless no_tppm)
    memory_cache: dict[str, dict] = {}
    if not no_tppm:
        snapshot_dir = SNAPSHOTS_DIR / config_id
        if snapshot_dir.exists():
            for fpath in snapshot_dir.glob("*.json"):
                with fpath.open("r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                ctx_hash = snapshot.get("context_hash", fpath.stem)
                memory_cache[ctx_hash] = snapshot
        else:
            print(f"[WARN] No snapshots found for config '{config_id}' at {snapshot_dir}")

    # Output path
    output_dir = EVAL_DIR / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    # Read questions
    total_correct = 0
    total_questions = 0
    prev_sid = None
    prev_context = None

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "score", "persona_id", "question_id", "question_type", "topic",
            "correct_answer", "predicted_answer", "model_response",
            "config_id", "context_length_in_tokens",
        ])

        with open(QUESTIONS_CSV, "r", newline="", encoding="utf-8") as csv_f:
            reader = csv.DictReader(csv_f)
            for row in tqdm(reader, desc=f"Evaluating {config_id}",
                            total=max_questions or 589):
                if max_questions and total_questions >= max_questions:
                    break

                total_questions += 1

                sid = row["shared_context_id"]
                end_index = int(row["end_index_in_shared_context"])

                # Load shared context (cache by sid)
                if sid != prev_sid:
                    if sid in jsonl_index:
                        prev_context = load_context_by_id(
                            SHARED_CONTEXTS_JSONL, jsonl_index[sid])
                    else:
                        prev_context = []
                    prev_sid = sid
                context = prev_context

                # Get memory snapshot for this context
                memory = memory_cache.get(sid) if not no_tppm else None

                # Build context window
                question_text = row["user_question_or_message"]
                all_options = row["all_options"]
                correct_answer = row["correct_answer"]

                messages = build_context_window(
                    context, end_index, memory,
                    question_text, all_options,
                )

                try:
                    response = client.chat.completions.create(
                        model=VLLM_MODEL,
                        messages=messages,
                        max_tokens=256,
                        temperature=0,
                    )
                    model_response = response.choices[0].message.content or ""
                except Exception as e:
                    tqdm.write(f"[ERROR] LLM call failed: {e}")
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
                    model_response[:500],  # Truncate long responses
                    config_id,
                    row["context_length_in_tokens"],
                ])

    accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
    print(f"[DONE] {config_id}: {total_correct}/{total_questions} = {accuracy:.2f}%")
    return output_path, total_correct, total_questions

# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: Evaluate QA accuracy with TPPM memory via vLLM")
    parser.add_argument("--config-id", type=str, required=True,
                        help="Config ID from Phase 2 (e.g. 'baseline')")
    parser.add_argument("--no-tppm", action="store_true",
                        help="Run without TPPM memory (ablation baseline)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions for smoke testing")
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL,
                        help="vLLM server URL")
    args = parser.parse_args()

    output_path, correct, total = run_evaluation(
        config_id=args.config_id,
        no_tppm=args.no_tppm,
        max_questions=args.max_questions,
        vllm_url=args.vllm_url,
    )

    print(f"\n[DONE] Results saved to {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
