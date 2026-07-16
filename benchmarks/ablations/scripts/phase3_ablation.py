#!/usr/bin/env python3
"""Phase 3 benchmarks/ablations: Evaluate QA accuracy for each ablation condition via DeepSeek API.

Supports ablation-specific flags:
  --no-ltm       → w/o Long-term Retrieval (exclude long_term_memory)
  --no-evidence  → w/o Evidence Collection (exclude context/evidence text)

The baseline (Full TPPM) reuses benchmarks/personamem Phase 3 results directly.

Usage:
    # Full TPPM (reuses existing baseline)
    python3 phase3_ablation.py --condition full_tppm

    # benchmarks/ablations 2: w/o Consolidation
    python3 phase3_ablation.py --condition ablation_consolidation

    # benchmarks/ablations 3: w/o Scene Branching
    python3 phase3_ablation.py --condition ablation_branching

    # benchmarks/ablations 4: w/o Temporal Decay
    python3 phase3_ablation.py --condition ablation_decay

    # benchmarks/ablations 5: w/o Evidence Collection (uses baseline snapshots + --no-evidence)
    python3 phase3_ablation.py --condition full_tppm --no-evidence

    # benchmarks/ablations 6: w/o Long-term Retrieval (uses baseline snapshots + --no-ltm)
    python3 phase3_ablation.py --condition full_tppm --no-ltm

    # Smoke test
    python3 phase3_ablation.py --condition ablation_branching --max-questions 10
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import re
import sys
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

# Snapshot sources: benchmarks/ablations's own snapshots + benchmarks/personamem baseline
SNAPSHOTS_ABLATION = ROOT / "memory_snapshots"
SNAPSHOTS_TABLE3 = TABLE3_DATA / "memory_snapshots"
EVAL_DIR = ROOT / "eval_results"

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

# ===== Tokenizer =====
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

# ===== Condition definitions =====
# Maps condition name → (snapshot_source_dir, snapshot_config_id)
# source='abl' → benchmarks/ablations/memory_snapshots/<config_id>
# source='t3'  → benchmarks/personamem/memory_snapshots/<config_id>
CONDITIONS = {
    "full_tppm": {
        "source": "t3",
        "config_id": "baseline",
        "label": "Full TPPM",
        "no_ltm": False,
        "no_evidence": False,
    },
    "ablation_consolidation": {
        "source": "abl",
        "config_id": "ablation_consolidation",
        "label": "w/o Consolidation",
        "no_ltm": False,
        "no_evidence": False,
    },
    "ablation_branching": {
        "source": "abl",
        "config_id": "ablation_branching",
        "label": "w/o Scene Branching",
        "no_ltm": False,
        "no_evidence": False,
    },
    "ablation_decay": {
        "source": "abl",
        "config_id": "ablation_decay",
        "label": "w/o Temporal Decay",
        "no_ltm": False,
        "no_evidence": False,
    },
    "ablation_no_evidence": {
        "source": "t3",
        "config_id": "baseline",
        "label": "w/o Evidence Collection",
        "no_ltm": False,
        "no_evidence": True,
    },
    "ablation_no_ltm": {
        "source": "t3",
        "config_id": "baseline",
        "label": "w/o Long-term Retrieval",
        "no_ltm": True,
        "no_evidence": False,
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


# ===== Context window builder =====

def format_memory_block(
    memory_snapshot: dict[str, Any],
    max_tokens: int,
    no_ltm: bool = False,
    include_evidence: bool = True,
) -> str:
    """Format TPPM memory into a compact text block for LLM context.

    Args:
        memory_snapshot: Phase 2 output dict with memory stores.
        max_tokens: Token budget for the memory block.
        no_ltm: If True, use working_memory + short_term_memory only.
        include_evidence: If True, include truncated evidence/context snippets.

    Returns:
        Formatted memory string.
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

    # Score and sort PMUs by stability_score * quality_score descending
    scored: list[tuple[float, dict]] = []
    for pmu in all_memories:
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

def run_evaluation(
    condition: str,
    max_questions: int | None = None,
    resume: bool = False,
) -> tuple[Path, int, int]:
    """Run QA evaluation for a single ablation condition.

    Returns:
        (output_path, num_correct, num_total)
    """
    cond = CONDITIONS[condition]
    config_id = cond["config_id"]
    no_ltm = cond["no_ltm"]
    no_evidence = cond["no_evidence"]
    label = cond["label"]

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # Build JSONL index
    jsonl_index = build_jsonl_index(SHARED_CONTEXTS_JSONL)

    # Load memory snapshots
    snapshot_dir = (SNAPSHOTS_ABLATION if cond["source"] == "abl"
                    else SNAPSHOTS_TABLE3) / config_id
    memory_cache: dict[str, dict] = {}
    if snapshot_dir.exists():
        for fpath in snapshot_dir.glob("*.json"):
            with fpath.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
            ctx_hash = snapshot.get("context_hash", fpath.stem)
            memory_cache[ctx_hash] = snapshot
    else:
        print(f"[WARN] No snapshots found at {snapshot_dir}")

    # Output
    output_dir = EVAL_DIR / "deepseek" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    # Resume support
    completed_ids: set[str] = set()
    total_correct = 0
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_ids.add(row["question_id"])
                if row.get("score", "").strip().lower() in ("true", "1", "yes"):
                    total_correct += 1
        print(f"[RESUME] {len(completed_ids)} already evaluated, {total_correct} correct")

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

        with open(QUESTIONS_CSV, "r", newline="", encoding="utf-8") as csv_f:
            reader = csv.DictReader(csv_f)
            for row in tqdm(reader, desc=f"Evaluating {condition}",
                            total=max_questions or 589):
                if max_questions and total_questions >= max_questions:
                    break
                if resume and row["question_id"] in completed_ids:
                    continue

                total_questions += 1
                sid = row["shared_context_id"]
                end_index = int(row["end_index_in_shared_context"])

                if sid != prev_sid:
                    if sid in jsonl_index:
                        prev_context = load_context_by_id(
                            SHARED_CONTEXTS_JSONL, jsonl_index[sid])
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
                    model_response[:500],
                    condition,
                    row["context_length_in_tokens"],
                ])

    accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
    print(f"[DONE] {condition} ({label}): {total_correct}/{total_questions} = {accuracy:.2f}%")
    return output_path, total_correct, total_questions


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 benchmarks/ablations: QA evaluation for ablation conditions")
    parser.add_argument("--condition", type=str, required=True,
                        choices=list(CONDITIONS.keys()),
                        help="Condition to evaluate")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions for smoke testing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV, skipping completed questions")
    # Per-call overrides (for flexibility, but CONDITIONS already defines defaults)
    parser.add_argument("--no-ltm", action="store_true",
                        help="Override: exclude long-term memory")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Override: exclude evidence context")
    args = parser.parse_args()

    # Apply CLI overrides if given
    if args.no_ltm:
        CONDITIONS[args.condition]["no_ltm"] = True
    if args.no_evidence:
        CONDITIONS[args.condition]["no_evidence"] = True

    cond = CONDITIONS[args.condition]
    print(f"[INFO] Condition: {args.condition}")
    print(f"  Label:       {cond['label']}")
    print(f"  Snapshot:    {'benchmarks/ablations' if cond['source'] == 'abl' else 'benchmarks/personamem'}/{cond['config_id']}")
    print(f"  No LTM:      {cond['no_ltm']}")
    print(f"  No Evidence: {cond['no_evidence']}")
    print(f"  Max questions: {args.max_questions or 'all (589)'}")

    output_path, n_correct, n_total = run_evaluation(
        args.condition,
        max_questions=args.max_questions,
        resume=args.resume,
    )

    print(f"\n[RESULTS] {cond['label']}: {n_correct}/{n_total} = "
          f"{n_correct/n_total*100:.2f}%")
    print(f"  Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
