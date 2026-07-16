#!/usr/bin/env python3
"""Phase 3 Ablation v2: Evaluate QA accuracy for fine-grained & hierarchy ablation variants.

Context construction mirrors the full TPPM system:
  - Only current session's conversation (previous sessions via TPPM memory only)
  - TPPM memory: retrieve top-K PMUs from ALL 3 tiers using multi-factor scoring
  - Each ablation variant modifies only its target mechanism

Full TPPM context construction (from Mini-Agent begin_turn + augment_user_message):
  1. retrieve(query, scene, top_k) — scores PMUs across all 3 tiers using:
     Score = w1*Rel + w2*Stability + w3*Ctx + w4*Scene + w5*Quality
  2. Format retrieved PMUs and append to user message

Ablation modifications:
  - ablation_uniform_decay: Same retrieval, but snapshots have uniform decay rates
  - ablation_semantic_retrieval: Retrieval scoring = Rel only (w1=1, rest=0)
  - ablation_flat_pool: Snapshots have all PMUs in long_term (no tier distinction)
  - ablation_two_level: Snapshots have transient+long-term (no working tier)

Usage:
    python3 phase3_ablation_v2.py --condition ablation_uniform_decay
    python3 phase3_ablation_v2.py --condition ablation_flat_pool --max-questions 10
    python3 phase3_ablation_v2.py --all
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
from typing import Any

import tiktoken
from openai import OpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Ablation")
TABLE3_DATA = Path("/root/autodl-tmp/wangqihao/Table3-data")
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
QUESTIONS_CSV = DATASETS / "questions_32k.csv"
SHARED_CONTEXTS_JSONL = DATASETS / "shared_contexts_32k.jsonl"

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
RETRIEVAL_TOP_K = 5  # Same as TPMMemoryManager default

# ===== Full TPPM retrieval weights (from TPMConfig defaults) =====
FULL_RETRIEVE_WEIGHTS = (0.35, 0.2, 0.15, 0.2, 0.1)  # Rel, Stability, Ctx, Scene, Quality

# ===== Tokenizer =====
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

# ===== Condition definitions =====
CONDITIONS = {
    "full_tppm": {
        "source": "t3",
        "config_id": "baseline",
        "label": "Full TPPM",
        "retrieve_weights": FULL_RETRIEVE_WEIGHTS,
    },
    "ablation_uniform_decay": {
        "source": "abl",
        "config_id": "ablation_uniform_decay",
        "label": "w/o Type-Conditioned Decay",
        "retrieve_weights": FULL_RETRIEVE_WEIGHTS,  # Full retrieval scoring
    },
    "ablation_semantic_retrieval": {
        "source": "t3",
        "config_id": "baseline",
        "label": "Semantic-Only Retrieval",
        "retrieve_weights": (1.0, 0.0, 0.0, 0.0, 0.0),  # Only Rel term
    },
    "ablation_flat_pool": {
        "source": "abl",
        "config_id": "ablation_flat_pool",
        "label": "Flat PPMU Pool",
        "retrieve_weights": FULL_RETRIEVE_WEIGHTS,  # Full retrieval scoring
    },
    "ablation_two_level": {
        "source": "abl",
        "config_id": "ablation_two_level",
        "label": "Two-Level Memory",
        "retrieve_weights": FULL_RETRIEVE_WEIGHTS,  # Full retrieval scoring
    },
}


# ===== Helper functions (same as Mini-Agent) =====

def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _similarity(left: str, right: str) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _get_pmu_value(pmu: dict) -> str:
    """Extract the best value from a PMU dict."""
    canonical_value = pmu.get("canonical_value", "")
    if canonical_value and str(canonical_value).strip():
        return str(canonical_value).strip()
    branches = pmu.get("branches", [])
    if branches:
        return str(branches[0].get("value", pmu.get("value", "?"))).strip()
    return str(pmu.get("value", "?")).strip()


# ===== Retrieval scoring (mirrors _retrieve_score from memory.py) =====

def retrieve_score(
    query_norm: str,
    pmu: dict,
    scene: str,
    weights: tuple[float, ...] = FULL_RETRIEVE_WEIGHTS,
) -> float:
    """Compute retrieval score for a PMU — mirrors TemporalProfileMemory._retrieve_score.

    Full: Score = w1*Rel(qt,mi) + w2*stability + w3*Ctx + w4*Scene + w5*Quality
    Semantic-only: Score = Rel(qt,mi) only (weights = (1,0,0,0,0))
    """
    value = _get_pmu_value(pmu)
    branches = pmu.get("branches", [])

    # Find branch matching scene
    branch_value = value
    branch_context = pmu.get("context", "")
    branch_scene = pmu.get("scene", "general")
    branch_quality = float(pmu.get("quality_score", 0))
    for b in branches:
        if b.get("scene") == scene:
            bv = b.get("value", "")
            if bv:
                branch_value = bv
            bc = b.get("context", "")
            if bc:
                branch_context = bc
            branch_scene = b.get("scene", branch_scene)
            bqs = b.get("quality_score")
            if bqs is not None:
                branch_quality = float(bqs)
            break

    # Rel(qt, mi) — semantic relevance
    rel = max(_similarity(query_norm, branch_value), _similarity(query_norm, value))

    # Scene score
    if branch_scene == scene:
        scene_score = 1.0
    elif branch_scene == "general" or scene == "general":
        scene_score = 0.7
    else:
        scene_score = 0.4

    # Context score
    ctx_score = max(
        _similarity(query_norm, branch_context),
        _similarity(query_norm, pmu.get("context", "")),
        1.0 if pmu.get("attribute", "") in query_norm or pmu.get("profile_type", "") in query_norm else 0.0,
    )

    stability = float(pmu.get("stability_score", 0))
    quality = max(float(pmu.get("quality_score", 0)), branch_quality)

    w1, w2, w3, w4, w5 = weights
    return w1 * rel + w2 * stability + w3 * ctx_score + w4 * scene_score + w5 * quality


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


# ===== Session boundary detection =====

def find_current_session_start(messages: list[dict], end_index: int) -> int:
    """Find the start index of the current session.

    Session boundaries are marked by role='system' messages.
    The current session is the last session before end_index.
    """
    last_session_start = 0
    for i in range(min(end_index, len(messages))):
        if messages[i].get("role") == "system":
            last_session_start = i
    return last_session_start


# ===== Memory retrieval and formatting (mirrors TPMMemoryManager) =====

def retrieve_memories(
    memory_snapshot: dict[str, Any],
    query: str,
    scene: str = "general",
    top_k: int = RETRIEVAL_TOP_K,
    retrieve_weights: tuple[float, ...] = FULL_RETRIEVE_WEIGHTS,
) -> list[tuple[float, dict, str]]:
    """Retrieve top-K PMUs from all 3 tiers using multi-factor scoring.

    Mirrors TemporalProfileMemory.retrieve() logic.
    Returns list of (score, pmu, tier_name).
    """
    query_norm = _normalize(query)
    scored: list[tuple[float, dict, str]] = []

    for tier_name in ["working_memory", "short_term_memory", "long_term_memory"]:
        for pmu in memory_snapshot.get(tier_name, []):
            score = retrieve_score(query_norm, pmu, scene, weights=retrieve_weights)
            level = pmu.get("memory_level", tier_name.replace("_memory", ""))
            scored.append((score, pmu, level))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def format_memory_block(
    memories: list[tuple[float, dict, str]],
    scene: str = "general",
) -> str:
    """Format retrieved PMUs into memory block.

    Mirrors TPMMemoryManager.augment_user_message() format.
    """
    if not memories:
        return ""

    lines: list[str] = []
    for _, pmu, level in memories:
        attribute = pmu.get("attribute", "?")
        value = _get_pmu_value(pmu)

        # Get scene-specific branch
        profile_type = pmu.get("profile_type", "general")
        stability = float(pmu.get("stability_score", 0))

        # Evidence snippet (mirrors evidence_for_unit)
        evidence_snippet = ""
        ctx = pmu.get("context", "")
        if ctx and str(ctx).strip():
            ctx_str = str(ctx).strip()
            if len(ctx_str) > 80:
                ctx_str = ctx_str[:80] + "..."
            evidence_snippet = f", evidence={ctx_str}"

        # Format matching TPMMemoryManager.augment_user_message
        lines.append(
            f"- {attribute}: {value} "
            f"(type={profile_type}, scene={scene}, "
            f"stability={stability:.2f}, level={level}{evidence_snippet})"
        )

    return "[Temporal Profile Memory]\n" + "\n".join(lines)


# ===== Context builder (current session only, mirrors full TPPM) =====

def build_context_window(
    conversation: list[dict],
    end_index: int,
    memory_snapshot: dict[str, Any] | None,
    question: str,
    all_options: str,
    retrieve_weights: tuple[float, ...] = FULL_RETRIEVE_WEIGHTS,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """Build context window mirroring full TPPM's logic.

    1. Current session's conversation only (previous sessions via TPPM memory)
    2. Retrieve top-K PMUs from all 3 tiers
    3. Format and append to context
    """
    instructions = (
        "Find the most appropriate model response and give your final answer "
        "(a), (b), (c), or (d) after the special token <final_answer>."
    )

    # Only current session's conversation
    session_start = find_current_session_start(conversation, end_index)
    conv = conversation[session_start:end_index]
    conv_text = _messages_to_text(conv)

    question_block = f"{question}\n\n{instructions}\n\n{all_options}"

    # Retrieve and format memory (mirrors begin_turn + augment_user_message)
    memory_block = ""
    if memory_snapshot is not None:
        # Use question as the retrieval query (mirrors retrieve in begin_turn)
        memories = retrieve_memories(
            memory_snapshot, question, scene="general",
            top_k=top_k, retrieve_weights=retrieve_weights,
        )
        memory_block = format_memory_block(memories, scene="general")

    # Truncate conversation if total exceeds 32K
    conv_tokens = len(TOKENIZER.encode(conv_text))
    question_tokens = len(TOKENIZER.encode(question_block))
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
        "based on the current conversation and structured profile memory."
    )

    user_content_parts = []
    if conv_text:
        user_content_parts.append(f"[Current Session]\n{conv_text}")
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
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[Path, int, int]:
    cond = CONDITIONS[condition]
    config_id = cond["config_id"]
    retrieve_weights = cond["retrieve_weights"]
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

    # Output — use a different subdir to avoid overwriting old results
    output_dir = EVAL_DIR / "deepseek_v2" / condition
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
                    retrieve_weights=retrieve_weights,
                    top_k=top_k,
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
        description="Phase 3 Ablation v2: QA evaluation (current session + full TPPM retrieval)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=list(CONDITIONS.keys()),
                        help="Condition to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Run all new conditions")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions for smoke testing")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K,
                        help=f"Number of PMUs to retrieve (default: {RETRIEVAL_TOP_K})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV")
    args = parser.parse_args()

    if not args.condition and not args.all:
        print("[ERROR] Specify --condition <name> or --all")
        return 1

    conditions = list(CONDITIONS.keys()) if args.all else [args.condition]

    for condition in conditions:
        cond = CONDITIONS[condition]
        w = cond["retrieve_weights"]
        is_full = (w == FULL_RETRIEVE_WEIGHTS)
        w_desc = "full" if is_full else f"Rel-only ({w[0]},{w[1]},{w[2]},{w[3]},{w[4]})"
        print(f"\n{'='*60}")
        print(f"[INFO] Condition: {condition}")
        print(f"  Label:          {cond['label']}")
        print(f"  Snapshot:       {'Ablation' if cond['source'] == 'abl' else 'Table3-data'}/{cond['config_id']}")
        print(f"  Retrieval:      {w_desc}")
        print(f"  Top-K:          {args.top_k}")
        print(f"  Context:        Current session only")
        print(f"  Memory tiers:   All 3 (working + short-term + long-term)")
        print(f"  Max questions:  {args.max_questions or 'all (589)'}")
        print(f"{'='*60}")

        run_evaluation(
            condition,
            max_questions=args.max_questions,
            resume=args.resume,
            top_k=args.top_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
