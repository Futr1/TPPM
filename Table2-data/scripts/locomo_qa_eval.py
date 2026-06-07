#!/usr/bin/env python3
"""Stage 2a: QA evaluation on LoCoMo with TPPM memory.

Loads the pre-extracted TPPM memory bank, builds hybrid context
(TPPM profile + recent sessions + early session summaries), generates
answers via vLLM Qwen3.5-9B, and evaluates with LoCoMo official F1.

Usage:
    python3 locomo_qa_eval.py                           # full 10 conversations
    python3 locomo_qa_eval.py --max-convs 1             # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow importing LoCoMo official evaluation
_LOCOMO_ROOT = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo")
if str(_LOCOMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCOMO_ROOT))

from task_eval.evaluation import eval_question_answering

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table2-data")
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
MEMORY_BANK_PATH = ROOT / "outputs" / "locomo_memory_bank.json"
MODEL_PATH = "/root/autodl-tmp/wangqihao/base_model/Qwen3.5-9B"
EVAL_DIR = ROOT / "outputs"

# ===== Hybrid context config =====
RECENT_SESSION_COUNT = 3
MAX_MODEL_LEN = 4096

QA_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Answer questions based on the conversation and profile information provided. "
    "Write a short answer in a few words. Do not write complete sentences. "
    "Answer with exact words from the conversations whenever possible. "
    "If the answer is not available in the provided context, "
    "say 'no information available'."
)


# ===== Data loading =====

def load_memory_bank(path: Path) -> dict[str, dict[str, Any]]:
    """Load TPPM memory bank indexed by conv_id."""
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
    """Extract sorted session info from a conversation."""
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


# ===== TPPM profile formatting =====

def format_tppm_profile(memory_entry: dict[str, Any], top_k: int = 10) -> str:
    """Format TPPM long-term and short-term memories as a profile text block."""
    long_term = memory_entry.get("long_term_memory", [])
    short_term = memory_entry.get("short_term_memory", [])

    all_memories = sorted(
        long_term + short_term,
        key=lambda m: (
            m.get("memory_level") != "long_term",
            -m.get("stability_score", 0),
        ),
    )

    if not all_memories:
        return "No profile information available."

    lines = ["[Speaker Profile — from long-term memory]"]
    for mem in all_memories[:top_k]:
        attr = mem.get("attribute", "")
        value = mem.get("value", "")
        ptype = mem.get("profile_type", "general")
        level = mem.get("memory_level", "?")
        stability = mem.get("stability_score", 0)
        session_count = mem.get("session_count", 0)
        if not value:
            continue
        lines.append(
            f"- {attr} ({ptype}): {value} "
            f"[stability={stability:.2f}, sessions={session_count}, level={level}]"
        )
    return "\n".join(lines) if len(lines) > 1 else "No profile information available."


# ===== Hybrid context builder =====

def build_hybrid_context(
    conv: dict[str, Any],
    memory_entry: dict[str, Any] | None,
) -> str:
    """Build hybrid context: TPPM profile + early summaries + recent full text.

    Strategy:
    - TPPM structured profile provides global memory across all sessions
    - Early sessions (1..N-RECENT): use pre-generated session_summary
    - Recent sessions (N-RECENT+1..N): full conversation text
    """
    conv_data = conv["conversation"]
    session_summaries = conv.get("session_summary", {})
    sessions = get_sorted_sessions(conv)
    total_sessions = len(sessions)

    parts: list[str] = []

    # 1. TPPM profile
    if memory_entry:
        profile_text = format_tppm_profile(memory_entry)
        parts.append(profile_text)
        parts.append("")

    # 2. Early session summaries
    summary_lines = ["[Earlier conversation summaries]"]
    recent_start = max(1, total_sessions - RECENT_SESSION_COUNT + 1)
    for num, key, turns in sessions:
        if num >= recent_start:
            break
        summary_key = f"session_{num}_summary"
        summary = session_summaries.get(summary_key, "")
        if summary:
            dt_key = f"session_{num}_date_time"
            dt = conv_data.get(dt_key, "")
            summary_lines.append(f"Session {num} ({dt}): {summary}")
    if len(summary_lines) > 1:
        parts.append("\n".join(summary_lines))
        parts.append("")

    # 3. Recent sessions full text
    for num, key, turns in sessions:
        if num < recent_start:
            continue
        dt_key = f"session_{num}_date_time"
        dt = conv_data.get(dt_key, "")
        parts.append(f"[Session {num} ({dt}) — full conversation]")
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
        parts.append("")

    return "\n".join(parts)


# ===== vLLM generation =====

def generate_qa_answers(
    conversations: list[dict[str, Any]],
    memory_bank: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate QA answers for all conversations using vLLM with TPPM context."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

    # Build all prompts
    all_prompts: list[str] = []
    all_qa_indices: list[tuple[int, int]] = []  # (conv_idx, qa_idx)

    for conv_idx, conv in enumerate(conversations):
        cid = conv.get("sample_id", "")
        memory_entry = memory_bank.get(cid)

        # Pre-build context once per conversation
        base_context = build_hybrid_context(conv, memory_entry)

        for qa_idx, qa_item in enumerate(conv["qa"]):
            question = qa_item["question"]
            prompt_text = (
                f"{base_context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            all_prompts.append(text)
            all_qa_indices.append((conv_idx, qa_idx))

    print(f"[INFO] Generating answers for {len(all_prompts)} QA pairs...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Collect answers back into conversation structure
    results: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        conv_idx, qa_idx = all_qa_indices[i]
        generated = output.outputs[0].text.strip()
        conv = conversations[conv_idx]
        qa_item = conv["qa"][qa_idx]

        while len(results) <= conv_idx:
            results.append({
                "sample_id": conversations[len(results)]["sample_id"],
                "qa": [],
            })

        results[conv_idx]["qa"].append({
            "question": qa_item["question"],
            "answer": qa_item["answer"],
            "category": qa_item["category"],
            "evidence": qa_item.get("evidence", []),
            "tppm_prediction": generated,
        })

    return results


# ===== Evaluation =====

def evaluate_and_save(
    results: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, float]:
    """Compute per-category and overall F1 using LoCoMo official evaluator."""
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
        try:
            scores, _, _ = eval_question_answering(qas, eval_key="tppm_prediction")
        except Exception as exc:
            print(f"[WARN] eval_question_answering failed for {conv_result['sample_id']}: {exc}")
            continue
        for i, qa in enumerate(qas):
            cat = qa["category"]
            if i < len(scores):
                all_f1s[cat].append(scores[i])

    summary: dict[str, float] = {}
    all_scores_flat: list[float] = []
    for cat, name in category_names.items():
        scores = all_f1s[cat]
        avg = round(float(np.mean(scores)) * 100, 1) if scores else 0.0
        summary[name] = avg
        all_scores_flat.extend(scores)

    summary["overall"] = (
        round(float(np.mean(all_scores_flat)) * 100, 1)
        if all_scores_flat else 0.0
    )

    payload = {
        "metadata": {
            "method": "tppm_memory",
            "model": MODEL_PATH,
            "context_strategy": "hybrid",
            "recent_sessions": RECENT_SESSION_COUNT,
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
        description="QA evaluation with TPPM on LoCoMo.")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--memory-bank", type=Path, default=MEMORY_BANK_PATH)
    parser.add_argument("--output", type=Path,
                        default=EVAL_DIR / "locomo_qa_results.json")
    parser.add_argument("--max-convs", type=int, default=None)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]
    print(f"[INFO] Loaded {len(conversations)} conversations")

    memory_bank = load_memory_bank(args.memory_bank)
    print(f"[INFO] Loaded TPPM memory bank: {len(memory_bank)} conversations indexed")

    results = generate_qa_answers(conversations, memory_bank)
    summary = evaluate_and_save(results, args.output)

    print(f"\n[SAVED] {args.output}")
    print(f"\n{'='*50}")
    print(f"LoCoMo QA — TPPM (n={sum(len(r['qa']) for r in results)})")
    for name, score in summary.items():
        print(f"  {name}: {score:.1f}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
