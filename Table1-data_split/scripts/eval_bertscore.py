#!/usr/bin/env python3
"""BERTScore evaluation for Table 1 Layer 1 — supports multiple memory methods.

Methods:
    no_memory       — only the last user turn, no history
    long_context    — full conversation history concatenated
    summary_memory  — LLM-summarised history + last user turn

Output per method:
    outputs/eval/{method}_generations.json
    outputs/eval/{method}_bertscore.json

Usage:
    python3 eval_bertscore.py --method no_memory
    python3 eval_bertscore.py --method long_context --max-cases 30
    python3 eval_bertscore.py --method summary_memory --min-turns 3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/wangqihao/Table1-data_split")
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
MODEL_PATH = "/root/autodl-tmp/wangqihao/base_model/Qwen3.5-9B"
BERT_MODEL_PATH = ROOT / "eval_model" / "bert-base-chinese"
EVAL_DIR = ROOT / "outputs" / "eval"

DEFAULT_MEMORY_BANK = ROOT / "outputs" / "d101_tppm_memory_bank.json"

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

SUMMARY_PROMPT = (
    "请用2-3句话简洁概括以下对话的核心内容，"
    "包括用户的主要问题、情绪状态和已讨论的关键话题。"
    "只输出摘要文本。\n\n"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_d101(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_bank(path: Path) -> dict[int, list[dict]]:
    """Load D101 TPPM memory bank and index by case_idx.

    Returns:
        dict mapping case_idx (int) -> list of memory dicts.
        Returns empty dict if file does not exist.
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
    """Format TPPM memories as【画像背景】text block.

    Format matches teacher_distill.py for consistency:
        1. 压力来源: <value>;显著性=<phi>;简要依据=<evidence>
        2. 情绪状态: ...
        3. 应对方式: ...
    """
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


# ---------------------------------------------------------------------------
# Prompt construction per method
# ---------------------------------------------------------------------------

def build_messages_no_memory(case: dict) -> list[dict]:
    """Only the last user turn — no conversation history."""
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
    """Full conversation history as context."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *case["messages"],
    ]


def build_messages_summary_memory(
    case: dict,
    tokenizer,
    llm,
    sampling_params,
) -> list[dict]:
    """LLM summarises the history, then last user turn with summary as context."""
    msgs = case["messages"]
    if len(msgs) <= 2:
        # Too short to need summarisation — fall back to long_context
        return build_messages_long_context(case)

    # Separate last user turn from history
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise ValueError(f"No user message found in case {case['idx']}")

    history = msgs[:last_user_idx]
    last_user = msgs[last_user_idx]

    # Build summary prompt
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '咨询师'}: {m['content']}"
        for m in history
    )
    summary_messages = [
        {"role": "user", "content": SUMMARY_PROMPT + history_text},
    ]
    summary_text = tokenizer.apply_chat_template(
        summary_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    summary_output = llm.generate([summary_text], sampling_params)
    summary = summary_output[0].outputs[0].text.strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【对话历史摘要】\n{summary}\n\n【用户最新消息】\n{last_user['content']}"},
    ]


def build_messages_tppm_memory(
    case: dict,
    memory_index: dict[int, list[dict]],
) -> tuple[list[dict], str | None]:
    """Build messages with TPPM psychological profile as context.

    Args:
        case: D101 case dict with 'idx' and 'messages'.
        memory_index: dict mapping case_idx -> list of TPPM memory dicts.

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


# ---------------------------------------------------------------------------
# vLLM batch generation
# ---------------------------------------------------------------------------

def generate_responses(
    test_cases: list[dict],
    method: str,
    min_turns: int,
    method_kwargs: dict | None = None,
) -> list[dict]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if method_kwargs is None:
        method_kwargs = {}

    # Load TPPM memory bank if needed
    memory_index: dict[int, list[dict]] = {}
    fallback_reasons: dict[int, str] = {}
    if method == "tppm_memory":
        memory_bank_path = method_kwargs.get("memory_bank", DEFAULT_MEMORY_BANK)
        memory_index = load_memory_bank(memory_bank_path)
        print(f"[INFO] Loaded TPPM memory bank: {len(memory_index)} cases indexed")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(temperature=0, max_tokens=256)

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        language_model_only=True,
    )

    prompts = []
    valid_cases = []
    skipped = 0

    for case in test_cases:
        if len(case["messages"]) < min_turns:
            skipped += 1
            continue

        if method == "no_memory":
            msgs = build_messages_no_memory(case)
        elif method == "long_context":
            msgs = build_messages_long_context(case)
        elif method == "summary_memory":
            msgs = build_messages_summary_memory(case, tokenizer, llm, sampling_params)
        elif method == "tppm_memory":
            msgs, fallback = build_messages_tppm_memory(case, memory_index)
            if fallback:
                fallback_reasons[case["idx"]] = fallback
        else:
            raise ValueError(f"Unknown method: {method}")

        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
        valid_cases.append(case)

    if skipped:
        print(f"[INFO] Skipped {skipped} cases (min_turns={min_turns})")

    print(f"[INFO] Generating responses for {len(prompts)} cases (method={method})...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text.strip()
        idx = valid_cases[i]["idx"]
        entry = {
            "idx": idx,
            "golden": valid_cases[i]["golden"]["content"],
            "generated": generated,
        }
        if idx in fallback_reasons:
            entry["fallback_reason"] = fallback_reasons[idx]
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# BERTScore computation
# ---------------------------------------------------------------------------

def compute_bertscore(results: list[dict]) -> dict:
    import os
    from bert_score import score as bert_score_fn

    # Prefer offline cache; if HF_ENDPOINT is pre-set (e.g. hf-mirror.com)
    # by the caller, transformers will use that instead.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    refs = [r["golden"] for r in results]
    cands = [r["generated"] for r in results]

    P, R, F1 = bert_score_fn(
        cands, refs, lang="zh",
        model_type="bert-base-chinese", verbose=True,
    )

    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
        "per_case": [
            {
                "idx": results[i]["idx"],
                "precision": round(P[i].item(), 4),
                "recall": round(R[i].item(), 4),
                "f1": round(F1[i].item(), 4),
            }
            for i in range(len(results))
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="BERTScore eval for Table 1.")
    parser.add_argument("--d101", type=Path, default=D101_PATH)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--method", required=True,
        choices=["no_memory", "long_context", "summary_memory", "tppm_memory"],
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--min-turns", type=int, default=1,
        help="Minimum conversation turns required (default: 1, keep all). "
             "Set to 3+ to exclude trivial openings like '你好'.",
    )
    parser.add_argument(
        "--memory-bank", type=Path, default=DEFAULT_MEMORY_BANK,
        help="Path to D101 TPPM memory bank JSON (only used with --method tppm_memory).",
    )
    args = parser.parse_args()

    method = args.method
    gen_path = args.eval_dir / f"{method}_generations.json"
    score_path = args.eval_dir / f"{method}_bertscore.json"
    args.eval_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    test_cases = load_d101(args.d101)
    if args.max_cases:
        test_cases = test_cases[:args.max_cases]
    print(f"[INFO] Loaded {len(test_cases)} test cases from D101")

    # Step 2: Generate
    method_kwargs = {}
    if args.method == "tppm_memory":
        method_kwargs["memory_bank"] = args.memory_bank

    results = generate_responses(test_cases, args.method, args.min_turns, method_kwargs)
    with gen_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "method": method, "model": MODEL_PATH,
                "test_cases": len(results), "min_turns": args.min_turns,
                "generated_at": utc_now(),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {gen_path}")

    # Step 3: BERTScore
    scores = compute_bertscore(results)
    with score_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "method": method, "metric": "BERTScore (bert-base-chinese)",
                "test_cases": len(results), "min_turns": args.min_turns,
                "computed_at": utc_now(),
            },
            "summary": {
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            },
            "per_case": scores["per_case"],
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {score_path}")

    print(f"\n{'='*50}")
    print(f"BERTScore — {method}  (n={len(results)})")
    print(f"  Precision: {scores['precision']:.4f}")
    print(f"  Recall:    {scores['recall']:.4f}")
    print(f"  F1:        {scores['f1']:.4f}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
