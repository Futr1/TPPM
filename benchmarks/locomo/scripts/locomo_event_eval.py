#!/usr/bin/env python3
"""Stage 2b: Event Summarization evaluation on LoCoMo with TPPM memory.

Given a time range (session), retrieves TPPM memories and conversation context,
generates event descriptions per speaker, and evaluates with ROUGE-L.
(FactScore computation requires external API and is deferred.)

Usage:
    python3 locomo_event_eval.py                           # full 10 conversations
    python3 locomo_event_eval.py --max-convs 1             # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

import numpy as np

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/locomo'
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
MEMORY_BANK_PATH = ROOT / "outputs" / "locomo_memory_bank.json"
MODEL_PATH = "/root/autodl-tmp/wangqihao/base_model/Qwen3.5-9B"
EVAL_DIR = ROOT / "outputs"

MAX_MODEL_LEN = 32768

EVENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts significant events from conversations. "
    "For each speaker, list the key events that happened in the given time period. "
    "Output as a JSON object with speaker names as keys and arrays of event "
    "descriptions as values. Events should be specific, factual, and causally "
    "connected when possible. Only output the JSON object, nothing else."
)

# ===== Data loading =====

def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_memory_bank(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Memory bank not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    return {e.get("conv_id", ""): e for e in bank.get("conversations", [])}

def format_tppm_profile(memory_entry: dict[str, Any], top_k: int = 10) -> str:
    """Format TPPM memories as profile text."""
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

    lines = ["[Speaker Profile]"]
    for mem in all_memories[:top_k]:
        attr = mem.get("attribute", "")
        value = mem.get("value", "")
        if not value:
            continue
        lines.append(f"- {attr}: {value}")
    return "\n".join(lines) if len(lines) > 1 else "No profile information available."

# ===== ROUGE computation =====

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute ROUGE-L F1 score across all prediction/reference pairs."""
    try:
        from rouge import Rouge
        rouge = Rouge()
        all_scores = []
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip():
                all_scores.append(0.0)
                continue
            try:
                scores = rouge.get_scores(pred, ref, avg=True)
                all_scores.append(scores["rouge-l"]["f"])
            except Exception:
                all_scores.append(0.0)
        return float(np.mean(all_scores))
    except Exception:
        return 0.0

# ===== Generation =====

def generate_event_summaries(
    conversations: list[dict[str, Any]],
    memory_bank: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate event summaries using vLLM with TPPM context."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

    all_prompts: list[str] = []
    all_meta: list[tuple[int, str]] = []  # (conv_idx, session_key)

    for conv_idx, conv in enumerate(conversations):
        cid = conv.get("sample_id", "")
        memory_entry = memory_bank.get(cid)
        conv_data = conv["conversation"]
        event_summary = conv["event_summary"]

        profile_text = ""
        if memory_entry:
            profile_text = format_tppm_profile(memory_entry)

        for session_key in sorted(event_summary.keys()):
            if not session_key.startswith("events_session_"):
                continue
            session_num_str = session_key.replace("events_session_", "")
            session_text_key = f"session_{session_num_str}"
            turns = conv_data.get(session_text_key, [])
            session_text = "\n".join(
                f"{t.get('speaker', '')}: {t.get('text', '')}"
                for t in turns if isinstance(t, dict)
            )
            dt_key = f"session_{session_num_str}_date_time"
            date_time = conv_data.get(dt_key, "")

            prompt = (
                f"{profile_text}\n\n"
                f"[Session {session_num_str} ({date_time})]\n"
                f"{session_text}\n\n"
                f"Extract the significant events for each speaker in this session. "
                f"Output as JSON with speaker names as keys and event arrays as values. "
                f'Example: {{"Alice": ["Alice did X"], "Bob": ["Bob did Y"]}}'
            )
            messages = [
                {"role": "system", "content": EVENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            all_prompts.append(text)
            all_meta.append((conv_idx, session_key))

    print(f"[INFO] Generating event summaries for {len(all_prompts)} sessions...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Organize results per conversation
    results: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        conv_idx, session_key = all_meta[i]
        generated = output.outputs[0].text.strip()

        while len(results) <= conv_idx:
            results.append({
                "sample_id": conversations[len(results)]["sample_id"],
                "event_summaries": {},
            })

        results[conv_idx]["event_summaries"][session_key] = {
            "generated": generated,
            "ground_truth": conversations[conv_idx]["event_summary"].get(
                session_key, {}
            ),
        }

    return results

# ===== Evaluation =====

def evaluate_events(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute ROUGE-L over all event summary pairs."""
    all_preds: list[str] = []
    all_refs: list[str] = []

    for conv_result in results:
        for session_key, data in conv_result["event_summaries"].items():
            pred = data["generated"]
            gt = data["ground_truth"]
            ref_parts = []
            for speaker, events in gt.items():
                if isinstance(events, list):
                    ref_parts.extend(events)
                elif isinstance(events, str):
                    ref_parts.append(events)
            ref_text = " ".join(ref_parts)
            all_preds.append(pred)
            all_refs.append(ref_text)

    rouge_l = compute_rouge_l(all_preds, all_refs)
    return {
        "rouge_l": round(rouge_l * 100, 1),
        "num_evaluated": len(all_preds),
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Event Summarization evaluation with TPPM on LoCoMo.")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--memory-bank", type=Path, default=MEMORY_BANK_PATH)
    parser.add_argument("--output", type=Path,
                        default=EVAL_DIR / "locomo_event_results.json")
    parser.add_argument("--max-convs", type=int, default=None)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]
    print(f"[INFO] Loaded {len(conversations)} conversations")

    memory_bank = load_memory_bank(args.memory_bank)
    print(f"[INFO] Loaded TPPM memory bank: {len(memory_bank)} conversations indexed")

    results = generate_event_summaries(conversations, memory_bank)
    summary = evaluate_events(results)

    payload = {
        "metadata": {
            "method": "tppm_memory",
            "model": MODEL_PATH,
            "metrics": ["ROUGE-L"],
            "note": "FactScore computation requires external LLM API for atomic fact decomposition",
        },
        "summary": summary,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {args.output}")
    print(f"\n{'='*50}")
    print(f"LoCoMo Event Summarization — TPPM")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
