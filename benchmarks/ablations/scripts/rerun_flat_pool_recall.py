#!/usr/bin/env python3
"""Re-run ONLY flat_pool's Recall (recall_user_shared_facts) questions to check
whether the +3.10 delta vs Full TPPM is stable or API noise.

Protocol: phase3_ablation_v2.py format (retrieve_memories top-K=5, multi-factor
scoring) + FULL-HISTORY context (conversation[:end_index]), matching the
deepseek/ablation_flat_pool results used in the paper.

Compares per-question scores to the existing
eval_results/deepseek/ablation_flat_pool/results.csv (Recall rows).
  - exact per-question match  => deterministic; +3.10 is stable (not API noise)
  - scores differ             => non-deterministic; this run is a fresh sample
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# import the v2 eval module (same retrieval/format functions)
SCRIPTS = REPO_ROOT / 'benchmarks/ablations'/scripts")
sys.path.insert(0, str(SCRIPTS))
import phase3_ablation_v2 as P3  # noqa: E402

from openai import OpenAI  # noqa: E402

DATASETS = P3.DATASETS
QUESTIONS_CSV = P3.QUESTIONS_CSV
SHARED_CONTEXTS_JSONL = P3.SHARED_CONTEXTS_JSONL
SNAPSHOTS_ABLATION = P3.SNAPSHOTS_ABLATION
RECALL_TYPE = "recall_user_shared_facts"

# flat_pool uses full retrieval weights + its own snapshot
COND = P3.CONDITIONS["ablation_flat_pool"]
RETRIEVE_W = COND["retrieve_weights"]
SNAPSHOT_DIR = SNAPSHOTS_ABLATION / COND["config_id"]

EXISTING_CSV = REPO_ROOT / 'benchmarks/ablations'/eval_results/deepseek/ablation_flat_pool/results.csv")


def build_context_full_history(conversation, end_index, memory_snapshot, question,
                               all_options, retrieve_weights, top_k):
    """Same as P3.build_context_window but with FULL history (conversation[:end_index])."""
    instructions = (
        "Find the most appropriate model response and give your final answer "
        "(a), (b), (c), or (d) after the special token <final_answer>."
    )
    conv = conversation[:end_index]                      # <-- FULL HISTORY
    conv_text = P3._messages_to_text(conv)
    question_block = f"{question}\n\n{instructions}\n\n{all_options}"

    memory_block = ""
    if memory_snapshot is not None:
        memories = P3.retrieve_memories(memory_snapshot, question, scene="general",
                                        top_k=top_k, retrieve_weights=retrieve_weights)
        memory_block = P3.format_memory_block(memories, scene="general")

    conv_tokens = len(P3.TOKENIZER.encode(conv_text))
    question_tokens = len(P3.TOKENIZER.encode(question_block))
    memory_tokens = len(P3.TOKENIZER.encode(memory_block)) if memory_block else 0
    total_used = conv_tokens + memory_tokens + question_tokens
    if total_used > P3.MAX_CONTEXT_TOKENS:
        excess = total_used - P3.MAX_CONTEXT_TOKENS + 200
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


def main():
    client = OpenAI(base_url=P3.API_BASE, api_key=P3.API_KEY)
    jsonl_index = P3.build_jsonl_index(SHARED_CONTEXTS_JSONL)

    # load flat_pool snapshots
    memory_cache = {}
    for fpath in SNAPSHOT_DIR.glob("*.json"):
        snap = __import__("json").load(open(fpath, encoding="utf-8"))
        memory_cache[snap.get("context_hash", fpath.stem)] = snap
    print(f"[INFO] loaded {len(memory_cache)} flat_pool snapshots from {SNAPSHOT_DIR}")

    # load existing Recall scores for comparison
    existing = {}
    with EXISTING_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["question_type"] == RECALL_TYPE:
                existing[r["question_id"]] = (
                    str(r.get("score", "")).strip().lower() in ("true", "1", "yes"),
                    r.get("predicted_answer", ""),
                )
    print(f"[INFO] {len(existing)} existing Recall scores to compare against")

    # iterate questions, keep only Recall
    run_scores = {}  # qid -> (correct_bool, predicted)
    prev_sid = None
    prev_context = None
    n = 0
    with QUESTIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["question_type"] != RECALL_TYPE:
                continue
            n += 1
            sid = row["shared_context_id"]
            end_index = int(row["end_index_in_shared_context"])
            if sid != prev_sid:
                prev_context = (P3.load_context_by_id(SHARED_CONTEXTS_JSONL, jsonl_index[sid])
                                if sid in jsonl_index else [])
                prev_sid = sid
            memory = memory_cache.get(sid)
            messages = build_context_full_history(
                prev_context, end_index, memory,
                row["user_question_or_message"], row["all_options"],
                retrieve_weights=RETRIEVE_W, top_k=P3.RETRIEVAL_TOP_K,
            )
            try:
                resp = client.chat.completions.create(
                    model=P3.API_MODEL, messages=messages, max_tokens=1024,
                    temperature=0, extra_body={"thinking": {"type": "disabled"}},
                )
                model_response = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[ERROR] {row['question_id']}: {e}")
                model_response = ""
            correct, predicted = P3.extract_answer(model_response, row["correct_answer"])
            run_scores[row["question_id"]] = (correct, predicted)
            if n % 20 == 0:
                print(f"  ...{n}/{len(existing)} done")

    # compare
    new_correct = sum(1 for v in run_scores.values() if v[0])
    new_total = len(run_scores)
    new_acc = 100 * new_correct / max(1, new_total)
    old_correct = sum(1 for v in existing.values() if v[0])
    old_acc = 100 * old_correct / max(1, len(existing))

    matches = 0
    flips = []
    for qid, (nc, np_) in run_scores.items():
        if qid in existing:
            oc, _ = existing[qid]
            if nc == oc:
                matches += 1
            else:
                flips.append((qid, oc, nc))

    print("\n" + "=" * 60)
    print(f"Recall questions re-run: {new_total}")
    print(f"existing flat_pool Recall acc: {old_correct}/{len(existing)} = {old_acc:.2f}%")
    print(f"new      flat_pool Recall acc: {new_correct}/{new_total} = {new_acc:.2f}%")
    print(f"per-question exact matches: {matches}/{len(run_scores)}")
    print(f"score flips: {len(flips)}")
    if flips:
        print("  flips (qid: old->new):")
        for qid, oc, nc in flips[:20]:
            print(f"    {qid[:8]}: {oc}->{nc}")
    if matches == len(run_scores):
        print("\n=> EXACT match: DeepSeek temp=0 is DETERMINISTIC here.")
        print("   +3.10 is a STABLE property of this question set, NOT API noise.")
    else:
        print(f"\n=> {len(flips)} differences: non-deterministic (or minor protocol drift).")
        print(f"   Fresh flat_pool Recall = {new_acc:.2f}% (was {old_acc:.2f}%).")
        baseline_recall = 75.97
        print(f"   Fresh delta vs Full TPPM Recall({baseline_recall}%) = {new_acc - baseline_recall:+.2f}")


if __name__ == "__main__":
    main()
