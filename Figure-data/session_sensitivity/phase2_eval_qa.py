#!/usr/bin/env python3
"""Phase 2: QA evaluation with session-truncated context.

For each (conv_id, N) pair from Phase 1:
  1. Load the TPPM memory extracted at that N
  2. Build hybrid context using only the first N sessions
  3. Filter QA by evidence session (≤ N)
  4. Compute Answerable F1 and Overall F1
  5. Aggregate across conversations

Usage:
    python3 phase2_eval_qa.py                       # full run
    python3 phase2_eval_qa.py --max-convs 2 -N 1 3 5  # quick test
"""

from __future__ import annotations

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
from tqdm import tqdm

# Allow importing LoCoMo official evaluation
_LOCOMO_ROOT = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo")
if str(_LOCOMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCOMO_ROOT))

from task_eval.evaluation import eval_question_answering

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
PROFILES_DIR = ROOT / "extracted_profiles"
EVAL_DIR = ROOT / "eval_results"
AGGREGATE_PATH = ROOT / "eval_results" / "aggregate_results.json"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "REDACTED_DEEPSEEK_KEY"

CONCURRENCY = 8
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 5
MAX_TOKENS = 256

DEFAULT_N_VALUES = [1, 3, 5, 7, 10, 15, 20]

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

def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_profile(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_sorted_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict], str]]:
    conv_data = conv["conversation"]
    sessions: list[tuple[int, str, list[dict], str]] = []
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.replace("session_", ""))
            except ValueError:
                continue
            turns = conv_data[key]
            dt_key = f"session_{num}_date_time"
            date_time = conv_data.get(dt_key, "")
            sessions.append((num, key, turns, date_time))
    sessions.sort(key=lambda x: x[0])
    return sessions


# ===== QA filtering by evidence session =====

def get_max_evidence_session(qa_item: dict[str, Any]) -> int:
    """Extract the maximum session number referenced in evidence.

    Evidence format: 'D{s}:{m}' where s = session number, m = message index.
    Returns 0 if no valid evidence found.
    """
    max_session = 0
    for ev in qa_item.get("evidence", []):
        if isinstance(ev, str) and ev.startswith("D") and ":" in ev:
            try:
                s = int(ev.split(":")[0][1:])
                max_session = max(max_session, s)
            except (ValueError, IndexError):
                pass
    return max_session


def filter_qa_by_session(
    qa_list: list[dict[str, Any]],
    max_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split QA into answerable (evidence ≤ max_n) and unanswerable.

    Returns:
        (answerable_qa, unanswerable_qa)
    """
    answerable = []
    unanswerable = []
    for qa in qa_list:
        if get_max_evidence_session(qa) <= max_n:
            answerable.append(qa)
        else:
            unanswerable.append(qa)
    return answerable, unanswerable


# ===== TPPM profile formatting =====

def format_tppm_profile(
    memory_entry: dict[str, Any],
    speakers: list[str] | None = None,
) -> str:
    lines: list[str] = ["[Structured Speaker Profile — extracted via TPPM]"]

    for memory_type in ["working_memory", "short_term_memory", "long_term_memory"]:
        pmus = memory_entry.get(memory_type, [])
        if not pmus:
            continue
        label = memory_type.replace("_", " ").title()
        lines.append(f"\n--- {label} ---")
        for pmu in pmus:
            attr = pmu.get("attribute", "")
            val = pmu.get("value", "")
            strength = pmu.get("strength", 1.0)
            ptype = pmu.get("profile_type", "")
            if attr and val:
                lines.append(f"  • {attr}: {val}  (type={ptype}, strength={strength:.2f})")

    return "\n".join(lines)


# ===== Hybrid context building =====

def build_hybrid_context(
    conv: dict[str, Any],
    memory_entry: dict[str, Any] | None,
    max_sessions: int | None = None,
) -> str:
    """Build hybrid context: TPPM profile + session summaries + full text.

    Args:
        conv: LoCoMo conversation dict.
        memory_entry: TPPM memory dict for this (conv, N) pair.
        max_sessions: If set, only include the first N sessions.
    """
    conv_data = conv["conversation"]
    session_summaries = conv.get("session_summary", {})
    sessions = get_sorted_sessions(conv)

    # Truncate sessions if max_sessions is set
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    parts: list[str] = []

    # Extract speaker names from first session
    speakers: list[str] = []
    for _, _, turns in sessions[:1]:
        for t in turns:
            if isinstance(t, dict):
                sp = t.get("speaker", "")
                if sp and sp not in speakers:
                    speakers.append(sp)

    # 1. TPPM profile
    if memory_entry:
        profile_text = format_tppm_profile(memory_entry, speakers=speakers)
        parts.append(profile_text)
        parts.append("")

    # 2. Session summaries (for sessions beyond recent window)
    summary_lines = ["[Earlier conversation summaries]"]
    for num, key, turns in sessions:
        summary_key = f"session_{num}_summary"
        summary = session_summaries.get(summary_key, "")
        if summary:
            dt_key = f"session_{num}_date_time"
            dt = conv_data.get(dt_key, "")
            summary_lines.append(f"Session {num} ({dt}): {summary}")
    if len(summary_lines) > 1:
        parts.append("\n".join(summary_lines))
        parts.append("")

    # 3. Full session text
    for num, key, turns in sessions:
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


# ===== Async generation =====

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


async def generate_and_eval_one_n(
    conversations: list[dict[str, Any]],
    n: int,
) -> dict[str, Any]:
    """Generate QA answers for all conversations at a given N and evaluate."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)

    # Per-conversation results
    conv_results: list[dict[str, Any]] = []
    # Dual-metric accumulators
    answerable_f1s: dict[str, list[float]] = {
        "overall": [], "temporal": [], "multi_hop": [], "open_domain": [],
        "adversarial": [], "single_hop": [],
    }
    overall_f1s: dict[str, list[float]] = {
        "overall": [], "temporal": [], "multi_hop": [], "open_domain": [],
        "adversarial": [], "single_hop": [],
    }

    category_names = {1: "multi_hop", 2: "single_hop", 3: "temporal",
                      4: "open_domain", 5: "adversarial"}

    # Fixed question set: all QA with evidence ≤ 20 (or conversation max)
    MAX_EVIDENCE_N = 20

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("sample_id", f"conv-{conv_idx}")
        all_sessions = get_sorted_sessions(conv)
        actual_n = min(n, len(all_sessions))

        # Load TPPM profile for this (conv, N)
        profile_path = PROFILES_DIR / f"{conv_id}_N{actual_n}.json"
        memory_entry = load_profile(profile_path)
        if memory_entry is None:
            print(f"  [WARN] No profile for {conv_id} N={actual_n}, skipping")
            continue

        # Build context with truncated sessions
        context = build_hybrid_context(conv, memory_entry, max_sessions=actual_n)

        # Filter QA
        all_qa = conv.get("qa", [])
        answerable_qa, unanswerable_qa = filter_qa_by_session(all_qa, actual_n)

        if not answerable_qa and not unanswerable_qa:
            continue

        # Fixed question set for Overall F1: QA with evidence ≤ MAX_EVIDENCE_N
        fixed_qa, _ = filter_qa_by_session(all_qa, MAX_EVIDENCE_N)

        # Generate answers for all QA in the fixed set
        all_tasks: list[tuple[list[dict[str, str]], int, int]] = []
        for qa_idx, qa_item in enumerate(fixed_qa):
            question = qa_item["question"]
            prompt_text = (
                f"{context}\n\n"
                f"Based on the above, write a short answer for the following "
                f"question in a few words. Do not write complete sentences. "
                f"Answer with exact words from the conversations whenever possible.\n\n"
                f"Question: {question}"
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            all_tasks.append((messages, conv_idx, qa_idx))

        # Generate answers
        results_map: dict[int, str] = {}
        pending = all_tasks[:]
        round_num = 0

        while pending:
            round_num += 1
            if round_num > 1:
                delay = min(60.0, 2 ** round_num)
                print(f"    [RETRY] Round {round_num}: {len(pending)} remaining")
                time.sleep(delay)

            tasks = [
                _generate_one(client, sem, msgs, ci, qi)
                for msgs, ci, qi in pending
            ]
            outputs = await asyncio.gather(*tasks, return_exceptions=True)

            next_pending = []
            for item, (msgs, ci, qi) in zip(outputs, pending):
                if isinstance(item, Exception):
                    next_pending.append((msgs, ci, qi))
                else:
                    results_map[qi] = item[2]

            pending = next_pending
            if not pending:
                break

        # Build result dicts for evaluation
        # Answerable QA (evidence ≤ actual_n)
        ans_qa_for_eval = []
        for qa_idx, qa_item in enumerate(fixed_qa):
            if get_max_evidence_session(qa_item) <= actual_n:
                generated = results_map.get(qa_idx, "")
                generated = re.sub(r'<think>.*?</think>\s*', '', generated,
                                   flags=re.DOTALL).strip()
                gt = qa_item.get("answer") or qa_item.get("adversarial_answer", "")
                ans_qa_for_eval.append({
                    "question": qa_item["question"],
                    "answer": gt,
                    "category": qa_item["category"],
                    "evidence": qa_item.get("evidence", []),
                    "tppm_prediction": generated,
                })

        # Overall QA (fixed set, evidence ≤ MAX_EVIDENCE_N, unanswerable = 0 F1)
        overall_qa_for_eval = []
        for qa_idx, qa_item in enumerate(fixed_qa):
            generated = results_map.get(qa_idx, "")
            generated = re.sub(r'<think>.*?</think>\s*', '', generated,
                               flags=re.DOTALL).strip()
            gt = qa_item.get("answer") or qa_item.get("adversarial_answer", "")

            if get_max_evidence_session(qa_item) <= actual_n:
                overall_qa_for_eval.append({
                    "question": qa_item["question"],
                    "answer": gt,
                    "category": qa_item["category"],
                    "evidence": qa_item.get("evidence", []),
                    "tppm_prediction": generated,
                })
            else:
                # Unanswerable: model should say "I don't know"
                # Score as 0 F1 (wrong answer)
                overall_qa_for_eval.append({
                    "question": qa_item["question"],
                    "answer": gt,
                    "category": qa_item["category"],
                    "evidence": qa_item.get("evidence", []),
                    "tppm_prediction": generated,  # will likely score 0
                })

        # Evaluate Answerable F1
        if ans_qa_for_eval:
            try:
                scores, _, _ = eval_question_answering(
                    ans_qa_for_eval, eval_key="tppm_prediction")
                for i, qa in enumerate(ans_qa_for_eval):
                    cat = qa["category"]
                    cat_name = category_names.get(cat, "other")
                    if i < len(scores):
                        answerable_f1s[cat_name].append(scores[i])
                        answerable_f1s["overall"].append(scores[i])
            except Exception as exc:
                print(f"  [WARN] Answerable eval failed for {conv_id} N={actual_n}: {exc}")

        # Evaluate Overall F1 (fixed set)
        if overall_qa_for_eval:
            try:
                scores, _, _ = eval_question_answering(
                    overall_qa_for_eval, eval_key="tppm_prediction")
                for i, qa in enumerate(overall_qa_for_eval):
                    cat = qa["category"]
                    cat_name = category_names.get(cat, "other")
                    if i < len(scores):
                        overall_f1s[cat_name].append(scores[i])
                        overall_f1s["overall"].append(scores[i])
            except Exception as exc:
                print(f"  [WARN] Overall eval failed for {conv_id} N={actual_n}: {exc}")

        conv_results.append({
            "sample_id": conv_id,
            "n_sessions": actual_n,
            "num_answerable_qa": len(ans_qa_for_eval),
            "num_overall_qa": len(overall_qa_for_eval),
        })

    # Compute summary metrics
    def _mean_se(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values)
        return float(np.mean(arr)) * 100, float(np.std(arr, ddof=1) / np.sqrt(len(arr))) * 100

    summary = {
        "n": n,
        "num_conversations": len(conv_results),
        "num_answerable_qa_total": sum(c["num_answerable_qa"] for c in conv_results),
        "num_overall_qa_total": sum(c["num_overall_qa"] for c in conv_results),
        "answerable": {},
        "overall": {},
    }

    for metric_name in ["overall", "temporal", "multi_hop"]:
        ans_mean, ans_se = _mean_se(answerable_f1s[metric_name])
        ovr_mean, ovr_se = _mean_se(overall_f1s[metric_name])
        summary["answerable"][metric_name] = {
            "mean": round(ans_mean, 2),
            "se": round(ans_se, 2),
            "n_samples": len(answerable_f1s[metric_name]),
        }
        summary["overall"][metric_name] = {
            "mean": round(ovr_mean, 2),
            "se": round(ovr_se, 2),
            "n_samples": len(overall_f1s[metric_name]),
        }

    # Save per-N results
    out_path = EVAL_DIR / f"results_N{n}.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


async def run_all_n(
    conversations: list[dict[str, Any]],
    n_values: list[int],
) -> list[dict[str, Any]]:
    """Evaluate all N values sequentially."""
    all_summaries = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  Evaluating N = {n}")
        print(f"{'='*60}")

        summary = await generate_and_eval_one_n(conversations, n)
        all_summaries.append(summary)

        # Print interim results
        for metric in ["overall", "temporal", "multi_hop"]:
            ans = summary["answerable"].get(metric, {})
            ovr = summary["overall"].get(metric, {})
            print(f"  {metric}: Answerable={ans.get('mean', 0):.1f}±{ans.get('se', 0):.1f}, "
                  f"Overall={ovr.get('mean', 0):.1f}±{ovr.get('se', 0):.1f}")

    # Save aggregate
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AGGREGATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Phase 2 evaluation complete. Aggregate: {AGGREGATE_PATH}")
    return all_summaries


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: QA evaluation with session-truncated context.")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("-N", "--n-values", type=int, nargs="+",
                        default=DEFAULT_N_VALUES)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]

    print(f"[INFO] Conversations: {len(conversations)}")
    print(f"[INFO] N values: {args.n_values}")

    summaries = asyncio.run(run_all_n(conversations, args.n_values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
