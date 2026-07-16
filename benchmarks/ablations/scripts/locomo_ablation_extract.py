#!/usr/bin/env python3
"""LoCoMo ablation: extract memory banks for each ablation variant.

Based on benchmarks/locomo/scripts/locomo_tppm_extract.py, adapted for ablation configs.

Variants:
    baseline              — default TPMConfig
    ablation_consolidation — promote_threshold=999.0 (never promote)
    ablation_branching    — context_threshold=1.0 (no scene branching)
    ablation_decay        — all decay_lambdas=0.0 + skip decay_long_term()

Note: w/o Evidence and w/o LTM reuse the baseline memory bank;
      their modifications are applied at evaluation time.

Usage:
    python3 locomo_ablation_extract.py                                # all variants
    python3 locomo_ablation_extract.py --variant ablation_branching   # single variant
    python3 locomo_ablation_extract.py --max-convs 1                  # smoke test
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

from tppm.core.memory import TemporalProfileMemory, TPMConfig
from tppm.core.models import ProfileCandidate, utc_now

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/ablations'
LOCOMO_PATH = Path("/root/autodl-tmp/wangqihao/datasets/LoCoMo/data/locomo10.json")
SNAPSHOTS_DIR = ROOT / "memory_snapshots" / "locomo"
FAILED_DIR = ROOT / "logs"

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
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# ===== Variant definitions =====
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Full TPPM (baseline)",
        "tpm_config": TPMConfig(),
        "skip_decay": False,
    },
    "ablation_consolidation": {
        "label": "w/o Consolidation",
        "tpm_config": TPMConfig(
            promote_threshold=999.0,
            promotion_min_sessions=999,
        ),
        "skip_decay": False,
    },
    "ablation_branching": {
        "label": "w/o Scene Branching",
        "tpm_config": TPMConfig(
            context_threshold=1.0,
        ),
        "skip_decay": False,
    },
    "ablation_decay": {
        "label": "w/o Temporal Decay",
        "tpm_config": TPMConfig(
            decay_lambdas={
                "goal": 0.0, "interest": 0.0, "style": 0.0,
                "background": 0.0, "preference": 0.0, "general": 0.0,
            },
        ),
        "skip_decay": True,  # also skip decay_long_term() call
    },
}

# ===== Data loading =====

def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("locomo10.json must be a JSON array.")
    return data

def get_sorted_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict], str]]:
    conv_data = conv["conversation"]
    sessions: list[tuple[int, str, list[dict], str]] = []
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            num_str = key.replace("session_", "")
            try:
                num = int(num_str)
            except ValueError:
                continue
            turns = conv_data[key]
            dt_key = f"session_{num}_date_time"
            date_time = conv_data.get(dt_key, "")
            sessions.append((num, key, turns, date_time))
    sessions.sort(key=lambda x: x[0])
    return sessions

def format_turns_for_extraction(turns: list[dict]) -> str:
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if speaker and text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

# ===== Async LLM extraction (identical to original) =====

def build_extraction_payload(dialogue_text: str, scene: str = "general") -> dict[str, Any]:
    schema_hint = {
        "candidates": [
            {
                "attribute": "short_attribute_name",
                "value": "profile_value",
                "context": "supporting_span_or_short_reason",
                "profile_type": "background|preference|goal|style|interest|general",
                "scene": scene,
                "confidence": 0.0,
                "stability": 0.0,
                "recency": 1.0,
                "explicitness": 0.0,
                "user_relevance": 0.0,
                "source": "llm_deepseek",
            }
        ]
    }
    system_prompt = (
        "You are a profile candidate extractor for Temporal Profile Memory (TPM). "
        "Extract stable, reusable, and scene-conditioned user profile information "
        "from the latest conversation session. "
        "Return ONLY valid JSON, no markdown, no explanation."
    )
    user_prompt = (
        "Task: extract profile candidates for TPM.\n"
        f"Current scene: {scene}\n"
        f"Latest conversation session:\n{dialogue_text}\n\n"
        "Extraction rules:\n"
        "1. Keep only user-related profile facts, preferences, goals, style tendencies, "
        "identity, or stable context. Focus on information about the speakers (not the assistant).\n"
        "2. Ignore generic conversational filler and greetings.\n"
        "3. Use concise attribute names like identity, interest, preference, "
        "current_goal, style, project_focus, personal_background.\n"
        "4. profile_type must be one of: background, preference, goal, style, interest, general.\n"
        "5. confidence, stability, recency, explicitness, user_relevance must be numbers in [0,1].\n"
        "6. user_relevance measures how central this fact is to the user's enduring profile.\n"
        "7. Prefer higher stability for repeated or enduring traits; "
        "lower stability for short-term goals.\n"
        "8. If there is no useful profile memory candidate, return {\"candidates\": []}.\n\n"
        f"Output JSON schema example:\n{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    return {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

def parse_candidates_from_response(
    content: str, scene: str, original_text: str
) -> list[ProfileCandidate]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, dict):
        raw_list = parsed.get("candidates", [])
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        return []

    def _clamp(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _default_stability(ptype: str) -> float:
        defaults = {
            "background": 0.9, "style": 0.78, "preference": 0.72,
            "interest": 0.7, "goal": 0.56, "general": 0.6,
        }
        return defaults.get(ptype, 0.6)

    candidates: list[ProfileCandidate] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        attr = str(item.get("attribute", "")).strip()
        val = str(item.get("value", "")).strip()
        if not attr or not val:
            continue
        ptype = str(item.get("profile_type", "general")).strip().lower()
        if ptype not in {"background", "preference", "goal", "style", "interest", "general"}:
            ptype = "general"
        candidates.append(ProfileCandidate(
            attribute=attr,
            value=val,
            context=str(item.get("context") or original_text).strip() or original_text,
            profile_type=ptype,
            scene=str(item.get("scene") or scene).strip() or scene,
            confidence=_clamp(item.get("confidence"), 0.72),
            stability=_clamp(item.get("stability"), _default_stability(ptype)),
            recency=_clamp(item.get("recency"), 1.0),
            explicitness=_clamp(item.get("explicitness"), 0.8),
            user_relevance=_clamp(item.get("user_relevance"), 0.82),
            source=str(item.get("source") or "llm_deepseek").strip() or "llm_deepseek",
        ))
    return candidates

async def extract_candidates_async(
    client: AsyncOpenAI,
    dialogue_text: str,
    scene: str = "general",
    conv_id: str = "",
    session_num: int = 0,
) -> list[ProfileCandidate]:
    if not dialogue_text.strip():
        return []

    payload = build_extraction_payload(dialogue_text, scene)
    max_token_cap = max(MAX_TOKENS, 4096)

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_max_tokens = min(MAX_TOKENS * (2 ** (attempt - 1)), max_token_cap)
        try:
            resp = await client.chat.completions.create(
                model=API_MODEL,
                temperature=0,
                max_tokens=attempt_max_tokens,
                response_format={"type": "json_object"},
                messages=payload["messages"],
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content or ""
            if not content.strip():
                continue
            candidates = parse_candidates_from_response(content, scene, dialogue_text)
            return candidates[:8]
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    return []

# ===== Cross-session TPPM engine (variant-aware) =====

async def process_conversation(
    conv: dict[str, Any],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    variant_id: str,
    variant_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None]:
    conv_id = conv.get("sample_id", "unknown")
    sessions = get_sorted_sessions(conv)

    if not sessions:
        return conv_id, None, f"no sessions found for {conv_id}"

    tpm = TemporalProfileMemory(variant_cfg["tpm_config"])
    skip_decay = variant_cfg.get("skip_decay", False)
    error_msg: str | None = None

    for session_num, session_key, turns, date_time in sessions:
        dialogue_text = format_turns_for_extraction(turns)
        scene = f"session_{session_num}"

        tpm.start_session(scene=scene, session_id=f"{conv_id}_{session_key}")

        try:
            async with sem:
                candidates = await extract_candidates_async(
                    client, dialogue_text, scene=scene,
                    conv_id=conv_id, session_num=session_num,
                )
        except Exception as exc:
            FAILED_DIR.mkdir(parents=True, exist_ok=True)
            with (FAILED_DIR / "locomo_extract_failed.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "variant": variant_id,
                    "conv_id": conv_id,
                    "session_num": session_num,
                    "error": repr(exc),
                }, ensure_ascii=False) + "\n")
            error_msg = f"extraction failed at session {session_num}: {exc}"
            candidates = []

        if candidates:
            tpm.ingest_candidates(
                candidates, scene=scene,
                session_id=f"{conv_id}_{session_key}",
            )

        tpm.finish_session(scene=scene)

    # Long-term decay (skip for w/o Temporal Decay variant)
    if not skip_decay:
        tpm.decay_long_term()

    memory_dict = tpm.to_dict()
    memory_dict["conv_id"] = conv_id
    memory_dict["num_sessions"] = len(sessions)
    memory_dict["variant"] = variant_id
    return conv_id, memory_dict, error_msg

async def run_extraction(
    conversations: list[dict[str, Any]],
    variant_id: str,
    variant_cfg: dict[str, Any],
    concurrency: int = CONCURRENCY,
) -> tuple[list[dict[str, Any]], int, int]:
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        process_conversation(conv, client, sem, variant_id, variant_cfg)
        for conv in conversations
    ]

    memory_entries: list[dict[str, Any]] = []
    failed = 0
    total_pmus = 0

    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc=f"  {variant_id}")
    for coro in progress:
        conv_id, memory_dict, error = await coro
        if memory_dict is not None:
            memory_entries.append(memory_dict)
            total_pmus += (
                len(memory_dict.get("working_memory", []))
                + len(memory_dict.get("short_term_memory", []))
                + len(memory_dict.get("long_term_memory", []))
            )
        if error:
            failed += 1
            tqdm.write(f"[WARN] {variant_id}/{conv_id}: {error}")

    return memory_entries, failed, total_pmus

# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LoCoMo ablation: extract memory banks for each variant.")
    parser.add_argument("--variant", type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="Single variant (default: all)")
    parser.add_argument("--input", type=Path, default=LOCOMO_PATH)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    conversations = load_locomo(args.input)
    if args.max_convs:
        conversations = conversations[:args.max_convs]

    variants_to_run = (
        {args.variant: VARIANTS[args.variant]}
        if args.variant
        else VARIANTS
    )

    print(f"[INFO] LoCoMo ablation extraction")
    print(f"[INFO] Conversations: {len(conversations)}")
    print(f"[INFO] Variants: {list(variants_to_run.keys())}")

    for vid, vcfg in variants_to_run.items():
        output_dir = SNAPSHOTS_DIR / vid
        output_path = output_dir / "locomo_memory_bank.json"

        if output_path.exists():
            print(f"\n[SKIP] {vid}: output already exists at {output_path}")
            print(f"       Delete it to re-extract.")
            continue

        cfg = vcfg["tpm_config"]
        print(f"\n[INFO] Variant: {vid} ({vcfg['label']})")
        print(f"  write_thr={cfg.write_threshold}, promote_thr={cfg.promote_threshold}, "
              f"ctx_thr={cfg.context_threshold}, skip_decay={vcfg['skip_decay']}")

        memory_entries, failed, total_pmus = asyncio.run(
            run_extraction(conversations, vid, vcfg, concurrency=args.concurrency)
        )

        # Compute memory tier stats
        n_working = sum(len(e.get("working_memory", [])) for e in memory_entries)
        n_short = sum(len(e.get("short_term_memory", [])) for e in memory_entries)
        n_long = sum(len(e.get("long_term_memory", [])) for e in memory_entries)

        payload = {
            "metadata": {
                "source": "LoCoMo-10",
                "variant": vid,
                "extractor_model": API_MODEL,
                "tpm_config": {
                    "write_threshold": cfg.write_threshold,
                    "context_threshold": cfg.context_threshold,
                    "promote_threshold": cfg.promote_threshold,
                    "promotion_min_sessions": cfg.promotion_min_sessions,
                    "decay_lambdas": cfg.decay_lambdas,
                },
                "skip_decay": vcfg["skip_decay"],
                "total_conversations": len(conversations),
                "extracted_conversations": len(memory_entries),
                "failed_conversations": failed,
                "total_pmus": total_pmus,
                "tier_counts": {
                    "working": n_working,
                    "short_term": n_short,
                    "long_term": n_long,
                },
            },
            "conversations": memory_entries,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"  [DONE] {output_path}")
        print(f"  [DONE] PMUs: {total_pmus} (working={n_working}, "
              f"short_term={n_short}, long_term={n_long})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
