# TPPM 方法 BERTScore 评估 — 实现计划

> **面向执行代理：** 必读子技能 — 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐步实现。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标：** 在 BERTScore 评估流水线中新增 `tppm_memory` 方法，通过预提取的心理画像记忆增强模型回复质量评估。

**架构：** 双阶段流水线 — 阶段 1 新建 `tppm_extract_d101.py` 从 D101 离线提取 TPPM 记忆并保存为 JSON；阶段 2 修改 `eval_bertscore.py` 新增 `tppm_memory` 方法，加载 memory bank 拼接画像上下文后生成回复并计算 BERTScore。

**技术栈：** Python 3, asyncio + AsyncOpenAI, vLLM, bert-score, transformers

**涉及文件：**
- 新建: `Table1-data_split/scripts/tppm_extract_d101.py`
- 修改: `Table1-data_split/scripts/eval_bertscore.py`

---

### Task 1: 新建 TPPM D101 离线提取脚本（常量与工具函数）

**文件:**
- 新建: `Table1-data_split/scripts/tppm_extract_d101.py`

- [ ] **Step 1: 写入脚本头部——导入、路径常量、TPPM 参数**

```python
#!/usr/bin/env python3
"""Offline TPPM memory extraction for PsyDial-D101 BERTScore evaluation.

Extracts psychological profile memories from D101 conversations using
messages[:-1] (all turns before the final user message), then saves a
memory bank JSON for eval_bertscore.py --method tppm_memory.

Pipeline:
    1. Read PsyDial-D101.json (1278 cases)
    2. For each case, use messages[:-1] as extraction material
    3. Call DeepSeek API (async, 8-concurrent) to extract scored candidates
    4. Python-side phi scoring → tiered admission (phi > 0.62)
    5. Save filtered memory bank indexed by case_idx

Usage:
    python3 tppm_extract_d101.py                           # full 1278 cases
    python3 tppm_extract_d101.py --max-cases 30            # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table1-data_split")
D101_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "d101_tppm_memory_bank.json"
DEFAULT_FAILED = ROOT / "logs" / "d101_tppm_failed.jsonl"
DEFAULT_DEBUG = ROOT / "logs" / "d101_tppm_invalid_responses.jsonl"

# ===== Hybrid TPPM Hyperparameters (unchanged from tppm_extract.py) =====
ALPHA_1 = 0.25
ALPHA_2 = 0.30
ALPHA_3 = 0.25
ALPHA_4 = 0.20

# ===== Multi-level Thresholds =====
CONTEXT_THRESHOLD = 0.62   # phi > 0.62 → save for context injection
WRITE_THRESHOLD = 0.68     # phi > 0.68 → tier = "stable"
PROMOTE_THRESHOLD = 0.72   # phi > 0.72 → tier = "long_term"

# ===== API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "sk-REDACTED-do-not-commit-real-keys"

MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
CONCURRENCY = 8
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

MIN_HISTORY_TURNS = 3  # len(messages) must be >= 3 to have history

VALID_ATTRIBUTES = {"stressor", "affective_state", "coping_style"}
```

- [ ] **Step 2: 写入数据类定义和工具函数**

```python
class EmptyLLMResponseError(ValueError):
    pass

class LLMJSONParseError(ValueError):
    pass

class LLMSchemaError(ValueError):
    pass


@dataclass(slots=True)
class CandidateMemory:
    attribute: str
    value: str
    evidence: str
    r_score: float
    e_score: float
    u_score: float
    b_score: float
    phi: float
    tier: str  # "context_only" | "stable" | "long_term"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def compute_phi(r: float, e: float, u: float, b: float) -> float:
    return ALPHA_1 * r + ALPHA_2 * e + ALPHA_3 * u + ALPHA_4 * b


def assign_tier(phi: float) -> str:
    """Assign memory tier based on phi value."""
    if phi > PROMOTE_THRESHOLD:
        return "long_term"
    elif phi > WRITE_THRESHOLD:
        return "stable"
    else:
        return "context_only"


def load_d101(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("D101 must be a JSON array.")
    return data


def format_messages_for_extraction(case: dict[str, Any]) -> tuple[str | None, int]:
    """Extract messages[:-1] as dialogue text. Returns (text, num_messages)."""
    msgs = case.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < MIN_HISTORY_TURNS:
        return None, len(msgs) if isinstance(msgs, list) else 0

    # Use all messages except the last (the final user message)
    history = msgs[:-1]

    lines: list[str] = []
    for m in history:
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    if not lines:
        return None, len(msgs)
    return "\n".join(lines), len(msgs)


def build_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
```

- [ ] **Step 3: 提交第一步和第二步**

```bash
git add Table1-data_split/scripts/tppm_extract_d101.py
git commit -m "feat: add TPPM D101 extraction script skeleton with constants and utilities

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 实现 LLM 提取核心逻辑（system prompt、JSON 解析、phi 计算）

**文件:**
- 修改: `Table1-data_split/scripts/tppm_extract_d101.py`

- [ ] **Step 1: 写入 system prompt 构建和 JSON 清理函数**

```python
def build_system_prompt() -> str:
    return """
You are a psychological profile memory extractor for a Hybrid TPPM system.

Your only job is feature extraction and scoring. Python code makes the final write decision.

Read the truncated mental-health support dialogue and extract candidate PPMUs for:
- stressor: core pressure source or triggering situation
- affective_state: emotional state or mood pattern
- coping_style: how the user responds, copes, avoids, suppresses, seeks help, etc.

For each candidate, output four scores [0.0, 1.0]:
- r_score: relevance to psychological profile
- e_score: explicitness of evidence
- u_score: utility for future support
- b_score: tendency to persist beyond fleeting utterance

Constraints:
1. Output exactly one JSON object, nothing else.
2. No Markdown, explanations, comments, or trailing commas.
3. Do not copy long raw dialogue. Keep evidence short.
4. No clinical diagnosis labels.
5. If nothing useful, return {"candidates":[]}.

Required schema:
{"candidates":[{"attribute":"stressor|affective_state|coping_style","value":"...","evidence":"...","r_score":0.0,"e_score":0.0,"u_score":0.0,"b_score":0.0}]}
""".strip()


def clean_json_text(content: str) -> str:
    stripped = (content or "").lstrip("﻿").strip()
    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    return stripped


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise LLMJSONParseError("No complete JSON object found.")


def parse_llm_json(content: str) -> dict[str, Any]:
    stripped = clean_json_text(content)
    if not stripped:
        raise EmptyLLMResponseError("Empty content.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        repaired = remove_trailing_commas(stripped)
        if repaired != stripped:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        try:
            return first_json_object(repaired)
        except Exception as e2:
            raise LLMJSONParseError(f"Cannot parse: {str(e)[:200]}") from e2
```

- [ ] **Step 2: 写入候选记忆标准化和准入函数**

```python
def normalize_candidate(raw: dict) -> tuple[CandidateMemory | None, str | None]:
    """Validate one raw LLM candidate, compute phi, and assign tier."""
    if not isinstance(raw, dict):
        return None, "not an object"

    attr = str(raw.get("attribute", "")).strip().lower()
    if attr not in VALID_ATTRIBUTES:
        return None, f"invalid attribute {attr!r}"

    value = raw.get("value")
    if not isinstance(value, str) or not value.strip():
        return None, "value must be non-empty string"

    evidence = raw.get("evidence", "")
    if not isinstance(evidence, str):
        return None, "evidence must be string"

    try:
        r = clamp(raw.get("r_score"))
        e = clamp(raw.get("e_score"))
        u = clamp(raw.get("u_score"))
        b = clamp(raw.get("b_score"))
    except Exception as exc:
        return None, str(exc)

    phi = round(compute_phi(r, e, u, b), 6)

    if phi <= CONTEXT_THRESHOLD:
        return None, f"phi={phi:.4f} <= CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}"

    tier = assign_tier(phi)
    return CandidateMemory(attr, value.strip(), evidence.strip(), r, e, u, b, phi, tier), None


def append_invalid_response(*, case_idx: int, model: str, attempt: int,
                            content: str, error: Exception) -> None:
    DEFAULT_DEBUG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": utc_now_iso(),
        "case_idx": case_idx,
        "model": model,
        "attempt": attempt,
        "error": repr(error),
        "content_preview": content[:2000],
    }
    with DEFAULT_DEBUG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 3: 写入异步 API 调用和重试逻辑**

```python
async def get_llm_candidates(
    dialogue_text: str,
    case_idx: int,
    client: AsyncOpenAI,
    model: str = API_MODEL,
    max_retries: int = MAX_RETRIES,
    max_tokens: int = MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Call DeepSeek API asynchronously to extract scored PPMU candidates."""
    if not dialogue_text.strip():
        return []

    system_prompt = build_system_prompt()
    user_prompt = (
        "Extract scored TPPM candidate memories from the following truncated dialogue.\n"
        "Remember: output JSON only.\n\n"
        f"{dialogue_text}"
    )

    max_token_cap = max(max_tokens, 4096)
    for attempt in range(1, max_retries + 1):
        attempt_max_tokens = min(max_tokens * (2 ** (attempt - 1)), max_token_cap)
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=attempt_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                raise EmptyLLMResponseError("Empty response.")

            payload = parse_llm_json(content)
            if not isinstance(payload, dict) or "candidates" not in payload:
                raise LLMSchemaError("Missing 'candidates' key.")
            return payload["candidates"]
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"LLM extraction failed after {max_retries} attempts.")
```

- [ ] **Step 4: 提交 Task 2**

```bash
git add Table1-data_split/scripts/tppm_extract_d101.py
git commit -m "feat: add LLM extraction core — system prompt, JSON parse, async API, phi/tier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 实现单 case 处理和批量并发调度

**文件:**
- 修改: `Table1-data_split/scripts/tppm_extract_d101.py`

- [ ] **Step 1: 写入单 case 异步处理函数**

```python
async def process_case(
    case_idx: int,
    dialogue_text: str,
    client: AsyncOpenAI,
    model: str = API_MODEL,
) -> tuple[int, list[CandidateMemory]]:
    """Run TPPM extraction for a single D101 case. Returns (case_idx, memories)."""
    raw_candidates = await get_llm_candidates(
        dialogue_text, case_idx, client, model=model,
        max_retries=MAX_RETRIES, max_tokens=MAX_TOKENS,
    )
    accepted: list[CandidateMemory] = []
    for raw in raw_candidates:
        candidate, _reason = normalize_candidate(raw)
        if candidate is not None:
            accepted.append(candidate)
    return case_idx, accepted


async def run_extraction(
    dataset: list[dict[str, Any]],
    model: str = API_MODEL,
    concurrency: int = CONCURRENCY,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Run async concurrent extraction over all D101 cases.

    Returns:
        memories_out: list of {"case_idx": int, "tppm_memory": [...]} dicts
        skipped: count of cases skipped (insufficient history)
        failed: count of cases that failed after retries
        empty_memory: count of cases where phi all below threshold
    """
    client = build_client()
    sem = asyncio.Semaphore(concurrency)

    skipped = 0
    failed = 0
    empty_memory = 0
    tasks: list[asyncio.Task] = []
    case_indices: list[int] = []

    async def run_one(case_idx: int, dialogue_text: str) -> dict[str, Any] | None:
        async with sem:
            try:
                _, memories = await process_case(case_idx, dialogue_text, client, model)
                if not memories:
                    nonlocal empty_memory
                    empty_memory += 1
                return {"case_idx": case_idx, "tppm_memory": [asdict(m) for m in memories]}
            except Exception as exc:
                nonlocal failed
                failed += 1
                DEFAULT_FAILED.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "timestamp": utc_now_iso(),
                    "case_idx": case_idx,
                    "error": repr(exc),
                    "error_type": type(exc).__name__,
                    "dialogue_length": len(dialogue_text),
                    "model": model,
                    "max_retries": MAX_RETRIES,
                }
                with DEFAULT_FAILED.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return None

    for case in dataset:
        case_idx = case["idx"]
        dialogue_text, num_msgs = format_messages_for_extraction(case)
        if dialogue_text is None:
            skipped += 1
            continue
        case_indices.append(case_idx)
        tasks.append(asyncio.create_task(run_one(case_idx, dialogue_text)))

    print(f"[INFO] Total D101 cases: {len(dataset)}")
    print(f"[INFO] Skipped (insufficient history, < {MIN_HISTORY_TURNS} messages): {skipped}")
    print(f"[INFO] Extraction candidates: {len(tasks)}")
    print(f"[INFO] Concurrency: {concurrency}")

    results = []
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting TPPM memories")
    for coro in progress:
        result = await coro
        if result is not None:
            results.append(result)

    # Sort by case_idx for consistent output
    results.sort(key=lambda r: r["case_idx"])
    return results, skipped, failed, empty_memory
```

注意：在 Python 3.10+ 中，`nonlocal` 不能在嵌套函数中用于重新绑定闭包变量。需要将计数器改为可变容器。

修正后的 `run_one` 使用计数器列表：

```python
async def run_extraction(
    dataset: list[dict[str, Any]],
    model: str = API_MODEL,
    concurrency: int = CONCURRENCY,
) -> tuple[list[dict[str, Any]], int, int, int]:
    client = build_client()
    sem = asyncio.Semaphore(concurrency)

    skipped = 0
    failed = [0]   # mutable container for nested function
    empty_memory = [0]
    tasks: list[asyncio.Task] = []

    async def run_one(case_idx: int, dialogue_text: str) -> dict[str, Any] | None:
        async with sem:
            try:
                _, memories = await process_case(case_idx, dialogue_text, client, model)
                if not memories:
                    empty_memory[0] += 1
                return {"case_idx": case_idx, "tppm_memory": [asdict(m) for m in memories]}
            except Exception as exc:
                failed[0] += 1
                DEFAULT_FAILED.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "timestamp": utc_now_iso(),
                    "case_idx": case_idx,
                    "error": repr(exc),
                    "error_type": type(exc).__name__,
                    "dialogue_length": len(dialogue_text),
                    "model": model,
                    "max_retries": MAX_RETRIES,
                }
                with DEFAULT_FAILED.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return None

    for case in dataset:
        case_idx = case["idx"]
        dialogue_text, num_msgs = format_messages_for_extraction(case)
        if dialogue_text is None:
            skipped += 1
            continue
        tasks.append(asyncio.create_task(run_one(case_idx, dialogue_text)))

    print(f"[INFO] Total D101 cases: {len(dataset)}")
    print(f"[INFO] Skipped (insufficient history, < {MIN_HISTORY_TURNS} messages): {skipped}")
    print(f"[INFO] Extraction candidates: {len(tasks)}")
    print(f"[INFO] Concurrency: {concurrency}")

    results = []
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting TPPM memories")
    for coro in progress:
        result = await coro
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r["case_idx"])
    return results, skipped, failed[0], empty_memory[0]
```

- [ ] **Step 2: 写入 main 函数**

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="TPPM memory extraction for D101 BERTScore eval.")
    parser.add_argument("--d101", type=Path, default=D101_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    dataset = load_d101(args.d101)
    if args.max_cases:
        dataset = dataset[:args.max_cases]

    print(f"[INFO] Input: {args.d101}")
    print(f"[INFO] Cases: {len(dataset)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] API base: {args.api_base}")
    print(f"[INFO] Max tokens: {args.max_tokens}")
    print(f"[INFO] phi = {ALPHA_1}*r + {ALPHA_2}*e + {ALPHA_3}*u + {ALPHA_4}*b")
    print(f"[INFO] CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}, "
          f"WRITE_THRESHOLD={WRITE_THRESHOLD}, PROMOTE_THRESHOLD={PROMOTE_THRESHOLD}")

    memories_out, skipped, failed, empty_memory = asyncio.run(
        run_extraction(dataset, model=args.model, concurrency=args.concurrency)
    )

    total_memories = sum(len(m["tppm_memory"]) for m in memories_out)
    payload = {
        "metadata": {
            "source": "PsyDial-D101",
            "extraction_range": "messages[:-1]",
            "extractor_model": args.model,
            "alphas": {"r": ALPHA_1, "e": ALPHA_2, "u": ALPHA_3, "b": ALPHA_4},
            "context_threshold": CONTEXT_THRESHOLD,
            "write_threshold": WRITE_THRESHOLD,
            "promote_threshold": PROMOTE_THRESHOLD,
            "tier_labels": {
                "context_only": "0.62 < phi <= 0.68",
                "stable": "0.68 < phi <= 0.72",
                "long_term": "phi > 0.72",
            },
            "total_cases": len(dataset),
            "skipped_short_cases": skipped,
            "failed_cases": failed,
            "extracted_cases": len(memories_out),
            "empty_memory_cases": empty_memory,
            "total_memories": total_memories,
            "min_history_turns": MIN_HISTORY_TURNS,
        },
        "memories": memories_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {args.output}")
    print(f"[DONE] Extracted: {len(memories_out)} cases, {total_memories} memories")
    print(f"[DONE] Skipped (short): {skipped}, Failed: {failed}, Empty memories: {empty_memory}")
    if failed:
        print(f"[DONE] Failure log: {args.failed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 提交 Task 3**

```bash
git add Table1-data_split/scripts/tppm_extract_d101.py
git commit -m "feat: add async batch extraction with concurrency and main entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 修改 eval_bertscore.py — 新增 CLI 参数和工具函数

**文件:**
- 修改: `Table1-data_split/scripts/eval_bertscore.py`

- [ ] **Step 1: 新增常量、属性标签映射和 memory bank 加载函数**

在文件顶部的常量区域（`ROOT` 和 `MODEL_PATH` 行之后），新增：

```python
DEFAULT_MEMORY_BANK = ROOT / "outputs" / "d101_tppm_memory_bank.json"

ATTRIBUTE_LABELS = {
    "stressor": "压力来源",
    "affective_state": "情绪状态",
    "coping_style": "应对方式",
}
```

在 `load_d101` 函数之后，新增 memory bank 加载和索引函数：

```python
def load_memory_bank(path: Path) -> dict[int, list[dict]]:
    """Load D101 TPPM memory bank and index by case_idx.

    Returns:
        dict mapping case_idx (int) → list of memory dicts.
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
        1. 压力来源: <value>；显著性=<phi>；简要依据=<evidence>
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
            suffix += f"；显著性={float(phi):.3f}"
        if evidence:
            suffix += f"；简要依据={evidence[:120]}"
        lines.append(f"{i}. {label}: {value}{suffix}")
    return "\n".join(lines) if lines else "暂无可用的长期画像背景。"
```

- [ ] **Step 2: 提交 Step 1**

```bash
git add Table1-data_split/scripts/eval_bertscore.py
git commit -m "feat: add memory bank loader and format_memory_background to eval_bertscore

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 实现 build_messages_tppm_memory 和降级策略

**文件:**
- 修改: `Table1-data_split/scripts/eval_bertscore.py`

- [ ] **Step 1: 在 `build_messages_summary_memory` 函数之后，新增 `build_messages_tppm_memory` 函数**

```python
def build_messages_tppm_memory(
    case: dict,
    memory_index: dict[int, list[dict]],
) -> tuple[list[dict], str | None]:
    """Build messages with TPPM psychological profile as context.

    Args:
        case: D101 case dict with 'idx' and 'messages'.
        memory_index: dict mapping case_idx → list of TPPM memory dicts.

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
```

- [ ] **Step 2: 修改 `generate_responses` 函数，添加 tppm_memory 分支**

在 `generate_responses` 函数中：
1. 将 `method` 参数的类型改为接受 `tppm_memory`
2. 添加 tppm_memory 分支的 prompt 构造逻辑

找到函数中的这段代码（约第 160-174 行）：

```python
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
        else:
            raise ValueError(f"Unknown method: {method}")
```

替换为：

```python
    # Load TPPM memory bank if needed
    memory_index: dict[int, list[dict]] = {}
    fallback_reasons: dict[int, str] = {}
    if method == "tppm_memory":
        memory_bank_path = method_kwargs.get("memory_bank", DEFAULT_MEMORY_BANK)
        memory_index = load_memory_bank(memory_bank_path)
        print(f"[INFO] Loaded TPPM memory bank: {len(memory_index)} cases indexed")

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
```

同时，在 `generate_responses` 函数签名中新增 `method_kwargs` 参数：

```python
def generate_responses(
    test_cases: list[dict],
    method: str,
    min_turns: int,
    method_kwargs: dict | None = None,
) -> list[dict]:
```

在函数体开头添加：

```python
    if method_kwargs is None:
        method_kwargs = {}
```

- [ ] **Step 3: 在生成结果中添加 fallback_reason 字段**

找到结果收集代码（约第 188-196 行）：

```python
    results = []
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text.strip()
        results.append({
            "idx": valid_cases[i]["idx"],
            "golden": valid_cases[i]["golden"]["content"],
            "generated": generated,
        })
```

替换为：

```python
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
```

- [ ] **Step 4: 修改 main 函数，传递 method_kwargs 和新增 CLI 参数**

在 `main` 函数中，找到 `parser.add_argument("--min-turns", ...)` 之后，新增：

```python
    parser.add_argument(
        "--memory-bank", type=Path, default=DEFAULT_MEMORY_BANK,
        help="Path to D101 TPPM memory bank JSON (only used with --method tppm_memory).",
    )
```

同时修改 `--method` 的 choices：

```python
    parser.add_argument(
        "--method", required=True,
        choices=["no_memory", "long_context", "summary_memory", "tppm_memory"],
    )
```

找到 `generate_responses` 的调用（约第 267 行），改为：

```python
    method_kwargs = {}
    if args.method == "tppm_memory":
        method_kwargs["memory_bank"] = args.memory_bank

    results = generate_responses(test_cases, args.method, args.min_turns, method_kwargs)
```

- [ ] **Step 5: 提交 Task 5**

```bash
git add Table1-data_split/scripts/eval_bertscore.py
git commit -m "feat: add tppm_memory method with fallback strategy to eval_bertscore

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Smoke test 和验证

**文件:**
- 无新建文件

- [ ] **Step 1: Smoke test — Phase 1 提取（少量 case）**

```bash
cd /root/autodl-tmp/wangqihao/Table1-data_split
python3 scripts/tppm_extract_d101.py --max-cases 10
```

预期输出:
```
[INFO] Input: .../PsyDial-D101.json
[INFO] Cases: 10
...
[DONE] .../d101_tppm_memory_bank.json
[DONE] Extracted: N cases, M memories
```

- [ ] **Step 2: 验证 memory bank JSON 结构**

```bash
python3 -c "
import json
with open('outputs/d101_tppm_memory_bank.json') as f:
    bank = json.load(f)
meta = bank['metadata']
print('Metadata keys:', list(meta.keys()))
print('Context threshold:', meta['context_threshold'])
print('Extracted cases:', meta['extracted_cases'])
print('Total memories:', meta['total_memories'])
if bank['memories']:
    m = bank['memories'][0]
    print('First entry case_idx:', m['case_idx'])
    if m['tppm_memory']:
        mem = m['tppm_memory'][0]
        print('First memory keys:', list(mem.keys()))
        print('First memory tier:', mem.get('tier'))
"
```

- [ ] **Step 3: Smoke test — Phase 2 BERTScore eval（少量 case）**

```bash
cd /root/autodl-tmp/wangqihao/Table1-data_split
python3 scripts/eval_bertscore.py --method tppm_memory --max-cases 10 --min-turns 3
```

预期输出:
```
[INFO] Loaded TPPM memory bank: N cases indexed
[INFO] Generating responses for N cases (method=tppm_memory)...
...
[SAVED] outputs/eval/tppm_memory_generations.json
[SAVED] outputs/eval/tppm_memory_bertscore.json

BERTScore — tppm_memory  (n=N)
  Precision: X.XXXX
  Recall:    X.XXXX
  F1:        X.XXXX
```

- [ ] **Step 4: 验证输出 JSON schema 与现有方法一致**

```bash
python3 -c "
import json
# Compare schema of tppm_memory vs long_context output
for method in ['tppm_memory', 'long_context']:
    path = f'outputs/eval/{method}_bertscore.json'
    with open(path) as f:
        data = json.load(f)
    print(f'{method}:')
    print(f'  metadata keys: {list(data[\"metadata\"].keys())}')
    print(f'  summary keys: {list(data[\"summary\"].keys())}')
    print(f'  per_case count: {len(data[\"per_case\"])}')
    if data['per_case']:
        print(f'  per_case[0] keys: {list(data[\"per_case\"][0].keys())}')
    print()
"
```

- [ ] **Step 5: 检查 fallback 标记是否正确写入**

```bash
python3 -c "
import json
with open('outputs/eval/tppm_memory_generations.json') as f:
    data = json.load(f)
fallback_counts = {}
for r in data['results']:
    reason = r.get('fallback_reason', 'none')
    fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
print('Fallback distribution:')
for reason, count in sorted(fallback_counts.items()):
    print(f'  {reason}: {count}')
"
```

- [ ] **Step 6: 提交最终验证结果**

```bash
git add outputs/eval/tppm_memory_bertscore.json outputs/eval/tppm_memory_generations.json outputs/d101_tppm_memory_bank.json
git commit -m "test: smoke test results for tppm_memory BERTScore eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 自查清单

**1. Spec 覆盖检查:**
- [x] 3.2 多级阈值 → Task 1 Step 1 定义 CONTEXT/WRITE/PROMOTE_THRESHOLD，Task 2 Step 2 实现 assign_tier
- [x] 4.1 输入输出 → Task 1 Step 1 定义 D101_PATH，Task 3 Step 2 实现 main 输出
- [x] 4.2 API 配置 → Task 1 Step 1 定义 API 常量
- [x] 4.3 Memory Bank 格式 → Task 3 Step 2 构建 payload
- [x] 4.4 提取 System Prompt → Task 2 Step 1 build_system_prompt
- [x] 4.5 异步并发 → Task 3 Step 1 run_extraction with asyncio.Semaphore(8)
- [x] 5.1 CLI 参数 → Task 5 Step 4 新增 --memory-bank 和扩展 --method
- [x] 5.2 System Prompt → Task 5 Step 1 build_messages_tppm_memory
- [x] 5.3 降级策略 → Task 5 Step 1 三种 fallback
- [x] 5.4 format_memory_background → Task 4 Step 1
- [x] 5.5 输出文件 → 复用现有路径模式，无需额外代码
- [x] 6. 边界情况 → Task 5 Step 1 fallback 覆盖全部三种

**2. Placeholder 扫描:** 无 TBD/TODO，每步包含完整代码。

**3. 类型一致性:**
- `case_idx` 在 Phase 1 输出为 int → Phase 2 用 `int(case_idx)` 索引，一致
- `fallback_reason` 在 `build_messages_tppm_memory` 返回 `str | None` → `generate_responses` 正确处理
- `method_kwargs` 类型 `dict | None`，默认 `{}`，传递一致
