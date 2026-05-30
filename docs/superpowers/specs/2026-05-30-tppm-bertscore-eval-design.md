# TPPM Method BERTScore Evaluation — Design Spec

**Date:** 2026-05-30
**Status:** Approved
**Topic:** Add TPPM (Temporal Psychological Profile Memory) as a 4th evaluation method to the existing BERTScore evaluation pipeline.

---

## 1. Motivation

`eval_bertscore.py` currently evaluates 3 memory methods:

| Method | Context |
|--------|---------|
| `no_memory` | system + last user turn only |
| `long_context` | system + full conversation history |
| `summary_memory` | system + LLM summary + last user turn |

**TPPM is missing.** This spec adds `tppm_memory` — structured psychological profile memories extracted from earlier conversation turns, injected as context for response generation.

---

## 2. Architecture — Two-Phase Pipeline

```
Phase 1 (offline, one-time)                  Phase 2 (eval, repeatable)
┌────────────────────────────┐      ┌──────────────────────────────┐
│ tppm_extract_d101.py       │      │ eval_bertscore.py            │
│                            │  →   │  --method tppm_memory        │
│ 1278 D101 cases             │      │  --memory-bank <path>        │
│ messages[:-1] → DeepSeek   │      │                              │
│ async 8-concurrent          │      │ Load memory bank JSON        │
│ phi > 0.62 → tiered save   │      │ Format【画像背景】as context  │
│                            │      │ vLLM Qwen3.5-9B generate     │
│ Output:                     │      │ BERTScore vs golden          │
│  d101_tppm_memory_bank.json │      │                              │
└────────────────────────────┘      └──────────────────────────────┘
```

---

## 3. TPPM Parameters

### 3.1 Alpha Weights (unchanged — method definition)

```
α₁ = 0.25  (r_score: relevance)
α₂ = 0.30  (e_score: explicitness of evidence)
α₃ = 0.25  (u_score: utility for future support)
α₄ = 0.20  (b_score: tendency to persist)
phi = α₁·r + α₂·e + α₃·u + α₄·b
```

### 3.2 Threshold System

| Threshold | Value | Role in BERTScore Eval |
|-----------|-------|------------------------|
| `context_threshold` | 0.62 | phi > 0.62 → save for context injection |
| `write_threshold` | 0.68 | phi > 0.68 → tier = "stable" (marker only) |
| `promote_threshold` | 0.72 | phi > 0.72 → tier = "long_term" (marker only) |

Tier labels:
- `context_only`: 0.62 < phi ≤ 0.68
- `stable`: 0.68 < phi ≤ 0.72
- `long_term`: phi > 0.72

`promotion_min_sessions`, `distill_stability_threshold`, `distill_quality_threshold`, `distill_session_threshold` are not applicable (single-session evaluation scenario).

---

## 4. Phase 1 — `tppm_extract_d101.py`

### 4.1 Input/Output

| Item | Value |
|------|-------|
| Input | `datasets/PsyDial/PsyDial-D101/PsyDial-D101.json` |
| Extraction range | `messages[:-1]` — all turns before the final user message |
| Skip condition | `len(messages) ≤ 2` (no history to extract from) |
| Output | `Table1-data_split/outputs/d101_tppm_memory_bank.json` |
| Failed log | `Table1-data_split/logs/d101_tppm_failed.jsonl` |

### 4.2 API Configuration

| Setting | Value |
|---------|-------|
| API base | `https://api.deepseek.com` |
| Model | `deepseek-v4-flash` |
| Temperature | 0 |
| Response format | `json_object` |
| Max tokens (output) | 2048 |
| Max retries | 5 with exponential backoff |
| Concurrency | asyncio + Semaphore(8) |

### 4.3 Memory Bank Output Format

```json
{
  "metadata": {
    "source": "PsyDial-D101",
    "extraction_range": "messages[:-1]",
    "extractor_model": "deepseek-v4-flash",
    "alphas": {"r": 0.25, "e": 0.30, "u": 0.25, "b": 0.20},
    "context_threshold": 0.62,
    "write_threshold": 0.68,
    "promote_threshold": 0.72,
    "tier_labels": {
      "context_only": "0.62 < phi <= 0.68",
      "stable": "0.68 < phi <= 0.72",
      "long_term": "phi > 0.72"
    },
    "total_cases": 1278,
    "skipped_short_cases": "<count>",
    "failed_cases": "<count>",
    "extracted_cases": "<count>",
    "total_memories": "<count>"
  },
  "memories": [
    {
      "case_idx": 0,
      "tppm_memory": [
        {
          "attribute": "stressor",
          "value": "...",
          "evidence": "...",
          "r_score": 0.85,
          "e_score": 0.90,
          "u_score": 0.75,
          "b_score": 0.70,
          "phi": 0.795,
          "tier": "long_term"
        }
      ]
    }
  ]
}
```

Key design: indexed by `case_idx` (D101's `"idx"` field), not `session_id`. D101 cases have no session-level identifiers — each case is an independent evaluation unit.

### 4.4 Extraction System Prompt

Unchanged from `tppm_extract.py` — same psychological profile memory extractor system prompt, same `{"candidates":[...]}` schema.

---

## 5. Phase 2 — `eval_bertscore.py` Modifications

### 5.1 New CLI Arguments

```
--method tppm_memory          (add to choices)
--memory-bank <path>          (default: outputs/d101_tppm_memory_bank.json)
```

### 5.2 System Prompt

```python
BASE_SYSTEM = (
    "你是一名经验丰富的专业心理咨询师。"
    "请根据对话历史，直接给出下一句回复。"
    "只输出回复文本本身，不要输出思考过程、分析、解释或任何额外内容。"
)

# tppm_memory variant — base + 画像 appendix
def build_messages_tppm_memory(case, memory_bank):
    memories = memory_bank.get(str(case["idx"]))

    if not memories:
        # Fallback: no memories → long_context
        return build_messages_long_context(case)

    memory_text = format_memory_background(memories)

    system_content = (
        f"{BASE_SYSTEM}\n\n"
        f"【来访者长期画像 — 内部参考】\n"
        f"{memory_text}\n\n"
        f"注意：请自然运用画像信息理解来访者，"
        f"不要在回复中直接复述画像内容或提及记忆系统。"
    )

    return [
        {"role": "system", "content": system_content},
        *case["messages"],
    ]
```

Design principles:
- Base instruction identical across all 4 methods — fair comparison
- Only TPPM method appends the【画像】block
- vs tppm_memory, long_context differs ONLY in the presence of memory context
- Conservative constraint ("自然运用") — minimal steering, maximal measurement of memory's incremental value

### 5.3 Fallback Strategy

| Condition | Fallback | `fallback_reason` in output |
|-----------|----------|-----------------------------|
| `len(messages) ≤ 2` | `no_memory` | `"insufficient_history"` |
| No memories for idx in bank | `long_context` | `"no_memories_above_threshold"` |
| Extraction failed (not in bank) | `long_context` | `"extraction_failed"` |

Fallback cases are included in BERTScore calculation (consistent with how other methods handle edge cases).

### 5.4 `format_memory_background()`

Reuses the same format as `teacher_distill.py`:

```
1. 压力来源: <value>；显著性=<phi>；简要依据=<evidence>
2. 情绪状态: <value>；显著性=<phi>；简要依据=<evidence>
3. 应对方式: <value>；显著性=<phi>；简要依据=<evidence>
```

If no memories: returns `"暂无可用的长期画像背景。"` — but this path is handled by fallback before reaching format.

### 5.5 Output Files

| File | Path |
|------|------|
| Generations | `outputs/eval/tppm_memory_generations.json` |
| BERTScore | `outputs/eval/tppm_memory_bertscore.json` |

Output JSON schema identical to existing methods — metadata + per_case + summary.

---

## 6. Edge Cases

| Case | Handling |
|------|----------|
| Single-message D101 case (101 cases) | Phase 1: skip. Phase 2: fallback → no_memory |
| Multi-turn but no qualifying memories | Phase 1: empty array. Phase 2: fallback → long_context |
| Very long dialogues (max 149 turns) | No truncation needed — DeepSeek v4 1M context window covers all |
| DeepSeek API transient failure | 5 retries with exponential backoff + jitter. Fail → log, Phase 2 falls back |
| `--min-turns` filter | Same as other methods — skip cases below threshold before generate |
| `--max-cases` filter | Same as other methods — limit total cases |

---

## 7. Fair Comparison Table

All 4 methods share:
- Same base system prompt (first 3 lines)
- Same model: vLLM Qwen3.5-9B, temp=0, max_tokens=256
- Same metric: BERTScore bert-base-chinese
- Same output format

The ONLY variable across methods is **what context the model sees**:

```
no_memory:       BASE + last user
long_context:    BASE + all messages
summary_memory:  BASE + 【摘要】+ last user  (external LLM summary)
tppm_memory:     BASE + 【画像】+ all messages (structured profile memory)
```

---

## 8. Implementation Checklist

- [ ] Create `Table1-data_split/scripts/tppm_extract_d101.py`
  - [ ] Async OpenAI client with Semaphore(8)
  - [ ] Reuse TPPM extraction logic (system prompt, parsing, phi, tiering)
  - [ ] Skip `len(messages) ≤ 2`
  - [ ] Save `d101_tppm_memory_bank.json` with full metadata
  - [ ] Log failures to `d101_tppm_failed.jsonl`
- [ ] Modify `Table1-data_split/scripts/eval_bertscore.py`
  - [ ] Add `tppm_memory` to `--method` choices
  - [ ] Add `--memory-bank` argument
  - [ ] Implement `build_messages_tppm_memory()` with fallback
  - [ ] Implement `format_memory_background()` (copy from teacher_distill)
  - [ ] Handle `min_turns` / `max_cases` consistently
- [ ] Smoke test with `--max-cases 10`
- [ ] Verify output JSON schema matches existing methods
