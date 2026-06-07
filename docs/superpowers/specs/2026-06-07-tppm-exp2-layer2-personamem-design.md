# TPPM Experiment 2 Layer 2 — PersonaMem Parameter Sensitivity Analysis Design Spec

**Date**: 2026-06-07
**Status**: Approved
**Context**: Experiment 2 Layer 2 of the TPPM paper — parameter sensitivity analysis of three dynamic mechanisms using the PersonaMem benchmark.

---

## 1. Objective

Quantify how TPPM's three dynamic mechanisms affect downstream QA accuracy on PersonaMem:

| Sub-experiment | Mechanism | Parameter(s) Scanned | Baseline |
|:---|:---|:---|:---|
| 2a — Consolidation | Admission/promotion gates | `write_threshold` (0.68), `promote_threshold` (0.72) | 0.68 / 0.72 |
| 2b — Decay | Type-conditional forgetting | `decay_lambdas` per profile type (6 lambdas) | [0.10, 0.07, 0.04, ...] |
| 2c — Branching | Scene-dependent preference tracking | `context_threshold` (0.62) | 0.62 |

## 2. Dataset

**PersonaMem 32K subset** (chosen for hardware constraints — 2× RTX 4090 at 32K context):

| File | Contents |
|:---|:---|
| `questions_32k.csv` | 589 QA items (all 7 question types, 15 topics, 20 personas) |
| `shared_contexts_32k.jsonl` | ~37 shared interaction histories (1 JSON object per line, keyed by hash) |

Each shared context is a list of messages (role=system/user/assistant). `role=system` messages mark session boundaries (16–20 sessions per context).

## 3. Architecture — Three-Phase Pipeline

### Phase 1: Candidate Extraction (LLM, one-time)

For each shared context, detect session boundaries, then call DeepSeek API to extract `ProfileCandidate` lists.

**Input**: `shared_contexts_32k.jsonl`
**Output**: `candidates/` directory — one JSON file per (shared_context, session) pair

```
candidates/
  <context_hash>/
    session_000.json   → {"candidates": [...], "session_idx": 0, "dialogue_text": "..."}
    session_001.json   → ...
    ...
```

**Key decisions**:
- Session boundary: `role == "system"` message
- LLM: DeepSeek API (`deepseek-v4-flash`), concurrency=8
- Scene tag: `session_{N}` (session ordinal within context)
- Each candidate file includes both raw LLM output and parsed `ProfileCandidate` fields for Phase 2 to ingest directly

### Phase 2: Memory Evolution Replay (Pure Python, per config)

For each parameter configuration, replay all sessions through a fresh `TemporalProfileMemory`. Zero LLM calls.

**Input**: Phase 1 `candidates/` + parameter sweep config
**Output**: `memory_snapshots/{config_id}/` — final memory state per context

```
memory_snapshots/
  baseline/
    <context_hash>.json   → {"working": [...], "short_term": [...], "long_term": [...]}
  write_0.60/
    ...
  write_0.68/
    ...
  decay_0.5x/
    ...
  ...
```

**Processing** (per config, per context):
1. `TPMConfig(params)` → `TemporalProfileMemory(config)`
2. For each session (chronological order):
   - Load candidates from Phase 1 output
   - `start_session(scene, session_id)`
   - `ingest_candidates(candidates, scene, session_id)`
   - `finish_session(scene)` → triggers consolidation promotion + working/short-term decay
3. After all sessions: `decay_long_term()`
4. Export `to_dict()` → JSON

**Parameter sweep ranges** (5 levels each, including baseline):

| Sub-exp | Parameter | Levels |
|:---|:---|:---|
| 2a | `write_threshold` | [0.56, 0.62, **0.68**, 0.74, 0.80] |
| 2a | `promote_threshold` | [0.60, 0.66, **0.72**, 0.78, 0.84] |
| 2b | `decay_lambdas` global scale | [0.25×, 0.5×, **1.0×**, 2.0×, 4.0×] |
| 2c | `context_threshold` | [0.50, 0.56, **0.62**, 0.68, 0.74] |

### Phase 3: QA Evaluation (vLLM inference)

For each (config, QA question) pair, build the 32K-token context window and query Qwen3.5-9B via vLLM.

**Input**: PersonaMem QA CSV + shared contexts + Phase 2 memory snapshots
**Output**: `eval_results/{config_id}/results.csv`

**Context window assembly** (≤32K tokens):
```
[Conversation history (truncated to end_index_in_shared_context)]
[TPPM Structured Memory Block]
  attribute: value | type: X | confidence: Y | stability: Z | scene: S
  attribute: value | type: X | confidence: Y | stability: Z | scene: S
  ...
[Question + Options + <final_answer> instruction]
```

**Memory formatting** (ordered by stability × confidence, top-K to fit remaining budget):
```
[TPPM Memory]
- {attribute}: {value} (confidence={c}, stability={s}, type={t}, scene={scene})
```

**vLLM configuration**:
- Model: Qwen3.5-9B (bf16)
- tensor_parallel=2, max_model_len=32768
- Server mode (one long-lived vLLM instance, HTTP API)

**Inherited from PersonaMem**: `extract_answer()` logic (regex-based option extraction + `<final_answer>` tag parsing).

## 4. Directory Layout

```
/root/autodl-tmp/wangqihao/Table3-data/
├── candidates/                          # Phase 1 output
│   └── {context_hash}/
│       ├── session_000.json
│       ├── session_001.json
│       └── ...
├── memory_snapshots/                    # Phase 2 output
│   └── {config_id}/
│       ├── {context_hash}.json
│       └── ...
├── eval_results/                        # Phase 3 output
│   └── {config_id}/
│       └── results.csv
├── scripts/
│   ├── phase1_extract_candidates.py     # Session detection + LLM extraction
│   ├── phase2_replay_evolution.py       # Memory replay with parameter sweeps
│   ├── phase3_eval_qa.py                # vLLM-based QA evaluation
│   └── summarize.py                     # Aggregate results across configs
└── configs/
    └── param_sweep.yaml                 # Parameter sweep definitions
```

## 5. Key Interfaces

### Phase 1 → Phase 2

```
# Phase 1 writes:
candidates/{context_hash}/session_{N:03d}.json = {
    "context_hash": str,
    "session_idx": int,
    "session_id": str,          # "{context_hash}_session_{N}"
    "scene": str,               # "session_{N}"
    "dialogue_text": str,
    "candidates": [              # Parsed ProfileCandidate dicts
        {
            "attribute": str,
            "value": str,
            "context": str,
            "profile_type": str,  # background|preference|goal|style|interest|general
            "scene": str,
            "confidence": float,
            "stability": float,
            "recency": float,
            "explicitness": float,
            "user_relevance": float,
            "source": str
        },
        ...
    ]
}
```

### Phase 2 → Phase 3

```
# Phase 2 writes:
memory_snapshots/{config_id}/{context_hash}.json = {
    "config_id": str,
    "context_hash": str,
    "num_sessions": int,
    "working_memory": [...],     # ProfileMemoryUnit dicts
    "short_term_memory": [...],
    "long_term_memory": [...]
}
```

Phase 3 reads this and formats the long-term memory (highest quality tier) into the context window. Working/short-term snapshots are also available if needed.

## 6. Evaluation Metrics

Per config, output:
- **Overall accuracy** (% correct across all 589 questions)
- **Per question_type accuracy** (7 types)
- **Per topic accuracy** (15 topics)
- **Per persona accuracy** (20 personas)

Final deliverable tables:
- **2a table**: write_threshold/promote_threshold × accuracy matrix
- **2b table**: decay λ scale × accuracy
- **2c table**: context_threshold × accuracy
- **Ablation**: No-TPPM (pure conversation) vs baseline TPPM vs best TPPM (optimal config per sub-experiment)

## 7. Implementation Units

### 7.1 `phase1_extract_candidates.py`
- Load `shared_contexts_32k.jsonl` with seek-index for memory efficiency
- For each context: split messages into sessions at `role=system` boundaries
- For each session: format dialogue → async DeepSeek API call → parse `ProfileCandidate` list
- Save per-session JSON files to `candidates/{context_hash}/`

### 7.2 `phase2_replay_evolution.py`
- Load parameter sweep definitions from `configs/param_sweep.yaml`
- For each config: create `TPMConfig`, iterate all (context, session) candidates in order
- Export final memory snapshot per config per context
- Support `--config-id` flag to run a single config (for parallel execution)

### 7.3 `phase3_eval_qa.py`
- vLLM server management (start if not running)
- Load QA CSV + shared contexts index + memory snapshots
- For each QA item: build 32K context window (conversation + TPPM memory + question)
- Query vLLM → extract answer → score
- Output per-config results CSV

### 7.4 `summarize.py`
- Aggregate all config results into summary tables
- Output LaTeX-ready tables for the paper

## 8. Dependencies & Reuse

| Component | Reuses |
|:---|:---|
| Phase 1 LLM extraction | `locomo_tppm_extract.py` patterns (DeepSeek API, retry logic, Candidate parsing) |
| Phase 2 memory replay | `TemporalProfileMemory` + `TPMConfig` from `Mini-Agent-5-1/mini_agent/tpm/` |
| Phase 3 QA evaluation | `inference_standalone_openai.py` → `extract_answer()`, `build_jsonl_index()`, `load_rows_with_context()` |
| Context window builder | New — 32K token budget management with conversation + memory |

## 9. Risks & Mitigations

| Risk | Mitigation |
|:---|:---|
| Phase 1 DeepSeek API cost/time | Cache all candidates; only re-extract if source data changes |
| 32K context overflows with long conversations + memory | Truncate conversation first, then fit memory; hard cap memory block at 2K tokens |
| vLLM OOM at 32K on 2×4090 | Set `--max-model-len 32768`, `--gpu-memory-utilization 0.95`, fallback to 24K if needed |
| Phase 2 compute time for many configs | Independent per config — run in parallel across tmux sessions |
