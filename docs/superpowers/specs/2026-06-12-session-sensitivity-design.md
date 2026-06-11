# Session Count Sensitivity Analysis — Design Spec

**Date**: 2026-06-12
**Status**: Draft
**Related**: TPPM-draft.tex §5.2.4 (会话数量的敏感性分析)

---

## 1. Goal

Quantify how TPPM's memory quality scales with the number of conversational sessions, producing a 1×3 panel figure for the paper that demonstrates TPPM's memory benefits accumulate with more sessions (vs. Long-Context saturation/degradation).

## 2. Dataset

**LoCoMo** (10 conversations, 19–32 sessions each, ~1,986 QA pairs).

Rationale (after systematic comparison with MemBench, LongMemEval, MSC):

| Factor | LoCoMo | MemBench | LongMemEval |
|--------|--------|----------|-------------|
| Effective N range | **1–20** (real content) | 1–8 (real), 9–50+ (noise only) | 1–5 |
| QA density | ~200/conv | 1/conv | Multiple |
| Cross-session QA | ✅ Multi-hop + Temporal | ❌ Single-session ref | ✅ |
| Evidence annotation | ✅ D1:3 (session-level) | ✅ target_step_id | ❓ |
| Temporal evaluation | ✅ 321 temporal Qs | ❌ | ✅ |

## 3. Experiment Design

### 3.1 Windowed Truncation

For each conversation, truncate to the first N sessions and evaluate:

- **N values**: {1, 3, 5, 7, 10, 15, 20}
- At N=20: 8/10 conversations qualify (conv-26/conv-30 have only 19 sessions → use N=19)
- 10 conversations averaged at each N, report mean ± SE

This design naturally controls for conversation-type confounds (same user, same topic across truncation points) — equivalent to the "within-window comparison" described in the paper.

### 3.2 QA Filtering (Critical for Fairness)

LoCoMo QA questions reference evidence via `D{s}:{m}` format (session s, message m). When truncating to N sessions, only questions whose evidence falls within sessions 1–N are answerable. Coverage varies dramatically:

| N | Answerable QA | Answerable Temporal QA |
|---|--------------|----------------------|
| 1 | 4% | 5% |
| 3 | 11% | 15% |
| 5 | 17% | 21% |
| 10 | 33% | 39% |
| 15 | 49% | 56% |
| 20 | 67% | 73% |

**Solution: Dual-metric reporting** to disentangle coverage from quality:

1. **Answerable F1**: F1 computed only on questions whose evidence ≤ N → measures memory quality on what the system *should* know (controls coverage confound)
2. **Overall F1**: F1 on a fixed question set (evidence ≤ 20), unanswerable questions scored 0 → measures total utility (naturally incorporates coverage)

Interpretation:
- If Answerable F1 also rises with N → strongest evidence: memory quality itself improves, not just coverage
- If Answerable F1 is flat but Overall F1 rises → weaker: gains only from more answerable questions

## 4. Evaluation Metrics

Three metrics, each reported with Answerable/Overall dual lines:

| Metric | Source | Direction | Rationale |
|--------|--------|-----------|-----------|
| **QA Overall F1** | LoCoMo official eval | ↑ | Basic memory utilization |
| **QA Temporal F1** | LoCoMo temporal subset (cat=2) | ↑↑ | Core differentiator — TPPM's temporal decay only shows value across many sessions |
| **Unsupported Profile Rate** | TPPM profile evaluation | ↓ | Contrasts with MemBench finding of "exponential hallucination growth" |

## 5. Pipeline (3 Phases)

### Phase 1: Session Truncation + TPPM Extraction

- Reuse `Table2-data/scripts/locomo_tppm_extract.py` (or `Figure-data/phase1_extract_locomo.py`)
- Add `--max-sessions N` parameter to truncate conversation before extraction
- Output: `session_sensitivity/extracted_profiles/{conv_id}_N{n}.json`

### Phase 2: QA Evaluation

- Reuse `Table2-data/scripts/locomo_qa_eval.py` with modifications:
  - `build_hybrid_context()`: add `max_sessions` parameter to truncate session list
  - Add QA filtering by evidence session number
  - Compute both Answerable F1 and Overall F1
  - Use LoCoMo official `eval_question_answering` for F1 computation
- Output: `session_sensitivity/eval_results/{conv_id}_N{n}_qa.json`

### Phase 3: Aggregation + Plotting

- Aggregate across 10 conversations per N value (mean ± SE)
- Generate 1×3 panel figure (PDF + PNG)
- Output: `session_sensitivity/figures/session_sensitivity.pdf`

## 6. Figure Design

**1×3 panel, each panel with dual lines (solid=Overall, dashed=Answerable)**:

```
┌──────────────────┬──────────────────┬──────────────────┐
│ (a) QA Overall   │ (b) QA Temporal  │ (c) Unsupported  │
│     F1 ↑         │     F1 ↑         │   Profile Rate ↓ │
│                  │                  │                  │
│  ─── Overall F1  │  ─── Overall F1  │  ─── UPR        │
│  - - Answerable  │  - - Answerable  │                  │
│      F1          │      F1          │                  │
│       └─────     │       └─────     │       └─────    │
│   1 3 5 7 10 20  │   1 3 5 7 10 20  │   1 3 5 7 10 20 │
└──────────────────┴──────────────────┴──────────────────┘
```

Design details:
- X-axis: Session Count N ({1,3,5,7,10,15,20}), non-uniform spacing with explicit labels
- Y-axis: metric range, auto-scaled per panel
- Error bands: mean ± SE (semi-transparent fill)
- Panel (c): lower is better (Unsupported Profile Rate decreases with N)
- Style consistent with existing `plot_decay_sensitivity.py`
- Output: vector PDF + raster PNG

## 7. Expected Results

| Metric | Overall F1 Trend | Answerable F1 Trend | Interpretation |
|--------|-----------------|---------------------|----------------|
| QA Overall F1 | ↑ steady rise | ↑ moderate rise | More sessions → more answerable + better memory utilization |
| QA Temporal F1 | ↑↑ significant rise | ↑ rise | Temporal decay mechanism needs many sessions to show value |
| Unsupported Profile Rate | ↓ decline | — | More evidence → less hallucination |

## 8. Paper Text Update

After experiment, replace TPPM-draft.tex lines 486–488 with:
1. Brief method description (LoCoMo windowed truncation + dual-metric design)
2. Key numerical results (Answerable F1 and Overall F1 at N=1 vs N=20)
3. Figure reference (Figure X)

## 9. Output Directory Structure

```
Figure-data/session_sensitivity/
├── phase1_extract.py          # Session truncation + TPPM extraction
├── phase2_eval_qa.py          # QA evaluation (dual-metric)
├── phase3_plot.py             # 3-panel figure generation
├── run_all.py                 # One-click runner
├── extracted_profiles/        # Phase 1 output
│   └── {conv_id}_N{n}.json
├── eval_results/              # Phase 2 output
│   └── {conv_id}_N{n}_qa.json
└── figures/                   # Phase 3 output
    ├── session_sensitivity.pdf
    └── session_sensitivity.png
```
